#include <dirent.h>
#include <string.h>
#include <fstream>
#include <time.h>

#include "ts_rne_c_api.h"
#include "ts_rne_log.h"
#include "ts_rne_nn_input.h"
#include "ts_rne_time.h"
#include "ts_rne_version.h"
#include "ts_rne_gp_layers.h"
#include "ts_rne_record_file.h"
#include "utils.h"

TS_S32 FillInputBlobs_BGR_OR_RGB_UINT8(RNE_NET_S *net,
                                       TS_S32 coreIdx, TS_VOID *input);
TS_S32 InitNetsGraphAndParams(RNE_NET_S **net,
                              TS_U8 **paramStride,
                              TS_S32 nNet)
{
    TS_S32 ret = TS_MPI_TRP_RNE_OpenDevice(NULL, TS_MPI_TRP_RNE_RegisterGpLayers);
    if (ret) {
        TS_MPI_TRP_RNE_Error("open device error!\n");
        return ret;
    }

    for (TS_S32 i = 0; i < nNet; ++i) {
        ret = TS_MPI_TRP_RNE_LoadModel(net[i]);
        if (ret) {
            TS_MPI_TRP_RNE_Error("load model error!\n");
            return ret;
        }
        ret = TS_MPI_TRP_RNE_OnceLoad(net[i]);
        if (ret) {
            TS_MPI_TRP_RNE_Error("once load error!\n");
            return ret;
        }
    }

    return ret;
}

TS_S32 main(int argc, char* argv[])
{
    if (argc != 3) {
        TS_MPI_TRP_RNE_Warn("Need 3 input Args, but get %d\n",argc);
        for(int i=0; i<argc; i++)
            printf("--->argc(%d) %d:%s\n",argc,i,argv[i]);
        return -1;
    }
    TS_MPI_TRP_RNE_ResetDevice();
    std::string model_file(argv[1]);
    std::string src_img_dir(argv[2]);
    struct dirent64 *entry = NULL;
    DIR *dp = opendir(src_img_dir.c_str());
    if (dp == NULL) {
        TS_MPI_TRP_RNE_Info("opendir %s error\n", src_img_dir.c_str());
        return -1;
    }

    RNE_TYPE_NAME_E chip_type_e = RNE_TYPE_NAME_TX5326DV500;
    TS_MPI_TRP_RNE_InitResourceByType(chip_type_e);
    TS_MPI_TRP_RNE_SetLogLevel(RNE_LOG_INFO);
    TS_MPI_TRP_RNE_Info("current log level: %d\n", TS_MPI_TRP_RNE_GetLogLevel());
    TS_MPI_TRP_RNE_Info("current lib version: %s\n",
                        TS_MPI_TRP_RNE_GetSdkVersion());
    TS_MPI_TRP_RNE_Info("main start...\n");

    // 加载部署资源
    FILE *modelFd = fopen(model_file.c_str(), "rb");
    if (NULL == modelFd) {
        closedir(dp);
        dp = NULL;
        TS_MPI_TRP_RNE_Error("Failed to open model\n");
        return -1;
    }
    int modelSize = GetFileSize(modelFd);
    TS_U8 *p_modelBuffer = (TS_U8 *)calloc(modelSize + 1, 1);
    if (p_modelBuffer == NULL) {
        TS_MPI_TRP_RNE_Error("Failed to alloc model_cfg\n");
        fclose(modelFd);
        modelFd = NULL;
        closedir(dp);
        dp = NULL;
        return -1;
    }
    fread(p_modelBuffer, 1, modelSize, modelFd);
    TS_MPI_TRP_RNE_Info("load model of net success\n");

    // 初始化
    RNE_NET_S nnModel;
    memset(&nnModel, 0, sizeof(RNE_NET_S));
    nnModel.u8pGraph = p_modelBuffer;
    nnModel.s32GraphLen = modelSize;
    nnModel.eInputType = RNE_NET_INPUT_TYPE_INT_HWC;
    RNE_NET_S *net[] = {&nnModel};
    const TS_S32 num = sizeof(net) / sizeof(net[0]);
    TS_U8 *paramStride[num];
    memset(paramStride, 0, sizeof(paramStride));
    if (0 != InitNetsGraphAndParams(net, paramStride, num)) {
        free(p_modelBuffer);
        p_modelBuffer = NULL;
        fclose(modelFd);
        modelFd = NULL;
        closedir(dp);
        dp = NULL;
        TS_MPI_TRP_RNE_Error("nets init error!\n");
        return -1;
    }

    RNE_SDK_OBJECT_RESULT_S *nms_out;
    nms_out = (RNE_SDK_OBJECT_RESULT_S *)malloc(sizeof(RNE_SDK_OBJECT_RESULT_S));
    if (nms_out == NULL) {
        TS_MPI_TRP_RNE_Error("Memory allocation failed\n");
        return 1;
    }

#if defined(SAVE_PRED_JSON)
    std::vector<DetectionResult> results_;
    results_.clear();
    results_.shrink_to_fit();  // 释放内存
#endif

    RNE_BLOBS_VIEW_S *outputBlobsView = TS_MPI_TRP_RNE_GetOutputBlobsView(net[0]);
    int process_img_count = 0;
    while ((entry = readdir64(dp))) {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            std::string str_src = src_img_dir + "/" + std::string(entry->d_name);
            int image_id = extract_number_regex(std::string(entry->d_name));
            process_img_count ++;
            TS_MPI_TRP_RNE_Info("%s image_id %d, process_jpg_count:%d\n", entry->d_name,
                                image_id, process_img_count);

            // 读取原始图片进行前处理
            cv::Mat src_img = cv::imread(str_src.data());
            if (src_img.empty()) {
                TS_MPI_TRP_RNE_Info("input is error\n");
                return -1;
            }
            LETTERBOX_T pad_value;
            cv::Mat pre_img;
            PreProcess(src_img, pre_img, &pad_value);
            FillInputBlobs_BGR_OR_RGB_UINT8(net[0], 0, pre_img.data);

#ifdef LOG_TIME
            TS_U64 start = TS_MPI_TRP_RNE_GetSysTimeMs();
#endif
            RNE_BLOBS_S *outputBlobs = TS_MPI_TRP_RNE_Forward(net[0]);
#ifdef LOG_TIME
            TS_U64 end = TS_MPI_TRP_RNE_GetSysTimeMs();
            TS_MPI_TRP_RNE_Info("The inference time: %llu ms\n", end - start);
#endif
            // 记录输出结果到文件
            RNE_RecordFile(outputBlobs, (TS_CHAR *)"./", NULL, 0, 1, outputBlobsView);

            ts_yolov5_post_process(outputBlobs, NMS_SCORE_THRESHOLD, NMS_THRESHOLD,
                                   outputBlobsView->blob_view->bit_num, pad_value, nms_out);

            save_result_img(nms_out, src_img);
#if defined(SAVE_PRED_JSON)
            WriteOneJSON(nms_out, image_id, results_);
#endif
        }
    }
#if defined(SAVE_PRED_JSON)
    SaveJSONsToFile(results_);
#endif

    TS_MPI_TRP_RNE_ReleaseBlobsView(outputBlobsView);
    free(nms_out);
    nms_out = NULL;
    TS_MPI_TRP_RNE_Free(paramStride[0]);
    TS_MPI_TRP_RNE_UnloadModel(net[0]);
    TS_MPI_TRP_RNE_CloseDevice();
    if (p_modelBuffer != NULL) {
        free(p_modelBuffer);
        p_modelBuffer = NULL;
    }
    if (modelFd != NULL) {
        fclose(modelFd);
        modelFd = NULL;
    }
    closedir(dp);
    dp = NULL;

    TS_MPI_TRP_RNE_Info("program finished...\n");
    return 0;
}

#ifndef BLOB_ALIGN_BYTES
#define BLOB_ALIGN_BYTES (32 / 8)
#endif

TS_S32 FillInputBlobs_BGR_OR_RGB_UINT8(RNE_NET_S *net,
                                       TS_S32 coreIdx, TS_VOID *input)
{
    RNE_BLOBS_S *blobs = TS_MPI_TRP_RNE_GetInputBlobs(net);
    TS_U8 *src_ptr = (TS_U8 *)(input);
    for (uint32_t idx_blob = 0; idx_blob < blobs->u32NBlob; ++idx_blob) {
        RNE_BLOB_S *blob = &blobs->stpBlob[idx_blob];
        TS_S32 n = blob->s32N;
        TS_S32 w = blob->s32W;
        TS_S32 h = blob->s32H;
        TS_S32 c = blob->s32C;
        TS_S32 c_stride = TS_MPI_TRP_RNE_CStride(blob->s32C, blob->s32BitNum,
                          blob->bIsJoined);
        TS_U8 *dst = (TS_U8 *)(blob->vpAddr);
        TS_U8 *src = (TS_U8 *)src_ptr;
        for (TS_S32 bat = 0; bat < n; bat++) {
            for (TS_S32 i = 0; i < h * w; ++i) {
                memcpy(dst, src, c );
                *(dst + 0) = *(src + 2);
                *(dst + 1) = *(src + 1);
                *(dst + 2) = *(src + 0);
                src += c;
                dst += c_stride;
            }
            dst = (TS_U8 *)(((TS_SIZE_T)dst + BLOB_ALIGN_BYTES - 1) & (~((
                                TS_SIZE_T)BLOB_ALIGN_BYTES - 1)));
        }
        src_ptr += n * h * w * c;
    }
    return 0;
}

