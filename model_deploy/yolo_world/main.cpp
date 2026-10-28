#include <string.h>
#include <fstream>
#include <time.h>

#include "ts_rne_c_api.h"
#include "ts_rne_log.h"
#include "ts_rne_nn_input.h"
#include "ts_rne_time.h"
#include "ts_rne_version.h"
#include "ts_rne_gp_layers.h"
#include <ts_rne_record_file.h>

#include "../../../open_source/common/utils.h"
#include "ts_rne_get_file_data.h"
#include "ts_rne_get_input_type.h"

static inline TS_FLOAT TS_MPI_TRP_RNE_GetUnalignedAccess(TS_FLOAT *src)
{
    float data = 0;
    memcpy(&data, src, sizeof(TS_FLOAT));
    return data;
}

static inline TS_FLOAT TS_MPI_TRP_RNE_GetAlignedAccess(TS_FLOAT *src)
{
    return *src;
}

union RNE_NN_INPUT_U {
    TS_FLOAT fdata;
    TS_S32 data;
};

TS_S32 TS_MPI_TRP_RNE_FillInputBlobs_YoloWorld(RNE_NET_S *net, TS_S32 coreIdx,
        TS_VOID *input)
{
    TS_U8 *src_ptr = (TS_U8 *)(input);
    RNE_BLOBS_S* blobs = TS_MPI_TRP_RNE_GetInputBlobs(net);
    if (NULL == blobs) {
        TS_MPI_TRP_RNE_Error("input blobs is NULL!\n");
        return -1;
    }

    TS_FLOAT (*get_float_fuc[2])(TS_FLOAT *) = {TS_MPI_TRP_RNE_GetAlignedAccess, TS_MPI_TRP_RNE_GetUnalignedAccess};
    for (uint32_t idx_blob = 0; idx_blob < blobs->u32NBlob; ++idx_blob) {
        RNE_BLOB_S *blob = &blobs->stpBlob[idx_blob];
        TS_S32 n = blob->s32N;
        TS_S32 w = blob->s32W;
        TS_S32 h = blob->s32H;
        TS_S32 c = blob->s32C;
        TS_S32 q_scale = *blob->s32pQ;
        TS_S32 scale_int = *blob->fCoeff * (1LL << q_scale);
        TS_S32 c_stride = TS_MPI_TRP_RNE_GetBlobCStride(blob);
        TS_S32 unit_size = blob->s32BitNum / CHAR_BIT;
        TS_S32 bits = blob->bIsUnsigned ? blob->s32BitNum + 1 : blob->s32BitNum;
        TS_S64 data_max = ~MIN_BIT(bits - 1);
        TS_S32 sign_bit = sizeof(TS_S64) * CHAR_BIT - blob->s32BitNum;
        TS_S64 data_min = ((TS_S64)(MIN_BIT(bits - 1) & MASK_BITS(
                                        blob->s32BitNum)) << sign_bit) >> sign_bit;
        TS_S64 scale_max = ~MIN_BIT(16 - 1);
        TS_S64 scale_min = MIN_BIT(16 - 1);
        TS_S64 scale64_max = ~MIN_BIT(64 - 1);
        TS_S64 scale64_min = MIN_BIT(64 - 1);
        scale_int = scale_int > scale_max ? scale_max : scale_int;
        scale_int = scale_int < scale_min ? scale_min : scale_int;

        TS_U8 *dst = (TS_U8 *)(blob->vpAddr);
        TS_S32 in_type = net->eInputType;


        if (idx_blob == 0) {
            in_type = RNE_NET_INPUT_TYPE_INT_HWC;
        } else {
            in_type = RNE_NET_INPUT_TYPE_FLOAT_CHW;
        }

        switch (in_type) {
        case RNE_NET_INPUT_TYPE_FLOAT_CHW: {
            TS_FLOAT *src = (TS_FLOAT *)src_ptr;
            int access_index = (((size_t)src) / sizeof(float) == 0)? 0 : 1;
            if (q_scale == -1) {
                printf("q_scale=-1 is idx=%d\r\n",idx_blob);
                TS_FLOAT *dst_float = (TS_FLOAT *)dst;
                TS_S32 idx = 0;
                for (TS_S32 bat = 0; bat < n; ++bat) {
                    for (TS_S32 i = 0; i < h; ++i) {
                        for (TS_S32 j = 0; j < w; ++j) {
                            for (TS_S32 k = 0; k < c; ++k) {
                                dst_float[idx++] = get_float_fuc[access_index](&src[(k * h + i) * w + j]);
                            }
                        }
                    }
                    src += h * w * c;
                }
            } else {
                for (TS_S32 bat = 0; bat < n; bat++) {
                    for (TS_S32 i = 0; i < h; ++i) {
                        for (TS_S32 j = 0; j < w; ++j) {
                            for (TS_S32 k = 0; k < c; ++k) {
                                TS_S64 data = 0;
                                union RNE_NN_INPUT_U tmp1;
                                tmp1.fdata = get_float_fuc[access_index](&src[(k * h + i) * w + j]);
                                TS_DOUBLE tmp = (TS_DOUBLE)(tmp1.fdata) * scale_int;
                                tmp = tmp > scale64_max ? scale64_max : tmp;
                                tmp = tmp < scale64_min ? scale64_min : tmp;
                                if (blob->s32BitNum == 8) {
                                    tmp /= (1 << q_scale);
                                    data = ((tmp1.data | (1 << 30)) >> 30) * 0.5f + tmp;
                                } else {
                                    data = tmp / (1 << q_scale);
                                }
                                data = data > data_max ? data_max : data;
                                data = data < data_min ? data_min : data;
                                for (TS_S32 u = 0; u < unit_size; ++u) {
                                    dst[k * unit_size + u] = data >> (u * CHAR_BIT);
                                }
                            }
                            dst += c_stride * unit_size;
                        }
                    }
                    dst = (TS_U8 *)(((TS_SIZE_T)dst + BLOB_ALIGN_BYTES - 1) & (~((
                                        TS_SIZE_T)BLOB_ALIGN_BYTES - 1)));
                    src += h * w * c;
                }
            }
            src_ptr += n * h * w * c * sizeof(TS_FLOAT);
            break;
        }
        case RNE_NET_INPUT_TYPE_INT_HWC: {
            TS_U8 *src = (TS_U8 *)src_ptr;
            for (TS_S32 bat = 0; bat < n; bat++) {
                for (TS_S32 i = 0; i < h * w; ++i) {
                    memcpy(dst, src, c * unit_size);
                    src += c * unit_size;
                    dst += c_stride * unit_size;
                }
                dst = (TS_U8 *)(((TS_SIZE_T)dst + BLOB_ALIGN_BYTES - 1) & (~((
                                    TS_SIZE_T)BLOB_ALIGN_BYTES - 1)));
            }
            src_ptr += n * h * w * c * unit_size;
            break;
        }
        case RNE_NET_INPUT_TYPE_INT_CHW: {
            TS_U8 *src = (TS_U8 *)src_ptr;
            for (TS_S32 bat = 0; bat < n; bat++) {
                for (TS_S32 i = 0; i < h; ++i) {
                    for (TS_S32 j = 0; j < w; ++j) {
                        for (TS_S32 k = 0; k < c; ++k) {
                            for (TS_S32 u = 0; u < unit_size; ++u) {
                                TS_S32 src_idx = ((k * h + i) * w + j) * unit_size + u;
                                dst[k * unit_size + u] = src[src_idx];
                            }
                        }
                        dst += c_stride * unit_size;
                    }
                }
                dst = (TS_U8 *)(((TS_SIZE_T)dst + BLOB_ALIGN_BYTES - 1) & (~((
                                    TS_SIZE_T)BLOB_ALIGN_BYTES - 1)));
                src += h * w * c * unit_size;
            }
            src_ptr += n * h * w * c * unit_size;
            break;
        }
        default:
            TS_MPI_TRP_RNE_Error("unsupport inputype:%d\n", net->eInputType);
            return -1;
            break;
        }
    }
    return 0;
}
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
    const char *modelFile = argv[1];
    const char *src_img_ptr = argv[2];
    const char *textDataFile = argv[3];
    // const char *dst_img_ptr = argv[4];
    char dst_img_ptr[200];
    RNE_TYPE_NAME_E chip_type_e;
    chip_type_e = RNE_TYPE_NAME_TX5336AV200;
    TS_MPI_TRP_RNE_InitResourceByType(chip_type_e);
    TS_MPI_TRP_RNE_SetLogLevel(RNE_LOG_INFO);

    TS_MPI_TRP_RNE_Info("current log level: %d\n", TS_MPI_TRP_RNE_GetLogLevel());
    TS_MPI_TRP_RNE_Info("current lib version: %s\n",
                        TS_MPI_TRP_RNE_GetSdkVersion());
    TS_MPI_TRP_RNE_Info("main start...\n");

    memset(dst_img_ptr,0,sizeof(dst_img_ptr));
    sprintf(dst_img_ptr,"%sresult.jpg","resource/");
    int ret = 0;
    if (argc != 4) {
        TS_MPI_TRP_RNE_Warn("Need 3 input Args, but get %d\n",argc);
        for(int i=0; i<argc; i++)
            printf("--->argc(%d) %d:%s\n",argc,i,argv[i]);
        return -1;
    }
    // 读取原始图片
    cv::Mat src_img = cv::imread(src_img_ptr);
    if (src_img.empty()) {
        TS_MPI_TRP_RNE_Info("input is error\n");
        return -1;
    }
    TS_MPI_TRP_RNE_Info("load input success\n");

    cv::Mat pre_img;
    cv::Mat end_img;
    cv::Size padded_size = ts_yoloworld_preprocess(src_img, pre_img);
    int *modelData = NULL;
    int *inputData = NULL;

    // 读取model文件数据
    if ((ret = getFileData(modelFile, &modelData)) == -1) {
        return -1;
    }

    // 读取输入数据文件
    if ((ret = getFileData(textDataFile, &inputData)) == -1) {
        return -1;
    }
    RNE_BLOBS_VIEW_S *input_view;
    // 推理
    RNE_NET_S nnModel;
    memset(&nnModel, 0, sizeof(RNE_NET_S));
    nnModel.u8pGraph = (TS_U8 *)modelData;
    nnModel.s32GraphLen = getFileSize(modelFile);
    nnModel.eInputType = RNE_NET_INPUT_TYPE_FLOAT_HWC;
    RNE_NET_S *net[] = {&nnModel};
    const TS_S32 num = sizeof(net) / sizeof(net[0]);
    TS_U8 *paramStride[num];
    memset(paramStride, 0, sizeof(paramStride));
    // 打开设备并注册 GP 层
    if ((ret = TS_MPI_TRP_RNE_OpenDevice(NULL,
                                         TS_MPI_TRP_RNE_RegisterGpLayers)) != 0) {
        goto cleanup_device;
    }

    // 加载模型
    if ((ret = TS_MPI_TRP_RNE_LoadModel(net[0])) != 0) {
        goto cleanup_model;
    }

    // 加载模型数据
    if ((ret = TS_MPI_TRP_RNE_OnceLoad(net[0])) != 0) {
        goto cleanup_model;
    }
    input_view = TS_MPI_TRP_RNE_GetInputBlobsView(net[0]);
    for (TS_U32 i=0; i<input_view->n_blob; i++) {
        printf("blob[%d]name=%s bit_num=%d\n", i, input_view->blob_view[i].name,
               input_view->blob_view[i].bit_num);
    }

    for (TS_S32 n = 0; n < num; ++n) {
        TS_U8 *input1 = NULL;
        // TS_S32 pic_size = u32picSize;
        size_t total_elements_1 = pre_img.total();  // 总元素数量
        size_t element_size = pre_img.elemSize();  // 每个元素的字节大小
        size_t pic_size = total_elements_1 * element_size;  // 总内存大小
        TS_S32 txt_size = input_view->blob_view[1].n * input_view->blob_view[1].c *
                          input_view->blob_view[1].h
                          * input_view->blob_view[1].w * (input_view->blob_view[1].bit_num /CHAR_BIT);
        TS_S32 input1_size = pic_size + txt_size;
        input1 = (TS_U8 *)malloc(input1_size);
        memcpy(input1, pre_img.data, pic_size);
        memcpy(input1 + pic_size, inputData, txt_size);

        if (0 != TS_MPI_TRP_RNE_FillInputBlobs_YoloWorld(net[n], 0, input1)) {
            TS_MPI_TRP_RNE_Error("fill inputBlobs error!\n");
            return -1;
        }
        RNE_BLOBS_S *outputBlobs = TS_MPI_TRP_RNE_Forward(net[n]);
        if (outputBlobs == NULL) {
            TS_MPI_TRP_RNE_Error("net forward error!\n");
            goto cleanup_model;
        }
        int bit_num = 8;
        RNE_BLOBS_VIEW_S *outputBlobsView = TS_MPI_TRP_RNE_GetOutputBlobsView(net[n]);
        if (outputBlobsView && outputBlobsView->blob_view)
            bit_num = outputBlobsView->blob_view->bit_num;

        // 记录输出结果到文件
        RNE_RecordFile(outputBlobs, (TS_CHAR *)"./", NULL, 0, 1, outputBlobsView);

        TS_MPI_TRP_RNE_ReleaseBlobsView(outputBlobsView);
        free(input1);
        // 后处理
        TS_SIZE_T sLen = 0;
        for (TS_U32 i = 0; i < outputBlobs->u32NBlob; ++i) {
            sLen += outputBlobs->stpBlob[i].u32Size;
        }
        TS_FLOAT *out = (TS_FLOAT *)malloc(sLen * 6);
        RNE_SDK_OBJECT_RESULT_S *yolo_out;
        yolo_out = (RNE_SDK_OBJECT_RESULT_S *)malloc(sizeof(
                       RNE_SDK_OBJECT_RESULT_S));
        if (yolo_out == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        memset(yolo_out, 0, sizeof(RNE_SDK_OBJECT_RESULT_S));
        TS_YoloWorld_PostProcess(outputBlobs,yolo_out, 0.45, 0.45, out,
                                 padded_size, std::string(dst_img_ptr), src_img, bit_num);
        free(out);
        free(yolo_out);
    }

    TS_MPI_TRP_RNE_ReleaseBlobsView(input_view);
cleanup_model:
    // 卸载模型
    TS_MPI_TRP_RNE_UnloadModel(net[0]);
cleanup_device:
    // 关闭设备
    TS_MPI_TRP_RNE_CloseDevice();

    // 释放内存
    TS_MPI_TRP_RNE_Free(modelData);
    TS_MPI_TRP_RNE_Free(inputData);
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

