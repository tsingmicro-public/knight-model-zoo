#include "../include/pre_post_process.h"

#include <opencv2/opencv.hpp>
#include <ts_rne_log.h>
#include <fstream>
#include <iostream>
void write_opencv_mat_to_file(const cv::Mat &data, const char *fileName)
{
    int N = 1;  // Assuming a single batch (N=1)
    int C = data.channels();
    int H = data.rows;
    int W = data.cols;

    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file '%s'\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "SHAPE:(N:%d, C:%d, H:%d, W:%d)\n", N, C, H, W);

    for (int i = 0; i < C; ++i) {  // Loop over rows
        fprintf(file, "C[%d]:\n",i);
        for (int j = 0; j < H; ++j) {  // Loop over columns
            for (int k = 0; k < W; ++k) {  // Loop over channels
                // Access the pixel value
                uchar value = data.ptr<uchar>(j)[k * C + i];

                // Print the value with a space separator
                fprintf(file, "%d ", value);
            }
            fprintf(file, "\n");  // Newline after each row
        }
    }

    fclose(file);
}

cv::Size LetterBox(const cv::Mat& im, cv::Mat& padded,
                   const cv::Size& new_shape,
                   const cv::Scalar& color = cv::Scalar(114, 114, 114), bool auto_resize = false,
                   bool scale_fill = false, bool scale_up = false, int stride = 32)
{
    cv::Size shape = im.size();

    double r = std::min(static_cast<double>(new_shape.width) / shape.width,
                        static_cast<double>(new_shape.height) / shape.height);

    if (!scale_up) {
        r = std::min(r, 1.0);
    }

    cv::Size new_unpad = cv::Size(static_cast<int>(round(shape.width * r)),
                                  static_cast<int>(round(shape.height * r)));
    float dw = new_shape.width - new_unpad.width;
    float dh = new_shape.height - new_unpad.height;

    if (auto_resize) {
    } else if (scale_fill) {
        dw = 0;
        dh = 0;
        new_unpad = new_shape;
        r = new_shape.width / static_cast<double>(shape.width);
    }
    // dw /= 2;
    // dh /= 2;
    cv::Mat resized;
    cv::resize(im, resized, new_unpad, 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh-0.1));
    int bottom = int(round(dh+0.1));
    int left = int(round(dh-0.1));
    int right = int(round(dw+0.1));

    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, color);
    return cv::Size(dw, dh);
}

int TS_MPI_TRP_RNE_HWC_TO_CHW(cv::Mat &src_img, cv::Mat &dst_img)
{
    uint8_t *src = src_img.data;
    uint8_t *dst = dst_img.data;
    // #pragma omp parallel for collapse(3)
    for (int c = 0; c < dst_img.channels(); c++) {
        for (int h = 0; h < dst_img.rows; h++) {
            for (int w = 0; w < dst_img.cols; w++) {
                int src_idex = h * dst_img.cols * dst_img.channels() + w * dst_img.channels() +
                               c;
                int dst_idex = c * dst_img.rows * dst_img.cols + h * dst_img.cols + w;
                dst[dst_idex] = src[src_idex];
            }
        }
    }
    return 0;
}



cv::Size PreProcess(cv::Mat &src_img, cv::Mat &dst_img)
{
    cv::Mat &img = src_img;
    cv::Mat resizedImage;
    cv::Size targetSize(MODEL_INPUT_REQUIRED_W, MODEL_INPUT_REQUIRED_H);
    cv::Size ret_pad = LetterBox(img, resizedImage,targetSize);
    cv::Mat rgbImage = resizedImage.clone();
    dst_img = resizedImage;
    return ret_pad;
}
static int quick_sort_indice_inverse(float *input, int left, int right,
                                     int *indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;

    if (left < right) {
        key_index = indices[left];
        key = input[left];

        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];

            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;

        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

int TS_Yoloxs_Nms(float *torchCat, int len1, int len2, int *nmsOut,
                  float thresh, int maxDet, int scoreOffset)
{
    float tmpX, tmpY, tmpW, tmpH;

    float* pScore = (float *)(nmsOut + len1);
    int* box_idx = (int *)(pScore + len1);
    float* areas = (float*)(box_idx + len1);
    for (int i = 0; i < len1; i++) {
        box_idx[i] = i;
        pScore[i] = torchCat[len2 * i + scoreOffset];
        tmpX = torchCat[len2 * i + 0];
        tmpY = torchCat[len2 * i + 1];
        tmpW = torchCat[len2 * i + 2];
        tmpH = torchCat[len2 * i + 3];
        // areas[i] = tmpW * tmpH;
        torchCat[len2 * i + 0] = (tmpX - tmpW / 2) / 0.592593;
        torchCat[len2 * i + 1] = (tmpY - tmpH / 2) / 0.592593;
        torchCat[len2 * i + 2] = (tmpX + tmpW / 2) / 0.592593;
        torchCat[len2 * i + 3] = (tmpY + tmpH / 2) / 0.592593;
        // printf("x1 is %f,y1 is %f,x2 is %f,y2 is %f,score is %f\r\n",torchCat[len2 * i + 0], torchCat[len2 * i + 1],
        //                                                    torchCat[len2 * i + 2], torchCat[len2 * i + 3], pScore[i] = torchCat[len2 * i + scoreOffset]);
        ;
        areas[i] = (torchCat[len2 * i + 2] - torchCat[len2 * i + 0] + 1) *
                   (torchCat[len2 * i + 3] - torchCat[len2 * i + 1] + 0);
    }
    // quick_sort_indice_inverse(pScore, 0, len1-1, box_idx);
    //分数从大到小排序
    for (int i = 0; i < len1; i++) {
        float max = pScore[i];
        int idx_t = i;
        for (int j = i + 1; j < len1; j++) {
            if (max < pScore[j]) {
                max = pScore[j];
                idx_t = j;
            }
        }
        int tt = box_idx[i];
        box_idx[i] = box_idx[idx_t];
        box_idx[idx_t] = tt;
        float tt1 = pScore[i];
        pScore[i] = pScore[idx_t];
        pScore[idx_t] = tt1;
    }
    // for (int i=0;i<len1;i++){
    //     printf("score is %f,box_id is %d\r\n",pScore[i],box_idx[i]);
    // }
    int	cnt = 0;
    int sSzie = len1;
    while (sSzie > 0) {// torch.numel()返回张量元素个数
        if (sSzie == 1) {//    保留框只剩一个
            nmsOut[cnt] = box_idx[0];
            cnt++;
            break;
        } else {
            nmsOut[cnt] = box_idx[0];
            cnt++;
        }
        if (cnt >= maxDet) {
            break;
        }

        float *xx1 = (float *)(areas + len1);
        float *yy1 = (float *)(xx1 + sSzie);
        float *xx2 = (float *)(yy1 + sSzie);
        float *yy2 = (float *)(xx2 + sSzie);
        // 取第二个框的右下角角和第一个框的左上角，组成重叠坐标
        for (int i=1; i<sSzie; i++) {
            xx1[i-1] = MAX(torchCat[len2*box_idx[0] + 0],torchCat[len2*box_idx[i] + 0]);
            yy1[i-1] = MAX(torchCat[len2*box_idx[0] + 1],torchCat[len2*box_idx[i] + 1]);
            xx2[i-1] = MIN(torchCat[len2*box_idx[i] + 2],torchCat[len2*box_idx[0] + 2]);
            yy2[i-1] = MIN(torchCat[len2*box_idx[i] + 3],torchCat[len2*box_idx[0] + 3]);
            // if(box_idx[i] == 3663){
            //     printf("xx1 is %f,yy1 is %f,xx2 is %f,yy2 is %f\r\n",xx1[i-1], yy1[i-1], xx2[i-1], yy2[i-1]);
            // }
        }
        // 计算重叠面积
        for (int i=0; i<sSzie-1; i++) {
            xx1[i] = MAX(0, xx2[i] - xx1[i] + 1);
            yy1[i] = MAX(0, yy2[i] - yy1[i] + 1);
            xx2[i] = xx1[i] * yy1[i];
        }

        int idx_len = 0;
        int *newbox_idx = (int *)yy2;
        for (int i = 1; i < sSzie; i++) {
            xx1[i - 1] = areas[box_idx[i]] + areas[box_idx[0]] - xx2[i - 1];
            yy1[i - 1] = xx2[i - 1] / xx1[i - 1];
            // if(box_idx[i] == 3663){
            //     printf("xx1 is %f, yy1 is %f\r\n", xx1[i-1],yy1[i-1]);
            // }
            if (yy1[i - 1] < thresh) {
                // printf("i is %d,box_idx is %d\r\n", i, box_idx[i]);
                newbox_idx[idx_len] = box_idx[i];
                idx_len++;
            }
        }

        if (idx_len == 0) {
            break;
        }
        memcpy(box_idx, newbox_idx, idx_len * sizeof(int));
        sSzie = idx_len;
        // printf("sSize is %d\r\n",sSzie);
    }
    return cnt;
}

void TS_Yoloxs_PostProcess(RNE_BLOBS_S *debug,
                           YOLOXS_DET_RESULT_S *yolo_out, TS_FLOAT confThres,
                           TS_FLOAT iouThres, TS_FLOAT *pPostbuf, cv::Size pad_size,
                           std::string outFilePath, cv::Mat &dst_img)
{
    float *dataVec = (float*)pPostbuf;
    int data_len = 0;
    int idxtmp = 0;
    int chn_num = 6;
    for (TS_U32 idx = 0; idx < debug->u32NBlob; idx++) {
        TS_S32 h = debug->stpBlob[idx].s32H;
        TS_S32 w = debug->stpBlob[idx].s32W;
        TS_S32 c = debug->stpBlob[idx].s32C;
        bool is_unsign = debug->stpBlob[idx].bIsUnsigned;
        TS_S32 cStride = TS_MPI_TRP_RNE_CStride(c, debug->stpBlob[idx].s32BitNum,
                                                debug->stpBlob[idx].bIsJoined);

        TS_S32 c_out=(debug->stpBlob[idx].eFormat==RNE_BLOB_N_H_W_C)? c:cStride;
        TS_S32 uSize = (debug->stpBlob[idx].s32BitNum / CHAR_BIT);
        TS_U8 *out = (TS_U8 *)(debug->stpBlob[idx].vpAddr);
        float coeff = *(debug->stpBlob[idx].fCoeff);
        TS_S32 lenStride = c /
                           1; // yolov5 demo模型c为255，c / ANCHORS_NUMBER = 85(cx, cy, cwidth, cheight, score + 80个class)
        int count=0;
        for(TS_S32 n = 0; n < BATCH_SIZE; n++) {
            for (TS_S32 i = 0; i < h; i++) {
                for (TS_S32 j = 0; j < w; j++) {
                    for(TS_S32 l = 0; l < 1; l++) {
                        // cv::Rect2i rect;
                        float cx = 0.0, cy=0.0, cwidth=0.0, cheight=0.0;
                        int max_class_id = 0;
                        // float scores = 0.0;
                        float max_class_score = 0.0;

                        TS_S32 *d = (TS_S32 *)(out + (((j + i * w) * c_out + l*lenStride+4)
                                                      *uSize)); // 跳过cx，cy，cwidth，cheight 4个位置，解析score
                        // 解析结果数据类型(有无符号，位宽)
                        TS_S32 data = *d & MASK_BITS(debug->stpBlob[idx].s32BitNum);
                        TS_S32 bitNum = (debug->stpBlob[idx].s32BitNum - 1);
                        TS_S32 bitValue = MASK_BITS(bitNum);
                        if (!is_unsign && data > bitValue) {
                            bitNum = (debug->stpBlob[idx].s32BitNum);
                            bitValue = MASK_BITS(bitNum);
                            data = data - (bitValue + 1);
                        }

                        // Score反量化为浮点
                        count++;
                        for (TS_S32 k = 0; k < lenStride; k++) {
                            // 解析cx, cy, cwidth, cheight
                            TS_S32 *d = (TS_S32 *)(out + (((j + i * w) * c_out + l*lenStride+k) *uSize));
                            TS_S32 data = *d & MASK_BITS(debug->stpBlob[idx].s32BitNum);
                            TS_S32 bitNum = (debug->stpBlob[idx].s32BitNum - 1);
                            TS_S32 bitValue = MASK_BITS(bitNum);
                            if (!is_unsign && data > bitValue) {
                                bitNum = (debug->stpBlob[idx].s32BitNum);
                                bitValue = MASK_BITS(bitNum);
                                data = data - (bitValue + 1);
                            }

                            // cx, cy, cwidth, cheight 反量化为浮点
                            float f_data =  data * coeff;
                            // f_data = Sigmoid(f_data);
                            // yoloxs模型边界框回归，不同检测模型计算公式不同，可能需要修改
                            if(k==0) { // cx
                                cx = (f_data + j) * strides[idx];
                                continue;
                            } else if(k==1) { // cy
                                cy = (f_data + i) * strides[idx];
                                continue;
                            } else if(k==2) { // cwidth
                                cwidth = exp(f_data) * strides[idx];
                                continue;
                            } else if(k==3) { // cheight
                                cheight = exp(f_data) * strides[idx];
                                continue;
                            } else if(k==4) {
                                // conf= f_data;
                                // max_class_score = f_data;
                                continue;
                            } else {
                                if (max_class_score < f_data) {
                                    max_class_id = k;
                                    max_class_score = f_data;
                                }
                            }
                        }
                        if (max_class_score > 0.15)
                            // 中心点坐标转化成左上角
                        {
                            float x1 = cx - cwidth / 2.0f ;
                            float y1 =  cy - cheight / 2.0f;
                            float w = cwidth;
                            float h = cheight;
                            dataVec[idxtmp+0] = x1 ;
                            dataVec[idxtmp+1] = y1 ;
                            dataVec[idxtmp+2] = w ;
                            dataVec[idxtmp+3] = h ;
                            dataVec[idxtmp + 4] = max_class_score;
                            dataVec[idxtmp + 5] = max_class_id - 5;

                            idxtmp += chn_num;
                            data_len++;

                        }
                    }
                }
            }
            out += h * w * cStride * uSize;
        }
    }
    const int maxDet = 1000;       // maximum detections per image
    int *nmsOut = (int *)(dataVec + data_len*6);
    int outNum = TS_Yoloxs_Nms(dataVec, data_len, 6, nmsOut, iouThres, maxDet,
                               4);  // NMS

    yolo_out->u32ObjNum = MIN(outNum, MAX_YOLOXS_DET_NUM);
    int color_rgb[30][3]= {{0, 255, 0},   // 绿色
        {255, 0, 0},   // 蓝色
        {0, 0, 255},   // 红色
        {255, 255, 0}, // 黄色
        {0, 255, 255}, // 青色
        {255, 0, 255}
    };
    // double r = std::min(static_cast<double>(MODEL_INPUT_REQUIRED_W) /dst_img.cols,
    //                     static_cast<double>(MODEL_INPUT_REQUIRED_H) /dst_img.rows );
    for(unsigned int i = 0; i < yolo_out->u32ObjNum; i++) {
        int nmsIdx = nmsOut[i];

        if(dataVec[nmsIdx * chn_num + 4] > 0.9) {
            yolo_out->stBox[i].f32Xmin = dataVec[nmsIdx * chn_num + 0];
            yolo_out->stBox[i].f32Ymin = dataVec[nmsIdx * chn_num + 1];
            yolo_out->stBox[i].f32Xmax = dataVec[nmsIdx * chn_num + 2];
            yolo_out->stBox[i].f32Ymax = dataVec[nmsIdx * chn_num + 3];

            float x1 = (yolo_out->stBox[i].f32Xmin + yolo_out->stBox[i].f32Xmax) / 2.0;
            float y1 = (yolo_out->stBox[i].f32Ymin + yolo_out->stBox[i].f32Ymax) / 2.0;

            float w = (yolo_out->stBox[i].f32Xmax - yolo_out->stBox[i].f32Xmin);
            float h = (yolo_out->stBox[i].f32Ymax - yolo_out->stBox[i].f32Ymin);

            yolo_out->stBox[i].f32Score = dataVec[nmsIdx * chn_num + 4];
            yolo_out->u8ObjNo[i] = dataVec[nmsIdx * chn_num + 5];

            TS_S32 X1 = (x1) ;
            TS_S32 Y1 = (y1) ;
            TS_S32  X2 = x1 + w;
            TS_S32 Y2 = y1 + h;
            X2= (X2);
            Y2 = (Y2);
            X1 = std::max(0, int(X1));
            Y1 = std::max(0, int(Y1));
            X2 = std::min(int(X2), int(dst_img.cols));
            Y2 = std::min(int(Y2), int(dst_img.rows));

            TS_S32 W = X2-X1;
            TS_S32 H = Y2-Y1;
            cv::Rect rect(X1, Y1, W, H);
            printf("draw box       %d:x1:%d, y1:%d, w:%d, h:%d score:%.2f, class_id:%d\r\n",
                   i, X1, Y1, X2, Y2, yolo_out->stBox[i].f32Score,
                   yolo_out->u8ObjNo[i]);


            int color_idx = yolo_out->u8ObjNo[i] % 6;
            cv::Scalar rectColor(color_rgb[color_idx][0],color_rgb[color_idx][1],
                                 color_rgb[color_idx][2]);
            cv::rectangle(dst_img, rect, rectColor,
                          3); // 参数3:框的颜色，参数4:框的粗细

            // 将score转化为字符串，保留2位小数并标注到图片
            std::string str;
            char strConf[20];
            memset(strConf,0,sizeof(strConf));
            sprintf(strConf,"%.2f", yolo_out->stBox[i].f32Score);
            str = strConf;
            std::string confidenceText = class_name[yolo_out->u8ObjNo[i]] + " " + strConf;
            cv::Point textPosition(X1,
                                   Y1 - 5);  // 设置标注内容距离框5个像素
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6; // 字体大小
            //BGR
            cv::Scalar fontColor(0, 255, 0);

            int thickness =1; // 字体粗细
            cv::putText(dst_img, confidenceText, textPosition, fontFace, fontScale,
                        fontColor, thickness);
        }
    }
    cv::imwrite(outFilePath, dst_img);
    TS_MPI_TRP_RNE_Info("save processed img: %s\n", outFilePath.c_str());

}

