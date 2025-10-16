
#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include <string>
#include "types/yolo_datatype.h"

#define pi 3.14159265358979323846

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum);

typedef struct
{
    int classId;
    float score;
    float x;
    float y;
    float w;
    float h;
    float angle;
} CSXYWHR;

namespace yolo
{

    // 分类识别
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects);                                                               // 浮点数版本
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects); // int8版本

    // obb使用
    int GetConvDetectionObbResult(float **pBlob, std::vector<float> &DetectiontRects);

    int GetConvDetectionObbResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

    // pose使用
    int GetConvDetectionPoseResult(float **pBlob, std::vector<float> &DetectiontRects,std::vector<std::map<int, KeyPoint>> &keypoints);

    int GetConvDetectionPoseResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects,std::vector<std::map<int, KeyPoint>> &keypoints);

    // 旋转处理使用
    void xywhr2xyxyxyxy(float x, float y, float w, float h, float angle,
                                float &pt1x, float &pt1y, float &pt2x, float &pt2y,
                                float &pt3x, float &pt3y, float &pt4x, float &pt4y);


    float objectThreshold;
    float nmsThreshold;
    std::string model_label_file_path;
    int class_num;

    // pose使用
    int keypoint_num;
    // seg使用
    int maskNum = 32;
    int mask_seg_w = 160;
    int mask_seg_h = 160;

    // obb使用
    int RegNum = 16;
    float RegDeq[16] = {0};

}

#endif // RK3588_DEMO_POSTPROCESS_H
