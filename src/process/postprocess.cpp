
#include "postprocess.h"

#include <string.h>
#include <stdlib.h>

#include <algorithm>

#include <math.h>

#include "utils/logging.h"

int get_top(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

namespace yolo
{
    static int input_w = 640;
    static int input_h = 640;
    // static float objectThreshold = 0.4;
    // static float nmsThreshold = 0.25;
    static int headNum = 3;
    // static int class_num = 2;
    static int strides[3] = {8, 16, 32};
    static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    #define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
    #define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))
    static inline float fast_exp(float x)
    {
        // return exp(x);
        union
        {
            uint32_t i;
            float f;
        } v;
        v.i = (12102203.1616540672 * x + 1064807160.56887296);
        return v.f;
    }

    float sigmoid(float x)
    {
        return 1 / (1 + fast_exp(-x));
    }

    static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
    {
        float Inter = 0;
        float Total = 0;
        float XMin = 0;
        float YMin = 0;
        float XMax = 0;
        float YMax = 0;
        float Area1 = 0;
        float Area2 = 0;
        float InterWidth = 0;
        float InterHeight = 0;

        XMin = ZQ_MAX(XMin1, XMin2);
        YMin = ZQ_MAX(YMin1, YMin2);
        XMax = ZQ_MIN(XMax1, XMax2);
        YMax = ZQ_MIN(YMax1, YMax2);

        InterWidth = XMax - XMin;
        InterHeight = YMax - YMin;

        InterWidth = (InterWidth >= 0) ? InterWidth : 0;
        InterHeight = (InterHeight >= 0) ? InterHeight : 0;

        Inter = InterWidth * InterHeight;

        Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
        Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);

        Total = Area1 + Area2 - Inter;

        return float(Inter) / float(Total);
    }

    static inline void GetCovarianceMatrix(CSXYWHR &Box, float &A, float &B, float &C)
    {
        float a = Box.w;
        float b = Box.h;
        float c = Box.angle;

        float cos1 = cos(c);
        float sin1 = sin(c);
        float cos2 = pow(cos1, 2);
        float sin2 = pow(sin1, 2);

        A = a * cos2 + b * sin2;
        B = a * sin2 + b * cos2;
        C = (a - b) * cos1 * sin1;
    }

    static inline float probiou(CSXYWHR &obb1, CSXYWHR &obb2)
    {
        float eps = 1e-7;

        float x1 = obb1.x;
        float y1 = obb1.y;
        float x2 = obb2.x;
        float y2 = obb2.y;
        float a1 = 0, b1 = 0, c1 = 0;
        GetCovarianceMatrix(obb1, a1, b1, c1);

        float a2 = 0, b2 = 0, c2 = 0;
        GetCovarianceMatrix(obb2, a2, b2, c2);

        float t1 = (((a1 + a2) * pow((y1 - y2), 2) + (b1 + b2) * pow((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.25;
        float t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2) + eps)) * 0.5;

        float temp1 = (a1 * b1 - pow(c1, 2));
        temp1 = temp1 > 0 ? temp1 : 0;

        float temp2 = (a2 * b2 - pow(c2, 2));
        temp2 = temp2 > 0 ? temp2 : 0;

        float t3 = log((((a1 + a2) * (b1 + b2) - pow((c1 + c2), 2)) / (4 * sqrt((temp1 * temp2)) + eps) + eps)) * 0.5;

        float bd = 0;
        if ((t1 + t2 + t3) > 100)
        {
            bd = 100;
        }
        else if ((t1 + t2 + t3) < eps)
        {
            bd = eps;
        }
        else
        {
            bd = t1 + t2 + t3;
        }

        float hd = sqrt((1.0 - exp(-bd) + eps));
        return 1 - hd;
    }

    void xywhr2xyxyxyxy(float x, float y, float w, float h, float angle,
                                    float &pt1x, float &pt1y, float &pt2x, float &pt2y,
                                    float &pt3x, float &pt3y, float &pt4x, float &pt4y)
    {
        float cos_value = cos(angle);
        float sin_value = sin(angle);

        float vec1x = w / 2 * cos_value;
        float vec1y = w / 2 * sin_value;
        float vec2x = -h / 2 * sin_value;
        float vec2y = h / 2 * cos_value;

        pt1x = x + vec1x + vec2x;
        pt1y = y + vec1y + vec2y;

        pt2x = x + vec1x - vec2x;
        pt2y = y + vec1y - vec2y;

        pt3x = x - vec1x - vec2x;
        pt3y = y - vec1y - vec2y;

        pt4x = x - vec1x + vec2x;
        pt4y = y - vec1y + vec2y;
    }

    static float DeQnt2F32(int8_t qnt, int zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    std::vector<float> GenerateMeshgrid()
    {
        std::vector<float> meshgrid;
        if (headNum == 0)
        {
            NN_LOG_ERROR("=== yolov8 Meshgrid  Generate failed! ");
            exit(-1);
        }

        for (int index = 0; index < headNum; index++)
        {
            for (int i = 0; i < mapSize[index][0]; i++)
            {
                for (int j = 0; j < mapSize[index][1]; j++)
                {
                    meshgrid.push_back(float(j + 0.5));
                    meshgrid.push_back(float(i + 0.5));
                }
            }
        }

        printf("=== yolov8 Meshgrid  Generate success! \n");
        return meshgrid;
    }
    // int8版本
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects)
    {
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        int quant_zp_cls = 0, quant_zp_reg = 0;
        float quant_scale_cls = 0, quant_scale_reg = 0;

        DetectRect temp;
        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
            int8_t *cls = (int8_t *)pBlob[index * 2 + 1];

            quant_zp_reg = qnt_zp[index * 2 + 0];
            quant_zp_cls = qnt_zp[index * 2 + 1];

            quant_scale_reg = qnt_scale[index * 2 + 0];
            quant_scale_cls = qnt_scale[index * 2 + 1];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < class_num; cl++)
                    {
                        cls_val = sigmoid(
                            DeQnt2F32(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                      quant_zp_cls, quant_scale_cls));
                        // NN_LOG_INFO("xcls: %f, quant_zp_cls: %f, quant_scale_cls: %f",cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],quant_zp_cls,quant_scale_cls);
                        if(cls_val>=1)
                        {   
                            // NN_LOG_INFO("cls_val: %f, cls_val > 1", cls_val);
                            // NN_LOG_INFO("cls: %f, quant_zp_cls: %f, quant_scale_cls: %f",cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],quant_zp_cls,quant_scale_cls);
                            continue;
                        }
                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    // NN_LOG_DEBUG("cls_index: %d, cls_max: %f", cls_index, cls_max);
                    
                    if (cls_max > objectThreshold)
                    {
                        xmin = (meshgrid[gridIndex + 0] -
                                DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;
                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;
            NN_LOG_DEBUG("xmin1:%f,ymin1:%f,xmax1:%f,ymax1:%f,classId:%d,score:%f", xmin1, ymin1, xmax1, ymax1, classId, score);

            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }
    // 浮点数版本
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects)
    {
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        DetectRect temp;
        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            float *reg = (float *)pBlob[index * 2 + 0];
            float *cls = (float *)pBlob[index * 2 + 1];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < class_num; cl++)
                    {
                        cls_val = sigmoid(
                            cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                        if(cls_val>=1)
                        {
                            continue;
                        }
                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }
                    // NN_LOG_DEBUG("cls_index: %d, cls_max: %f", cls_index, cls_max);
                    if (cls_max > objectThreshold)
                    {
                        xmin = (meshgrid[gridIndex + 0] -
                                reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;
                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;
            NN_LOG_DEBUG("xmin1:%f,ymin1:%f,xmax1:%f,ymax1:%f,classId:%d,score:%f", xmin1, ymin1, xmax1, ymax1, classId, score);
            
            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }

    // obb使用
    int GetConvDetectionObbResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
    {
        int ret = 0;
        static auto meshgrid = GenerateMeshgrid();

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0, angle = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_ang = 0;
        float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_ang = 0;

        float sfsum = 0;
        float locval = 0;
        float locvaltemp = 0;

        std::vector<float> RegDFL;

        CSXYWHR Temp;
        std::vector<CSXYWHR> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
            int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
            int8_t *ang = (int8_t *)pBlob[headNum * 2 + index];

            quant_zp_reg = qnt_zp[index * 2 + 0];
            quant_zp_cls = qnt_zp[index * 2 + 1];
            quant_zp_ang = qnt_zp[headNum * 2 + index];

            quant_scale_reg = qnt_scale[index * 2 + 0];
            quant_scale_cls = qnt_scale[index * 2 + 1];
            quant_scale_ang = qnt_scale[headNum * 2 + index];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    if (1 == class_num)
                    {
                        cls_max = sigmoid(DeQnt2F32(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                        cls_index = 0;
                    }
                    else
                    {
                        for (int cl = 0; cl < class_num; cl++)
                        {
                            cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w];

                            if (0 == cl)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                            else
                            {
                                if (cls_val > cls_max)
                                {
                                    cls_max = cls_val;
                                    cls_index = cl;
                                }
                            }
                        }
                        cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
                    }
                    // NN_LOG_INFO("cls_max:%f", cls_max);
                    if (cls_max > objectThreshold)
                    {
                        RegDFL.clear();
                        for (int lc = 0; lc < 4; lc++)
                        {
                            sfsum = 0;
                            locval = 0;
                            for (int df = 0; df < RegNum; df++)
                            {
                                locvaltemp = exp(DeQnt2F32(reg[((lc * RegNum) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg));
                                RegDeq[df] = locvaltemp;
                                sfsum += locvaltemp;
                            }
                            for (int df = 0; df < RegNum; df++)
                            {
                                locvaltemp = RegDeq[df] / sfsum;
                                locval += locvaltemp * df;
                            }

                            RegDFL.push_back(locval);
                        }

                        angle = (sigmoid(DeQnt2F32(ang[h * mapSize[index][1] + w], quant_zp_ang, quant_scale_ang)) - 0.25) * pi;
                        xmin = RegDFL[0];
                        ymin = RegDFL[1];
                        xmax = RegDFL[2];
                        ymax = RegDFL[3];
                        float cos1 = cos(angle);
                        float sin1 = sin(angle);

                        float fx = (xmax - xmin) / 2;
                        float fy = (ymax - ymin) / 2;

                        float cx = ((fx * cos1 - fy * sin1) + meshgrid[gridIndex + 0]) * strides[index];
                        float cy = ((fx * sin1 + fy * cos1) + meshgrid[gridIndex + 1]) * strides[index];
                        float cw = (xmin + xmax) * strides[index];
                        float ch = (ymin + ymax) * strides[index];
                        Temp = {cls_index, cls_max, cx, cy, cw, ch, angle};
                        // NN_LOG_INFO("classId:%d, score:%f, x:%f, y:%f, w:%f, h:%f, angle:%f", cls_index, cls_max, cx, cy, cw, ch, angle);
                        detectRects.push_back(Temp);
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(), [](CSXYWHR &Rect1, CSXYWHR &Rect2) -> bool
                { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%zu", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                if (detectRects[j].classId != -1)
                {
                    if (probiou(detectRects[i], detectRects[j]) > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        for (int i = 0; i < detectRects.size(); i++)
        {
            float classid = detectRects[i].classId;
            if (-1 == classid)
            {
                continue;
            }
            float score = detectRects[i].score;
            float cx = detectRects[i].x;
            float cy = detectRects[i].y;
            float cw = detectRects[i].w;
            float ch = detectRects[i].h;
            float angle = detectRects[i].angle;

            float bw_ = cw > ch ? cw : ch;
            float bh_ = cw > ch ? ch : cw;

            float bt = cw > ch ? (angle - float(int(angle / pi)) * pi) : ((angle + pi / 2) - float(int((angle + pi / 2) / pi)) * pi);
            float pt1x = 0, pt1y = 0, pt2x = 0, pt2y = 0, pt3x = 0, pt3y = 0, pt4x = 0, pt4y = 0;
            xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y);

            DetectiontRects.push_back(classid);
            DetectiontRects.push_back(score);
            DetectiontRects.push_back(pt1x / input_w);
            DetectiontRects.push_back(pt1y / input_h);
            DetectiontRects.push_back(pt2x / input_w);
            DetectiontRects.push_back(pt2y / input_h);
            DetectiontRects.push_back(pt3x / input_w);
            DetectiontRects.push_back(pt3y / input_h);
            DetectiontRects.push_back(pt4x / input_w);
            DetectiontRects.push_back(pt4y / input_h);
            DetectiontRects.push_back(cx / input_w);
            DetectiontRects.push_back(cy / input_h);
            DetectiontRects.push_back(cw / input_w);
            DetectiontRects.push_back(ch / input_h);
            DetectiontRects.push_back(angle);
        }

        return ret;
    }

    // obb使用
    int GetConvDetectionObbResult(float **pBlob, std::vector<float> &DetectiontRects)
    {
        int ret = 0;
        static auto meshgrid = GenerateMeshgrid();

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0, angle = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        float sfsum = 0;
        float locval = 0;
        float locvaltemp = 0;

        std::vector<float> RegDFL;

        CSXYWHR Temp;
        std::vector<CSXYWHR> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            float *reg = (float *)pBlob[index * 2 + 0];
            float *cls = (float *)pBlob[index * 2 + 1];
            float *ang = (float *)pBlob[headNum * 2 + index];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    if (1 == class_num)
                    {
                        cls_max = sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                        cls_index = 0;
                    }
                    else
                    {
                        for (int cl = 0; cl < class_num; cl++)
                        {
                            cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w];

                            if (0 == cl)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                            else
                            {
                                if (cls_val > cls_max)
                                {
                                    cls_max = cls_val;
                                    cls_index = cl;
                                }
                            }
                        }
                        cls_max = sigmoid(cls_max);
                    }
                    // NN_LOG_INFO("cls_max:%f", cls_max);
                    if (cls_max > objectThreshold)
                    {
                        RegDFL.clear();
                        for (int lc = 0; lc < 4; lc++)
                        {
                            sfsum = 0;
                            locval = 0;
                            for (int df = 0; df < RegNum; df++)
                            {
                                locvaltemp = exp(reg[((lc * RegNum) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                                RegDeq[df] = locvaltemp;
                                sfsum += locvaltemp;
                            }
                            for (int df = 0; df < RegNum; df++)
                            {
                                locvaltemp = RegDeq[df] / sfsum;
                                locval += locvaltemp * df;
                            }

                            RegDFL.push_back(locval);
                        }

                        angle = (sigmoid(ang[h * mapSize[index][1] + w]) - 0.25) * pi;
                        xmin = RegDFL[0];
                        ymin = RegDFL[1];
                        xmax = RegDFL[2];
                        ymax = RegDFL[3];
                        float cos1 = cos(angle);
                        float sin1 = sin(angle);

                        float fx = (xmax - xmin) / 2;
                        float fy = (ymax - ymin) / 2;

                        float cx = ((fx * cos1 - fy * sin1) + meshgrid[gridIndex + 0]) * strides[index];
                        float cy = ((fx * sin1 + fy * cos1) + meshgrid[gridIndex + 1]) * strides[index];
                        float cw = (xmin + xmax) * strides[index];
                        float ch = (ymin + ymax) * strides[index];
                        Temp = {cls_index, cls_max, cx, cy, cw, ch, angle};

                        // NN_LOG_INFO("classId:%d, score:%f, x:%f, y:%f, w:%f, h:%f, angle:%f", cls_index, cls_max, cx, cy, cw, ch, angle);

                        detectRects.push_back(Temp);
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(), [](CSXYWHR &Rect1, CSXYWHR &Rect2) -> bool
                { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%zu", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            for (int j = i + 1; j < detectRects.size(); ++j)
            {
                if (detectRects[j].classId != -1)
                {
                    if (probiou(detectRects[i], detectRects[j]) > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        for (int i = 0; i < detectRects.size(); i++)
        {
            float classid = detectRects[i].classId;
            if (-1 == classid)
            {
                continue;
            }
            float score = detectRects[i].score;
            float cx = detectRects[i].x;
            float cy = detectRects[i].y;
            float cw = detectRects[i].w;
            float ch = detectRects[i].h;
            float angle = detectRects[i].angle;

            // NN_LOG_INFO("classid:%d,score:%f,cx:%f,cy:%f,cw:%f,ch:%f,angle:%f", classid, score, cx, cy, cw, ch, angle);

            float bw_ = cw > ch ? cw : ch;
            float bh_ = cw > ch ? ch : cw;

            float bt = cw > ch ? (angle - float(int(angle / pi)) * pi) : ((angle + pi / 2) - float(int((angle + pi / 2) / pi)) * pi);
            float pt1x = 0, pt1y = 0, pt2x = 0, pt2y = 0, pt3x = 0, pt3y = 0, pt4x = 0, pt4y = 0;
            xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y);

            DetectiontRects.push_back(classid);
            DetectiontRects.push_back(score);
            DetectiontRects.push_back(pt1x / input_w);
            DetectiontRects.push_back(pt1y / input_h);
            DetectiontRects.push_back(pt2x / input_w);
            DetectiontRects.push_back(pt2y / input_h);
            DetectiontRects.push_back(pt3x / input_w);
            DetectiontRects.push_back(pt3y / input_h);
            DetectiontRects.push_back(pt4x / input_w);
            DetectiontRects.push_back(pt4y / input_h);
            DetectiontRects.push_back(cx / input_w);
            DetectiontRects.push_back(cy / input_h);
            DetectiontRects.push_back(cw / input_w);
            DetectiontRects.push_back(ch / input_h);
            DetectiontRects.push_back(angle);
            // NN_LOG_INFO("classid:%f,score:%f,pt1x:%f,pt1y:%f,pt2x:%f,pt2y:%f,pt3x:%f,pt3y:%f,pt4x:%f,pt4y:%f", classid, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y);
        }

        return ret;
    }

    // pose使用
    int GetConvDetectionPoseResult(float **pBlob, std::vector<float> &DetectiontRects,std::vector<std::map<int, KeyPoint>> &keypoints){
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        int model_class_num = class_num;

        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            float *reg = (float *)pBlob[index * 2 + 0];
            float *cls = (float *)pBlob[index * 2 + 1];
            float *pose = (float *)pBlob[index + headNum * 2];

            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < model_class_num; cl++)
                    {
                        cls_val = sigmoid(
                            cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }

                    if (cls_max > objectThreshold)
                    {
                        DetectRect temp;
                        xmin = (meshgrid[gridIndex + 0] -
                                reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;

                            for (int kc = 0; kc < keypoint_num; kc++)
                            {
                                KeyPoint kp;
                                kp.x = (pose[(kc * 3 + 0) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] * 2 + (meshgrid[gridIndex + 0] - 0.5)) * strides[index] / input_w;
                                kp.y = (pose[(kc * 3 + 1) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] * 2 + (meshgrid[gridIndex + 1] - 0.5)) * strides[index] / input_h;
                                kp.score = sigmoid(pose[(kc * 3 + 2) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]);
                                kp.id = kc;

                                temp.keyPoints.push_back(kp);
                            }

                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;

            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                std::map<int, KeyPoint> kps;

                for (int kn = 0; kn < keypoint_num; kn++)
                {
                    kps.insert({kn, detectRects[i].keyPoints[kn]});
                }

                keypoints.push_back(kps);

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }

    int GetConvDetectionPoseResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects,std::vector<std::map<int, KeyPoint>> &keypoints)
    {
        static auto meshgrid = GenerateMeshgrid();
        int ret = 0;

        int gridIndex = -2;
        float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
        float cls_val = 0;
        float cls_max = 0;
        int cls_index = 0;

        int quant_zp_cls = 0, quant_zp_reg = 0, quant_zp_msk, quant_zp_seg = 0;
        float quant_scale_cls = 0, quant_scale_reg = 0, quant_scale_msk = 0, quant_scale_seg = 0;
        int quant_zp_pose = 0;
        float quant_scale_pose = 0;

        int model_class_num = class_num;

        std::vector<DetectRect> detectRects;

        for (int index = 0; index < headNum; index++)
        {
            int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
            int8_t *cls = (int8_t *)pBlob[index * 2 + 1];
            int8_t *pose = (int8_t *)pBlob[index + headNum * 2];

            quant_zp_reg = qnt_zp[index * 2 + 0];
            quant_zp_cls = qnt_zp[index * 2 + 1];

            quant_scale_reg = qnt_scale[index * 2 + 0];
            quant_scale_cls = qnt_scale[index * 2 + 1];

            quant_zp_pose = qnt_zp[index + headNum * 2];
            quant_scale_pose = qnt_scale[index + headNum * 2];


            for (int h = 0; h < mapSize[index][0]; h++)
            {
                for (int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    for (int cl = 0; cl < model_class_num; cl++)
                    {
                        cls_val = sigmoid(
                            DeQnt2F32(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                      quant_zp_cls, quant_scale_cls));

                        if (0 == cl)
                        {
                            cls_max = cls_val;
                            cls_index = cl;
                        }
                        else
                        {
                            if (cls_val > cls_max)
                            {
                                cls_max = cls_val;
                                cls_index = cl;
                            }
                        }
                    }

                    if (cls_max > objectThreshold)
                    {
                        DetectRect temp;
                        xmin = (meshgrid[gridIndex + 0] -
                                DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymin = (meshgrid[gridIndex + 1] -
                                DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        xmax = (meshgrid[gridIndex + 0] +
                                DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];
                        ymax = (meshgrid[gridIndex + 1] +
                                DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w],
                                          quant_zp_reg, quant_scale_reg)) *
                               strides[index];

                        xmin = xmin > 0 ? xmin : 0;
                        ymin = ymin > 0 ? ymin : 0;
                        xmax = xmax < input_w ? xmax : input_w;
                        ymax = ymax < input_h ? ymax : input_h;

                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                        {
                            temp.xmin = xmin / input_w;
                            temp.ymin = ymin / input_h;
                            temp.xmax = xmax / input_w;
                            temp.ymax = ymax / input_h;
                            temp.classId = cls_index;
                            temp.score = cls_max;

                            for (int kc = 0; kc < keypoint_num; kc++)
                            {
                                KeyPoint kp;
                                kp.x = (DeQnt2F32(pose[(kc * 3 + 0) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose) * 2 + (meshgrid[gridIndex + 0] - 0.5)) * strides[index] / input_w;
                                kp.y = (DeQnt2F32(pose[(kc * 3 + 1) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose) * 2 + (meshgrid[gridIndex + 1] - 0.5)) * strides[index] / input_h;
                                kp.score = sigmoid(DeQnt2F32(pose[(kc * 3 + 2) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_pose, quant_scale_pose));
                                kp.id = kc;

                                temp.keyPoints.push_back(kp);
                            }

                            detectRects.push_back(temp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](DetectRect &Rect1, DetectRect &Rect2) -> bool
                  { return (Rect1.score > Rect2.score); });

        NN_LOG_DEBUG("NMS Before num :%ld", detectRects.size());
        for (int i = 0; i < detectRects.size(); ++i)
        {
            float xmin1 = detectRects[i].xmin;
            float ymin1 = detectRects[i].ymin;
            float xmax1 = detectRects[i].xmax;
            float ymax1 = detectRects[i].ymax;
            int classId = detectRects[i].classId;
            float score = detectRects[i].score;

            if (classId != -1)
            {
                // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
                DetectiontRects.push_back(float(classId));
                DetectiontRects.push_back(float(score));
                DetectiontRects.push_back(float(xmin1));
                DetectiontRects.push_back(float(ymin1));
                DetectiontRects.push_back(float(xmax1));
                DetectiontRects.push_back(float(ymax1));

                std::map<int, KeyPoint> kps;
                for (int kn = 0; kn < keypoint_num; kn++)
                {
                    kps.insert({kn, detectRects[i].keyPoints[kn]});
                }
                keypoints.push_back(kps);

                for (int j = i + 1; j < detectRects.size(); ++j)
                {
                    float xmin2 = detectRects[j].xmin;
                    float ymin2 = detectRects[j].ymin;
                    float xmax2 = detectRects[j].xmax;
                    float ymax2 = detectRects[j].ymax;
                    float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if (iou > nmsThreshold)
                    {
                        detectRects[j].classId = -1;
                    }
                }
            }
        }

        return ret;
    }
    

}
