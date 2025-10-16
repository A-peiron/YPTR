#ifndef OSNET_REID_H
#define OSNET_REID_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "engine/engine.h"
#include "types/error.h"

class OSNetReID {
public:
    OSNetReID();
    ~OSNetReID();
    
    nn_error_e LoadModel(const char* model_path);
    std::vector<float> ExtractFeature(const cv::Mat& person_img);
    
    bool IsReady() const { return ready_; }
    
    // 计算特征相似度
    static float CosineSimilarity(const std::vector<float>& feat1, const std::vector<float>& feat2);
    
private:
    bool ready_ = false;
    std::shared_ptr<NNEngine> engine_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_ = false;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
};

#endif // OSNET_REID_H