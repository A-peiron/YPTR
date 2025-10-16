#ifndef RK3588_DEMO_YOLOV8_CUSTOM_H
#define RK3588_DEMO_YOLOV8_CUSTOM_H

#include "engine/engine.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

class Yolov8Custom
{
public:
    Yolov8Custom();
    ~Yolov8Custom();

    nn_error_e LoadModel(const char *model_path);
    nn_error_e Run(const cv::Mat &img, std::vector<Detection> &objects);
    nn_error_e RunObb(const cv::Mat &img, std::vector<Detection> &objects);
    nn_error_e RunPose(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints);
    int setStaticParams(float NMS_threshold,float box_threshold, std::string model_labels_file_path, int obj_class_num,int keypoint_num = 0);

private:
    nn_error_e Preprocess(const cv::Mat &img, cv::Mat &image_letterbox);
    nn_error_e Inference();
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects);
    nn_error_e PostprocessObb(const cv::Mat &img, std::vector<Detection> &objects);
    nn_error_e PostprocessPose(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints);

    bool ready_;
    LetterBoxInfo letterbox_info_;
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    bool want_float_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_YOLOV8_CUSTOM_H
