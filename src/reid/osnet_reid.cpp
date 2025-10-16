#include "osnet_reid.h"
#include "utils/logging.h"
#include "process/preprocess.h"

OSNetReID::OSNetReID() {
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
}

OSNetReID::~OSNetReID() {
    // 释放输入张量内存
    if (input_tensor_.data != nullptr) {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    
    // 释放所有输出张量内存
    for (auto& tensor : output_tensors_) {
        if (tensor.data != nullptr) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
    output_tensors_.clear();
}

nn_error_e OSNetReID::LoadModel(const char* model_path) {
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS) {
        NN_LOG_ERROR("OSNet load model failed");
        return ret;
    }
    
    // 获取输入张量信息
    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1) {
        NN_LOG_ERROR("OSNet input tensor number is not 1");
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);
    
    // 获取输出张量信息
    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.empty()) {
        NN_LOG_ERROR("OSNet no output tensors");
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    
    if (output_shapes[0].type == NN_TENSOR_FLOAT16) {
        want_float_ = true;
    }
    
    for (const auto& shape : output_shapes) {
        tensor_data_s tensor;
        tensor.attr.n_elems = shape.n_elems;
        tensor.attr.n_dims = shape.n_dims;
        for (int j = 0; j < shape.n_dims; j++) {
            tensor.attr.dims[j] = shape.dims[j];
        }
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : shape.type;
        tensor.attr.index = 0;
        tensor.attr.size = shape.n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(shape.zp);
        out_scales_.push_back(shape.scale);
    }
    
    ready_ = true;
    NN_LOG_INFO("OSNet model loaded successfully");
    return NN_SUCCESS;
}

std::vector<float> OSNetReID::ExtractFeature(const cv::Mat& person_img) {
    std::vector<float> feature;
    
    if (!ready_ || person_img.empty()) {
        return feature;
    }
    
    // 预处理：调整到模型输入尺寸
    cv::Mat resized_img;
    cv::resize(person_img, resized_img, cv::Size(input_tensor_.attr.dims[2], input_tensor_.attr.dims[1]));
    
    // 转换为模型输入格式
    cvimg2tensor(resized_img, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    
    // 推理
    std::vector<tensor_data_s> inputs = {input_tensor_};
    auto ret = engine_->Run(inputs, output_tensors_, want_float_);
    if (ret != NN_SUCCESS) {
        NN_LOG_ERROR("OSNet inference failed");
        return feature;
    }
    
    // 提取特征
    int feature_size = output_tensors_[0].attr.n_elems;
    feature.resize(feature_size);
    
    if (want_float_) {
        float* output_data = (float*)output_tensors_[0].data;
        std::copy(output_data, output_data + feature_size, feature.begin());
    } else {
        int8_t* output_data = (int8_t*)output_tensors_[0].data;
        float scale = out_scales_[0];
        int zp = out_zps_[0];
        for (int i = 0; i < feature_size; i++) {
            feature[i] = (output_data[i] - zp) * scale;
        }
    }
    
    // L2归一化
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& val : feature) {
            val /= norm;
        }
    }
    
    return feature;
}

float OSNetReID::CosineSimilarity(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size() || feat1.empty()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    for (size_t i = 0; i < feat1.size(); i++) {
        dot_product += feat1[i] * feat2[i];
    }
    
    return dot_product; // 已经归一化，所以余弦相似度就是点积
}