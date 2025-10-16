#include "yolov8_custom.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

static std::vector<std::string> g_classes = {};

Yolov8Custom::Yolov8Custom()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false; // 是否使用浮点数版本的后处理
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolov8Custom::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolov8 load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolov8 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 6)
    {
        NN_LOG_WARNING("yolov8 output tensor number is not 6, but %ld", output_shapes.size());
        // return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolov8 output tensor type is float16, want type set to float32");
    }
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        // tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
 
    ready_ = true;
    return NN_SUCCESS;
}

char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int Yolov8Custom::setStaticParams(float NMS_threshold,float box_threshold, std::string model_labels_file_path,int obj_class_num,int keypoint_num) 
{
    yolo::nmsThreshold = NMS_threshold;
    yolo::objectThreshold = box_threshold;
    yolo::class_num = obj_class_num;
    yolo::keypoint_num = keypoint_num;
    FILE *file = fopen(model_labels_file_path.c_str(), "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", model_labels_file_path.c_str());
        return -1;
    }
    // 初始化g_classes
    // g_classes = std::vector<std::string>(0);

    while ((s = readLine(file, s, &n)) != NULL)
    {
        g_classes.push_back(s); 
        if (i >= obj_class_num)
            break;
    }
    fclose(file);
    return 0;  // 成功返回0
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, cv::Mat &image_letterbox)
{

    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox
    letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);

    cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);


    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    void *output_data[6];
    for (int i = 0; i < 6; i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        // 使用浮点数版本的后处理，他也支持量化的模型
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
    }
    else
    {
        // 使用量化版本的后处理，只能处理量化的模型
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::PostprocessObb(const cv::Mat &img, std::vector<Detection> &objects)
{
    void *output_data[output_tensors_.size()];
    for (int i = 0; i < output_tensors_.size(); i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;

    if (want_float_)
    {
        yolo::GetConvDetectionObbResult((float **)output_data, DetectiontRects);
    }
    else
    {
        yolo::GetConvDetectionObbResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 15)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int pt1x = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int pt1y = int(DetectiontRects[i + 3] * float(img_height) + 0.5);

        int pt2x = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int pt2y = int(DetectiontRects[i + 5] * float(img_height) + 0.5);

        int pt3x = int(DetectiontRects[i + 6] * float(img_width) + 0.5);
        int pt3y = int(DetectiontRects[i + 7] * float(img_height) + 0.5);

        int pt4x = int(DetectiontRects[i + 8] * float(img_width) + 0.5);
        int pt4y = int(DetectiontRects[i + 9] * float(img_height) + 0.5);

        int cx = int(DetectiontRects[i + 10] * float(img_width) + 0.5);
        int cy = int(DetectiontRects[i + 11] * float(img_height) + 0.5);
        int w = int(DetectiontRects[i + 12] * float(img_width) + 0.5);
        int h = int(DetectiontRects[i + 13] * float(img_height) + 0.5);

        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.point1 = cv::Point(pt1x, pt1y);
        result.point2 = cv::Point(pt2x, pt2y);
        result.point3 = cv::Point(pt3x, pt3y);
        result.point4 = cv::Point(pt4x, pt4y);
        result.x = cx;
        result.y = cy;
        result.w = w;
        result.h = h;
        result.angle = DetectiontRects[i + 14];

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::PostprocessPose(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints)
{
    void *output_data[output_tensors_.size()];
    for (int i = 0; i < output_tensors_.size(); i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        // 使用浮点数版本的后处理，他也支持量化的模型
        yolo::GetConvDetectionPoseResult((float **)output_data, DetectiontRects,keypoints);
    }
    else
    {
        // 使用量化版本的后处理，只能处理量化的模型
        yolo::GetConvDetectionPoseResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects,keypoints);
    }

    for (auto &kp : keypoints)
    {
        for (auto &kp_item : kp)
        {
            kp_item.second.x = kp_item.second.x * float(img.cols);
            kp_item.second.y = kp_item.second.y * float(img.rows);
        }
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}

void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}

void letterbox_decode_obb(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.point1.x -= pad;
            obj.point2.x -= pad;
            obj.point3.x -= pad;
            obj.point4.x -= pad;
            // obb+track使用
            obj.x -= pad;
        }
        else
        {
            obj.point1.y -= pad;
            obj.point2.y -= pad;
            obj.point3.y -= pad;
            obj.point4.y -= pad;
            // obb+track使用
            obj.y -= pad;
        }
    }
}

void letterbox_decode_pose(std::vector<std::map<int, KeyPoint>> &keypoints, bool hor, int pad)
{
    for (auto &keypoint : keypoints)
    {
        for (auto &keypoint_item : keypoint)
        {
            if (hor)
            {
                keypoint_item.second.x -= pad;
            }
            else
            {
                keypoint_item.second.y -= pad;
            }
        }
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    Preprocess(img, image_letterbox);
    // 推理
    Inference();
    // 后处理
    Postprocess(image_letterbox, objects);

    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::RunObb(const cv::Mat &img, std::vector<Detection> &objects)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    Preprocess(img, image_letterbox);
    // 推理
    Inference();
    // 后处理
    PostprocessObb(image_letterbox, objects);

    letterbox_decode_obb(objects, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::RunPose(const cv::Mat &img, std::vector<Detection> &objects, std::vector<std::map<int, KeyPoint>> &keypoints)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    Preprocess(img, image_letterbox);
    // 推理
    Inference();
    // 后处理
    PostprocessPose(image_letterbox, objects, keypoints);


    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);

    letterbox_decode_pose(keypoints, letterbox_info_.hor, letterbox_info_.pad);

    return NN_SUCCESS;
}