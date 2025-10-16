#ifndef POSITION_ESTIMATOR_H
#define POSITION_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include "../types/yolo_datatype.h"

struct PersonPosition {
    float x_world;         // 世界坐标系x位置 (米)
    float pixel_height;    // 像素高度
    float estimated_height; // 估算的身高 (米)
    cv::Point2f center_point; // 人体中心像素坐标（髋部中心优先）
    bool valid;            // 估算是否有效

    PersonPosition() : x_world(0), pixel_height(0), estimated_height(1.78f), valid(false) {}
};

class PositionEstimator {
private:
    bool calibrated_;

    // 摄像头参数 - 参考mtmc_opt项目
    cv::Point3f camera_position_;  // 相机位置：场地后方中央，高62cm
    float tilt_angle_;             // 仰角12度
    float assumed_height_;         // 固定假设身高178cm

    // 相机内参
    float fx_, fy_, cx_, cy_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Size image_size_;

    // COCO pose关键点索引定义
    enum CocoKeypoints {
        NOSE = 0,
        LEFT_EYE = 1,
        RIGHT_EYE = 2,
        LEFT_EAR = 3,
        RIGHT_EAR = 4,
        LEFT_SHOULDER = 5,
        RIGHT_SHOULDER = 6,
        LEFT_ELBOW = 7,
        RIGHT_ELBOW = 8,
        LEFT_WRIST = 9,
        RIGHT_WRIST = 10,
        LEFT_HIP = 11,
        RIGHT_HIP = 12,
        LEFT_KNEE = 13,
        RIGHT_KNEE = 14,
        LEFT_ANKLE = 15,
        RIGHT_ANKLE = 16
    };

public:
    PositionEstimator();
    ~PositionEstimator();

    // 初始化摄像头标定参数
    bool initialize(const std::string& calib_file);

    // 畸变校正
    cv::Point2f undistortPoint(const cv::Point2f& point);

    // 估算人的身高 (基于肩膀到脚踝的像素距离，减去约40cm)
    float estimatePersonHeight(const std::map<int, KeyPoint>& keypoints);

    // 估算x轴位置 (基于鼻子位置和固定身高178cm)
    float estimateXPosition(const cv::Point2f& nose_point);

    // 综合位置估算
    PersonPosition estimatePosition(const std::map<int, KeyPoint>& keypoints);

    // 获取有效的关键点
    bool getValidKeypoint(const std::map<int, KeyPoint>& keypoints, int idx, cv::Point2f& point, float min_score = 0.3f);

    // 获取人体中心点（髋部中心优先，备选肩膀中心）
    bool getPersonCenter(const std::map<int, KeyPoint>& keypoints, cv::Point2f& center_point);

    // 基于像素身高估算深度
    float estimateDepthFromPixelHeight(float pixel_height);

    // 改进的X坐标估算（基于人体中心和深度）
    float estimateXPositionImproved(const cv::Point2f& center_point, float depth);

    // 计算两点间像素距离
    float calculatePixelDistance(const cv::Point2f& p1, const cv::Point2f& p2);

private:
    // 加载YAML格式的标定文件
    bool loadCalibrationYAML(const std::string& calib_file);

    // 像素坐标转世界坐标 (基于固定身高假设)
    float pixelToWorldX(const cv::Point2f& pixel_point);
};

#endif // POSITION_ESTIMATOR_H