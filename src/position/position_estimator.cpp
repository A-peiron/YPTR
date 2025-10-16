#include "position_estimator.h"
#include <cmath>
#include <iostream>

PositionEstimator::PositionEstimator()
    : calibrated_(false),
      camera_position_(0.0f, -317.0f, 62.0f),  // 相机位置：场地后方中央，高62cm
      tilt_angle_(12.0f),                       // 仰角12度
      assumed_height_(178.0f),                  // 固定假设身高178cm
      fx_(0), fy_(0), cx_(0), cy_(0) {
}

PositionEstimator::~PositionEstimator() {
}

bool PositionEstimator::initialize(const std::string& calib_file) {
    return loadCalibrationYAML(calib_file);
}

bool PositionEstimator::loadCalibrationYAML(const std::string& calib_file) {
    try {
        cv::FileStorage fs(calib_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open calibration file: " << calib_file << std::endl;
            return false;
        }

        // 读取相机内参矩阵和畸变系数
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"] >> camera_matrix;
        fs["dist_coeffs"] >> dist_coeffs;

        // 读取图像尺寸
        int width, height;
        fs["image_size"]["width"] >> width;
        fs["image_size"]["height"] >> height;

        fs.release();

        if (camera_matrix.empty() || dist_coeffs.empty() || width <= 0 || height <= 0) {
            std::cerr << "Invalid calibration data in file: " << calib_file << std::endl;
            return false;
        }

        camera_matrix_ = camera_matrix;
        dist_coeffs_ = dist_coeffs;
        image_size_ = cv::Size(width, height);

        fx_ = camera_matrix.at<double>(0, 0);
        fy_ = camera_matrix.at<double>(1, 1);
        cx_ = camera_matrix.at<double>(0, 2);
        cy_ = camera_matrix.at<double>(1, 2);

        calibrated_ = true;
        std::cout << "Camera calibration loaded successfully." << std::endl;
        std::cout << "Camera matrix: fx=" << fx_ << ", fy=" << fy_ << ", cx=" << cx_ << ", cy=" << cy_ << std::endl;
        std::cout << "Image size: " << image_size_.width << "x" << image_size_.height << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error loading calibration file: " << e.what() << std::endl;
        return false;
    }
}

cv::Point2f PositionEstimator::undistortPoint(const cv::Point2f& point) {
    if (!calibrated_) {
        return point;
    }

    std::vector<cv::Point2f> input_points = {point};
    std::vector<cv::Point2f> output_points;

    cv::undistortPoints(input_points, output_points, camera_matrix_, dist_coeffs_, cv::Mat(), camera_matrix_);

    // 处理边缘区域的过度补偿，实现平滑过渡
    cv::Point2f result = output_points[0];

    // 计算点到图像中心的距离比例
    float center_x = image_size_.width * 0.5f;
    float center_y = image_size_.height * 0.5f;
    float max_dist = std::sqrt(center_x * center_x + center_y * center_y);

    float dist_from_center = std::sqrt((point.x - center_x) * (point.x - center_x) +
                                      (point.y - center_y) * (point.y - center_y));
    float dist_ratio = dist_from_center / max_dist;

    // 在边缘区域（距离中心>0.7）应用平滑系数
    if (dist_ratio > 0.7f) {
        float smooth_factor = 1.0f - 0.5f * (dist_ratio - 0.7f) / 0.3f; // 0.7-1.0区间内线性从1.0降到0.5
        smooth_factor = std::max(0.5f, smooth_factor); // 最小0.5，避免过度补偿

        // 混合原始点和校正点
        result.x = point.x + smooth_factor * (result.x - point.x);
        result.y = point.y + smooth_factor * (result.y - point.y);
    }

    return result;
}

bool PositionEstimator::getValidKeypoint(const std::map<int, KeyPoint>& keypoints, int idx, cv::Point2f& point, float min_score) {
    auto it = keypoints.find(idx);
    if (it != keypoints.end() && it->second.score >= min_score) {
        point = cv::Point2f(it->second.x, it->second.y);
        return true;
    }
    return false;
}

float PositionEstimator::calculatePixelDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float PositionEstimator::estimatePersonHeight(const std::map<int, KeyPoint>& keypoints) {
    cv::Point2f shoulder_point, ankle_point;

    // 优先使用左肩膀，如果没有则使用右肩膀
    bool has_shoulder = getValidKeypoint(keypoints, LEFT_SHOULDER, shoulder_point, 0.3f);
    if (!has_shoulder) {
        has_shoulder = getValidKeypoint(keypoints, RIGHT_SHOULDER, shoulder_point, 0.3f);
    }

    // 获取两个脚踝的位置，选择更低的那个（y值更大）
    cv::Point2f left_ankle, right_ankle;
    bool has_left_ankle = getValidKeypoint(keypoints, LEFT_ANKLE, left_ankle, 0.3f);
    bool has_right_ankle = getValidKeypoint(keypoints, RIGHT_ANKLE, right_ankle, 0.3f);

    bool has_ankle = false;
    if (has_left_ankle && has_right_ankle) {
        // 选择更低的脚踝（y值更大表示更靠下）
        ankle_point = (left_ankle.y > right_ankle.y) ? left_ankle : right_ankle;
        has_ankle = true;
    } else if (has_left_ankle) {
        ankle_point = left_ankle;
        has_ankle = true;
    } else if (has_right_ankle) {
        ankle_point = right_ankle;
        has_ankle = true;
    }

    if (has_shoulder && has_ankle) {
        float pixel_height = calculatePixelDistance(shoulder_point, ankle_point);
        // 肩膀到脚踝的距离约为身高减去40cm (头顶到肩膀30cm + 脚踝到地面10cm)
        return pixel_height; // 返回像素高度，用于后续计算
    }

    return 0.0f; // 无法估算
}

float PositionEstimator::estimateXPosition(const cv::Point2f& nose_point) {
    if (!calibrated_) {
        return 0.0f;
    }

    // 校正畸变
    cv::Point2f undistorted_nose = undistortPoint(nose_point);

    // 基于固定身高178cm和摄像头参数计算x轴位置
    // 这里使用简化的透视变换模型

    // 将仰角转换为弧度
    float tilt_rad = tilt_angle_ * M_PI / 180.0f;

    // 计算归一化坐标
    float norm_x = (undistorted_nose.x - cx_) / fx_;
    float norm_y = (undistorted_nose.y - cy_) / fy_;

    // 假设人站在地面上，计算射线与地面的交点
    // 考虑摄像头的仰角和高度
    float camera_height_m = camera_position_.z / 100.0f; // 转换为米

    // 简化的地面投影计算
    // 这里假设地面y=0，摄像头在(0, camera_position_.y, camera_height_m)
    float ground_distance = camera_height_m / std::tan(tilt_rad + norm_y);
    float x_world = norm_x * ground_distance;

    return x_world;
}

PersonPosition PositionEstimator::estimatePosition(const std::map<int, KeyPoint>& keypoints) {
    PersonPosition result;

    // 1. 获取人体中心点（髋部优先）
    cv::Point2f center_point;
    if (!getPersonCenter(keypoints, center_point)) {
        result.valid = false;
        return result;
    }

    // 2. 估算像素身高（基于肩膀到脚踝）
    float pixel_height = estimatePersonHeight(keypoints);
    if (pixel_height <= 0) {
        result.valid = false;
        return result;
    }

    // 3. 基于像素身高估算深度
    float depth = estimateDepthFromPixelHeight(pixel_height);
    if (depth <= 0) {
        result.valid = false;
        return result;
    }

    // 4. 基于深度和中心点估算X坐标
    float x_position = estimateXPositionImproved(center_point, depth);

    // 5. 填充结果
    result.center_point = center_point;
    result.pixel_height = pixel_height;
    result.estimated_height = assumed_height_ / 100.0f; // 转换为米
    result.x_world = x_position;
    result.valid = true;

    return result;
}

float PositionEstimator::pixelToWorldX(const cv::Point2f& pixel_point) {
    return estimateXPosition(pixel_point);
}

bool PositionEstimator::getPersonCenter(const std::map<int, KeyPoint>& keypoints, cv::Point2f& center_point) {
    // 策略1: 优先使用髋部中心（最稳定，不受手臂影响）
    cv::Point2f left_hip, right_hip;
    bool has_left_hip = getValidKeypoint(keypoints, LEFT_HIP, left_hip, 0.3f);
    bool has_right_hip = getValidKeypoint(keypoints, RIGHT_HIP, right_hip, 0.3f);

    if (has_left_hip && has_right_hip) {
        // 两个髋部都检测到，取中心
        center_point.x = (left_hip.x + right_hip.x) * 0.5f;
        center_point.y = (left_hip.y + right_hip.y) * 0.5f;
        return true;
    } else if (has_left_hip) {
        // 只有左髋部
        center_point = left_hip;
        return true;
    } else if (has_right_hip) {
        // 只有右髋部
        center_point = right_hip;
        return true;
    }

    // 策略2: 备选使用肩膀中心
    cv::Point2f left_shoulder, right_shoulder;
    bool has_left_shoulder = getValidKeypoint(keypoints, LEFT_SHOULDER, left_shoulder, 0.3f);
    bool has_right_shoulder = getValidKeypoint(keypoints, RIGHT_SHOULDER, right_shoulder, 0.3f);

    if (has_left_shoulder && has_right_shoulder) {
        // 两个肩膀都检测到，取中心
        center_point.x = (left_shoulder.x + right_shoulder.x) * 0.5f;
        center_point.y = (left_shoulder.y + right_shoulder.y) * 0.5f;
        return true;
    } else if (has_left_shoulder) {
        // 只有左肩膀
        center_point = left_shoulder;
        return true;
    } else if (has_right_shoulder) {
        // 只有右肩膀
        center_point = right_shoulder;
        return true;
    }

    // 策略3: 最后备选使用鼻子（虽然不稳定，但总比没有好）
    cv::Point2f nose;
    if (getValidKeypoint(keypoints, NOSE, nose, 0.3f)) {
        center_point = nose;
        return true;
    }

    return false; // 无法找到任何有效的中心点
}

float PositionEstimator::estimateDepthFromPixelHeight(float pixel_height) {
    if (pixel_height <= 0 || !calibrated_) {
        return -1.0f;
    }

    // 基于mtmc_opt项目的算法
    // 肩膀到脚踝对应真实身高减去40cm = 138cm
    float actual_height_cm = assumed_height_ - 40.0f; // 178 - 40 = 138cm

    // 考虑仰角对有效焦距的影响
    float tilt_rad = tilt_angle_ * CV_PI / 180.0f;
    float effective_fy = fy_ * std::cos(tilt_rad);

    // 估算深度：depth = (real_height * fy) / pixel_height
    float estimated_depth = (actual_height_cm * effective_fy) / pixel_height;

    return estimated_depth;
}

float PositionEstimator::estimateXPositionImproved(const cv::Point2f& center_point, float depth) {
    if (!calibrated_ || depth <= 0) {
        return 0.0f;
    }

    try {
        // 1. 畸变校正 - 参考mtmc_opt的边缘保护策略
        cv::Point2f undistorted_point = undistortPoint(center_point);
        float undistorted_x = undistorted_point.x;

        // 2. 转换为归一化坐标
        float normalized_x = (undistorted_x - cx_) / fx_;

        // 3. 计算实际X坐标（相机坐标系）
        float camera_coord_x = normalized_x * depth;

        // 4. 转换到世界坐标系
        // 相机在世界坐标系的X位置是0（场地中央），所以直接使用camera_coord_x
        float world_x = camera_coord_x + camera_position_.x; // camera_position_.x = 0

        // 5. 转换为米（depth是cm单位）并添加镜像效果（照镜子）
        return -(world_x / 100.0f); // 添加负号实现镜像

    } catch (const std::exception& e) {
        return 0.0f;
    }
}