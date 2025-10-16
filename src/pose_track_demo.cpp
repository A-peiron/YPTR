#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include "task/yolov8_custom.h"
#include "tracking/BYTETracker.h"
#include "tracking/dataType.h"
#include "reid/osnet_reid.h"
#include "position/position_estimator.h"
#include "websocket/position_server.h"
#include "reconfig/ReConfig.h"
#include "draw/cv_draw.h"
#include "utils/logging.h"
#include "filter/keypoint_smoothing.h"

// 将姿态检测结果转换为跟踪对象
void convert_detections_to_objects(const std::vector<Detection>& detections,
                                   std::vector<Object>& objects,
                                   const cv::Mat& frame,
                                   OSNetReID& reid_model,
                                   int frame_id,
                                   int reid_interval,
                                   float min_confidence) {
    objects.clear();

    // 判断是否需要提取ReID特征
    bool extract_reid = (reid_interval > 0) && (frame_id % reid_interval == 0);

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        // 只处理person类别(class_id=0),使用配置的最小置信度
        // 不要在这里过滤太严格,让ByteTracker处理低分检测
        if (det.class_id == 0 && det.confidence > min_confidence) {
            Object obj;
            obj.classId = det.class_id;
            obj.score = det.confidence;
            obj.box = det.box;
            obj.has_reid_extracted = false;
            obj.detection_index = static_cast<int>(i); // 记录原始检测索引

            // 提取ReID特征
            if (extract_reid && reid_model.IsReady()) {
                cv::Rect roi = det.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (roi.width > 20 && roi.height > 40) {  // 最小尺寸过滤
                    cv::Mat person_crop = frame(roi);
                    obj.reid_feature = reid_model.ExtractFeature(person_crop);
                    obj.has_reid_extracted = !obj.reid_feature.empty();
                }
            }

            objects.push_back(obj);
        }
    }
}

// 绘制性能信息
void draw_performance_info(cv::Mat& frame,
                          int frame_time,
                          size_t track_count,
                          bool reid_enabled,
                          bool websocket_enabled,
                          bool smoothing_enabled) {
    // FPS和帧时间
    cv::putText(frame,
               cv::format("Frame: %dms, FPS: %.1f", frame_time, 1000.0f / frame_time),
               cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7,
               cv::Scalar(0, 255, 0), 2);

    // 跟踪数量
    cv::putText(frame,
               cv::format("Tracks: %zu", track_count),
               cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7,
               cv::Scalar(0, 255, 0), 2);

    // ReID状态
    if (reid_enabled) {
        cv::putText(frame, "ReID: ON",
                   cv::Point(10, 90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 255), 2);
    }

    // WebSocket状态
    if (websocket_enabled) {
        cv::putText(frame, "WebSocket: ON",
                   cv::Point(10, 120),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 255), 2);
    }

    // Smoothing状态
    if (smoothing_enabled) {
        cv::putText(frame, "Smoothing: ON",
                   cv::Point(10, 150),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(0, 255, 255), 2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.ini>" << std::endl;
        std::cout << "Example: " << argv[0] << " pose_track_config.ini" << std::endl;
        return -1;
    }

    // 读取配置文件
    rr::RrConfig config;
    if (!config.ReadConfig(argv[1])) {
        std::cerr << "Failed to read config file: " << argv[1] << std::endl;
        return -1;
    }

    std::string model_path = config.ReadString("YUNYAN", "ModelPath", "weights/yolov8n-pose.int.rknn");
    std::string stream_url = config.ReadString("YUNYAN", "StreamUrl", "/dev/video0");
    std::string reid_model_path = config.ReadString("YUNYAN", "ReIDModelPath", "");
    std::string labels_file = config.ReadString("YUNYAN", "ModelLabelsFilePath", "pose_1_labels_list.txt");
    int reid_interval = config.ReadInt("YUNYAN", "ReIDInterval", 3);
    int keypoint_num = config.ReadInt("YUNYAN", "KeypointNum", 17);
    int obj_class_num = config.ReadInt("YUNYAN", "ObjClassNum", 1);
    float nms_threshold = config.ReadFloat("YUNYAN", "NMSThreshold", 0.65f);
    float box_threshold = config.ReadFloat("YUNYAN", "BoxThreshold", 0.5f);
    float min_confidence = config.ReadFloat("YUNYAN", "MinConfidence", 0.25f);  // 转换时的最小置信度
    bool enable_reid = config.ReadInt("YUNYAN", "EnableReID", 0) > 0;
    bool enable_websocket = config.ReadInt("YUNYAN", "EnableWebSocket", 0) > 0;
    std::string websocket_host = config.ReadString("YUNYAN", "WebSocketHost", "0.0.0.0");
    int websocket_port = config.ReadInt("YUNYAN", "WebSocketPort", 3000);

    // 读取平滑滤波配置
    bool enable_smoothing = config.ReadInt("YUNYAN", "EnableSmoothing", 1) > 0;
    float smoothing_min_cutoff = config.ReadFloat("YUNYAN", "SmoothingMinCutoff", 1.0f);
    float smoothing_beta = config.ReadFloat("YUNYAN", "SmoothingBeta", 0.007f);
    float smoothing_min_confidence = config.ReadFloat("YUNYAN", "SmoothingMinConfidence", 0.3f);

    std::cout << "=== Pose Tracking Demo ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Stream: " << stream_url << std::endl;
    std::cout << "ReID: " << (enable_reid ? "Enabled" : "Disabled") << std::endl;
    std::cout << "WebSocket: " << (enable_websocket ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Smoothing: " << (enable_smoothing ? "Enabled" : "Disabled") << std::endl;
    if (enable_smoothing) {
        std::cout << "  MinCutoff: " << smoothing_min_cutoff
                  << ", Beta: " << smoothing_beta
                  << ", MinConf: " << smoothing_min_confidence << std::endl;
    }
    std::cout << "MinConfidence: " << min_confidence << " (for convert filter)" << std::endl;

    // 初始化YOLOv8姿态检测
    Yolov8Custom pose_detector;
    if (pose_detector.LoadModel(model_path.c_str()) != NN_SUCCESS) {
        std::cerr << "Failed to load pose model: " << model_path << std::endl;
        return -1;
    }

    // 设置模型参数 - 关键步骤,设置全局后处理参数
    if (pose_detector.setStaticParams(nms_threshold, box_threshold,
                                     labels_file, obj_class_num, keypoint_num) != 0) {
        std::cerr << "Failed to set model parameters" << std::endl;
        return -1;
    }

    std::cout << "YOLOv8 Pose model loaded successfully" << std::endl;
    std::cout << "Parameters: NMS=" << nms_threshold << ", BoxThresh=" << box_threshold
              << ", KeypointNum=" << keypoint_num << std::endl;

    // 初始化ReID模型
    OSNetReID reid_model;
    if (enable_reid && !reid_model_path.empty()) {
        if (reid_model.LoadModel(reid_model_path.c_str()) != NN_SUCCESS) {
            std::cerr << "Failed to load ReID model: " << reid_model_path << std::endl;
            enable_reid = false;
        } else {
            std::cout << "ReID model loaded, interval: " << reid_interval << std::endl;
        }
    } else {
        enable_reid = false;
        std::cout << "ReID disabled" << std::endl;
    }

    // 初始化ByteTracker
    BYTETracker tracker(30, 30);  // fps=30, track_buffer=30
    std::cout << "ByteTracker initialized" << std::endl;

    // 初始化关键点平滑滤波器
    float estimated_fps = 30.0f; // 根据实际情况调整
    KeypointSmoothingManager keypoint_smoother(estimated_fps, smoothing_min_cutoff,
                                               smoothing_beta, smoothing_min_confidence);
    keypoint_smoother.SetEnabled(enable_smoothing);
    if (enable_smoothing) {
        std::cout << "Keypoint smoothing initialized (fps=" << estimated_fps << ")" << std::endl;
    }

    // 打开视频流
    cv::VideoCapture cap;
    if (stream_url.find("/dev/video") != std::string::npos) {
        // USB摄像头
        int camera_id = std::stoi(stream_url.substr(stream_url.rfind("o") + 1));
        cap.open(camera_id);
    } else {
        // RTSP或视频文件
        cap.open(stream_url);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video stream: " << stream_url << std::endl;
        return -1;
    }

    // 设置摄像头分辨率为640x480
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout << "Video stream opened successfully" << std::endl;
    std::cout << "Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH)
              << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // WebSocket服务器(可选)
    std::unique_ptr<PositionServer> ws_server;
    if (enable_websocket) {
        ws_server = std::make_unique<PositionServer>(websocket_host, websocket_port);
        if (!ws_server->Start()) {
            std::cerr << "Failed to start WebSocket server" << std::endl;
            enable_websocket = false;
        } else {
            std::cout << "WebSocket server started on ws://" << websocket_host << ":" << websocket_port << std::endl;
        }
    }

    // 主循环变量
    cv::Mat frame;
    std::vector<Detection> detections;
    std::vector<std::map<int, KeyPoint>> keypoints;
    std::vector<Object> objects;
    std::vector<STrack> tracks;
    int frame_count = 0;

    auto stats_start = std::chrono::steady_clock::now();
    int stats_frames = 0;

    std::cout << "\nStarting tracking loop. Press 'q' to quit\n" << std::endl;

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();

        // 读取帧
        if (!cap.read(frame)) {
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }

        // YOLOv8姿态检测 - 使用RunPose获取关键点
        detections.clear();
        keypoints.clear();
        pose_detector.RunPose(frame, detections, keypoints);

        // 调试输出
        if (frame_count % 30 == 0) {  // 每30帧输出一次
            std::cout << "Frame " << frame_count << ": Detected " << detections.size()
                     << " persons, " << keypoints.size() << " keypoint sets" << std::endl;
        }

        // 转换为跟踪对象(包含ReID特征提取)
        convert_detections_to_objects(detections, objects, frame,
                                     reid_model, frame_count,
                                     enable_reid ? reid_interval : 0,
                                     min_confidence);

        // 更新跟踪器
        tracks = tracker.update(objects);

        // 应用关键点平滑滤波（在跟踪之后，绘制和发送之前）
        if (enable_smoothing && !tracks.empty() && !keypoints.empty()) {
            for (const auto& track : tracks) {
                if (track.state == Tracked) {
                    int det_idx = track.detection_index;
                    if (det_idx >= 0 && det_idx < (int)keypoints.size()) {
                        keypoint_smoother.SmoothKeypoints(track.track_id, keypoints[det_idx]);
                    }
                }
            }
        }

        // 创建显示帧
        cv::Mat display_frame = frame.clone();

        // 绘制关键点
        if (!keypoints.empty()) {
            DrawCocoKps(display_frame, keypoints);
        }

        // 修改检测框显示：用track_id替换类名
        std::vector<Detection> display_detections = detections;
        for (size_t i = 0; i < display_detections.size() && i < tracks.size(); ++i) {
            // 找到对应的track
            for (const auto& track : tracks) {
                if (track.state == Tracked) {
                    std::vector<float> tlwh = track.tlwh;
                    cv::Rect track_box(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

                    // 检查是否是同一个检测框（通过IoU）
                    cv::Rect det_box = display_detections[i].box;
                    cv::Rect intersect = track_box & det_box;
                    float iou = (intersect.area() > 0) ?
                                (float)intersect.area() / (track_box.area() + det_box.area() - intersect.area()) : 0.0f;

                    if (iou > 0.5f) {
                        // 用ID替换类名
                        display_detections[i].className = "ID:" + std::to_string(track.track_id);
                        break;
                    }
                }
            }
        }

        // 绘制检测框
        DrawDetections(display_frame, display_detections);

        // WebSocket数据推送 - 使用关键点数据
        if (enable_websocket && ws_server) {
            std::vector<PositionData> position_data;

            // 为每个track找到对应的keypoint（通过detection_index）
            for (const auto& track : tracks) {
                if (track.state != Tracked) continue;

                std::vector<float> tlwh = track.tlwh;
                bool vertical = (tlwh[3] > 0) && (tlwh[2] / tlwh[3] > 1.6);
                if (tlwh[2] * tlwh[3] < 20 || vertical) continue;

                // 使用track_id作为person_id
                PositionData pos;
                pos.person_id = track.track_id;

                // 使用detection_index找到对应的关键点
                int det_idx = track.detection_index;
                if (det_idx < 0 || det_idx >= (int)keypoints.size()) {
                    continue; // 没有关键点数据，跳过
                }

                // 计算人体中心点（髋部优先，不受手臂影响）
                cv::Point2f body_center;
                bool has_center = false;

                // 策略1: 优先使用髋部中心（最稳定）
                cv::Point2f left_hip, right_hip;
                auto& kps = keypoints[det_idx];

                auto left_hip_it = kps.find(11);  // LEFT_HIP
                auto right_hip_it = kps.find(12); // RIGHT_HIP

                if (left_hip_it != kps.end() && left_hip_it->second.score >= 0.3f &&
                    right_hip_it != kps.end() && right_hip_it->second.score >= 0.3f) {
                    // 两个髋部都有，取中心
                    body_center.x = (left_hip_it->second.x + right_hip_it->second.x) * 0.5f;
                    body_center.y = (left_hip_it->second.y + right_hip_it->second.y) * 0.5f;
                    has_center = true;
                } else if (left_hip_it != kps.end() && left_hip_it->second.score >= 0.3f) {
                    // 只有左髋部
                    body_center.x = left_hip_it->second.x;
                    body_center.y = left_hip_it->second.y;
                    has_center = true;
                } else if (right_hip_it != kps.end() && right_hip_it->second.score >= 0.3f) {
                    // 只有右髋部
                    body_center.x = right_hip_it->second.x;
                    body_center.y = right_hip_it->second.y;
                    has_center = true;
                }

                // 策略2: 备选使用肩膀中心
                if (!has_center) {
                    auto left_shoulder_it = kps.find(5);  // LEFT_SHOULDER
                    auto right_shoulder_it = kps.find(6); // RIGHT_SHOULDER

                    if (left_shoulder_it != kps.end() && left_shoulder_it->second.score >= 0.3f &&
                        right_shoulder_it != kps.end() && right_shoulder_it->second.score >= 0.3f) {
                        body_center.x = (left_shoulder_it->second.x + right_shoulder_it->second.x) * 0.5f;
                        body_center.y = (left_shoulder_it->second.y + right_shoulder_it->second.y) * 0.5f;
                        has_center = true;
                    } else if (left_shoulder_it != kps.end() && left_shoulder_it->second.score >= 0.3f) {
                        body_center.x = left_shoulder_it->second.x;
                        body_center.y = left_shoulder_it->second.y;
                        has_center = true;
                    } else if (right_shoulder_it != kps.end() && right_shoulder_it->second.score >= 0.3f) {
                        body_center.x = right_shoulder_it->second.x;
                        body_center.y = right_shoulder_it->second.y;
                        has_center = true;
                    }
                }

                // 策略3: 最后备选使用鼻子
                if (!has_center) {
                    auto nose_it = kps.find(0); // NOSE
                    if (nose_it != kps.end() && nose_it->second.score >= 0.3f) {
                        body_center.x = nose_it->second.x;
                        body_center.y = nose_it->second.y;
                        has_center = true;
                    }
                }

                // 如果还是没有中心点，使用检测框中心作为最后的备选
                if (!has_center) {
                    body_center.x = tlwh[0] + tlwh[2] / 2.0f;
                    body_center.y = tlwh[1] + tlwh[3] / 2.0f;
                }

                // 归一化坐标到[-1, 1]（使用身体中心而非检测框中心）
                pos.x_normalized = (body_center.x / (frame.cols / 2.0f)) - 1.0f;
                pos.y_normalized = 0.0f;
                pos.valid = true;
                // 修改stable判断：使用is_activated状态，因为tracklet_len会在re_activate时重置
                pos.stable = (track.is_activated && track.tracklet_len >= 2);

                // 收集关键点数据（不过滤关键点score，传输所有关键点）
                for (const auto& kp_pair : kps) {
                    int kp_id = kp_pair.first;
                    const KeyPoint& kp = kp_pair.second;
                    // 创建归一化的关键点
                    KeyPoint normalized_kp;
                    normalized_kp.x = kp.x / (float)frame.cols;
                    normalized_kp.y = kp.y / (float)frame.rows;
                    normalized_kp.score = kp.score;
                    normalized_kp.id = kp_id;
                    pos.keypoints[kp_id] = normalized_kp;
                }

                position_data.push_back(pos);
            }

            // 按照person_id（即track_id）从小到大排序
            std::sort(position_data.begin(), position_data.end(),
                     [](const PositionData& a, const PositionData& b) {
                         return a.person_id < b.person_id;
                     });

            ws_server->UpdatePositionData(position_data);

            // 调试输出
            if (frame_count % 30 == 0 && !position_data.empty()) {
                std::cout << "WebSocket: Sent " << position_data.size()
                         << " persons with IDs: ";
                for (const auto& p : position_data) {
                    std::cout << p.person_id << "(" << p.keypoints.size() << "kp,stable="
                             << (p.stable ? "T" : "F") << ") ";
                }
                std::cout << std::endl;
            }
        }

        // 计算帧时间
        auto frame_end = std::chrono::steady_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_end - frame_start).count();

        // 绘制性能信息
        draw_performance_info(display_frame, static_cast<int>(frame_time),
                            tracks.size(), enable_reid, enable_websocket, enable_smoothing);

        // 显示
        cv::imshow("Pose Tracking Demo", display_frame);

        // 键盘控制
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            std::cout << "User requested exit" << std::endl;
            break;
        }
        // 's': 切换平滑开关
        else if (key == 's' || key == 'S') {
            enable_smoothing = !enable_smoothing;
            keypoint_smoother.SetEnabled(enable_smoothing);
            std::cout << "Smoothing: " << (enable_smoothing ? "ON" : "OFF") << std::endl;
        }
        // '-': 降低平滑强度（增加min_cutoff）
        else if (key == '-' || key == '_') {
            smoothing_min_cutoff = std::min(smoothing_min_cutoff + 0.2f, 5.0f);
            keypoint_smoother.SetParameters(smoothing_min_cutoff, smoothing_beta);
            std::cout << "MinCutoff: " << smoothing_min_cutoff << " (less smooth)" << std::endl;
        }
        // '+': 增加平滑强度（降低min_cutoff）
        else if (key == '+' || key == '=') {
            smoothing_min_cutoff = std::max(smoothing_min_cutoff - 0.2f, 0.1f);
            keypoint_smoother.SetParameters(smoothing_min_cutoff, smoothing_beta);
            std::cout << "MinCutoff: " << smoothing_min_cutoff << " (more smooth)" << std::endl;
        }
        // '[': 降低运动响应度（降低beta）
        else if (key == '[' || key == '{') {
            smoothing_beta = std::max(smoothing_beta - 0.002f, 0.001f);
            keypoint_smoother.SetParameters(smoothing_min_cutoff, smoothing_beta);
            std::cout << "Beta: " << smoothing_beta << " (less responsive)" << std::endl;
        }
        // ']': 增加运动响应度（增加beta）
        else if (key == ']' || key == '}') {
            smoothing_beta = std::min(smoothing_beta + 0.002f, 0.1f);
            keypoint_smoother.SetParameters(smoothing_min_cutoff, smoothing_beta);
            std::cout << "Beta: " << smoothing_beta << " (more responsive)" << std::endl;
        }

        frame_count++;
        stats_frames++;

        // 每5秒打印统计信息
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - stats_start).count();
        if (elapsed >= 5) {
            float avg_fps = stats_frames / static_cast<float>(elapsed);
            std::cout << "Performance: " << avg_fps
                     << " FPS, Active tracks: " << tracks.size() << std::endl;
            stats_frames = 0;
            stats_start = now;
        }
    }

    // 清理
    if (ws_server) {
        std::cout << "Stopping WebSocket server..." << std::endl;
        ws_server->Stop();
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Pose tracking demo completed" << std::endl;
    return 0;
}
