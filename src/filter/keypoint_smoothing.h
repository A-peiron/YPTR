#ifndef KEYPOINT_SMOOTHING_H
#define KEYPOINT_SMOOTHING_H

#include "one_euro_filter.h"
#include "types/yolo_datatype.h"
#include <vector>
#include <map>

/**
 * 关键点平滑管理器
 *
 * 功能:
 * - 为多个track管理独立的滤波器
 * - 每个track有17个关键点，每个关键点有独立的x/y滤波器
 * - 自动管理滤波器生命周期
 * - 支持运行时参数调整
 *
 * 使用示例:
 * ```cpp
 * KeypointSmoothingManager smoother(30.0f, 1.0f, 0.007f);
 * for (auto& track : tracks) {
 *     if (track.state == Tracked) {
 *         smoother.SmoothKeypoints(track.track_id, keypoints[track.detection_index]);
 *     }
 * }
 * ```
 */
class KeypointSmoothingManager {
public:
    /**
     * 构造函数
     * @param freq 帧率(fps)，例如30fps
     * @param min_cutoff 基础平滑强度
     *                   0.5 = 强平滑(适合静止场景)
     *                   1.0 = 中等平滑(推荐)
     *                   2.0 = 弱平滑(适合快速运动)
     * @param beta 运动响应度
     *             0.001 = 迟钝响应
     *             0.007 = 中等响应(推荐)
     *             0.05  = 敏感响应
     * @param min_confidence 低于此置信度的关键点不进行滤波
     */
    KeypointSmoothingManager(float freq = 30.0f,
                             float min_cutoff = 1.0f,
                             float beta = 0.007f,
                             float min_confidence = 0.3f)
        : freq_(freq), min_cutoff_(min_cutoff), beta_(beta),
          min_confidence_(min_confidence), enabled_(true) {}

    /**
     * 为指定track的关键点应用平滑滤波
     * @param track_id 跟踪ID
     * @param keypoints 关键点字典（会被直接修改）
     */
    void SmoothKeypoints(int track_id, std::map<int, KeyPoint>& keypoints) {
        if (!enabled_) {
            return;
        }

        // 确保此track有滤波器
        if (track_filters_.find(track_id) == track_filters_.end()) {
            InitializeTrack(track_id);
        }

        auto& filters = track_filters_[track_id];

        for (auto& kp_pair : keypoints) {
            int kp_id = kp_pair.first;
            KeyPoint& kp = kp_pair.second;

            // 跳过低置信度关键点
            if (kp.score < min_confidence_) {
                continue;
            }

            // 确保此关键点有滤波器
            if (filters.find(kp_id) == filters.end()) {
                filters[kp_id] = KeypointFilter(freq_, min_cutoff_, beta_);
            }

            // 应用滤波（直接修改原始坐标）
            filters[kp_id].Filter(kp.x, kp.y);
        }
    }

    /**
     * 移除不再活跃的track
     * 建议在track lost或removed时调用，释放内存
     */
    void RemoveTrack(int track_id) {
        track_filters_.erase(track_id);
    }

    /**
     * 清空所有滤波器
     * 用于场景切换或重置
     */
    void Clear() {
        track_filters_.clear();
    }

    /**
     * 清理长时间未使用的track
     * @param active_track_ids 当前活跃的track ID集合
     */
    void CleanupInactiveTracks(const std::vector<int>& active_track_ids) {
        std::vector<int> to_remove;

        // 找出不在活跃列表中的track
        for (const auto& pair : track_filters_) {
            int track_id = pair.first;
            bool is_active = false;
            for (int active_id : active_track_ids) {
                if (active_id == track_id) {
                    is_active = true;
                    break;
                }
            }
            if (!is_active) {
                to_remove.push_back(track_id);
            }
        }

        // 移除非活跃track
        for (int track_id : to_remove) {
            track_filters_.erase(track_id);
        }
    }

    /**
     * 动态调整参数
     * @param min_cutoff 新的平滑强度
     * @param beta 新的运动响应度
     *
     * 调优建议:
     * - 抖动严重: 降低min_cutoff (1.0 → 0.5)
     * - 延迟明显: 提高min_cutoff (1.0 → 2.0)
     * - 快速运动跟不上: 提高beta (0.007 → 0.05)
     * - 快速运动时抖动: 降低beta (0.007 → 0.001)
     */
    void SetParameters(float min_cutoff, float beta) {
        min_cutoff_ = min_cutoff;
        beta_ = beta;

        // 更新所有现有滤波器
        for (auto& track_pair : track_filters_) {
            for (auto& kp_pair : track_pair.second) {
                kp_pair.second.SetParameters(min_cutoff, beta);
            }
        }
    }

    /**
     * 启用/禁用滤波
     * 用于A/B测试或调试
     */
    void SetEnabled(bool enabled) {
        enabled_ = enabled;
        if (!enabled) {
            Clear(); // 禁用时清空滤波器状态
        }
    }

    bool IsEnabled() const { return enabled_; }

    /**
     * 获取统计信息
     */
    size_t GetTrackCount() const {
        return track_filters_.size();
    }

    size_t GetTotalFilterCount() const {
        size_t count = 0;
        for (const auto& track_pair : track_filters_) {
            count += track_pair.second.size();
        }
        return count;
    }

private:
    void InitializeTrack(int track_id) {
        track_filters_[track_id] = std::map<int, KeypointFilter>();
    }

    float freq_;              // 帧率
    float min_cutoff_;        // 平滑强度
    float beta_;              // 运动响应度
    float min_confidence_;    // 最小置信度阈值
    bool enabled_;            // 是否启用

    // track_id -> (keypoint_id -> filter)
    std::map<int, std::map<int, KeypointFilter>> track_filters_;
};

#endif // KEYPOINT_SMOOTHING_H
