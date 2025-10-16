#ifndef ONE_EURO_FILTER_H
#define ONE_EURO_FILTER_H

#include <cmath>
#include <chrono>

/**
 * 低通滤波器 - One Euro Filter的基础组件
 * 使用指数移动平均(EMA)实现
 */
class LowPassFilter {
public:
    LowPassFilter() : initialized_(false), prev_value_(0.0f) {}

    float Filter(float value, float alpha) {
        if (!initialized_) {
            initialized_ = true;
            prev_value_ = value;
            return value;
        }
        float filtered = alpha * value + (1.0f - alpha) * prev_value_;
        prev_value_ = filtered;
        return filtered;
    }

    void Reset() {
        initialized_ = false;
        prev_value_ = 0.0f;
    }

    float prev_value_; // 公开访问用于速度计算

private:
    bool initialized_;
};

/**
 * One Euro Filter - 自适应低通滤波器
 *
 * 论文: Casiez, G., Roussel, N. and Vogel, D. (2012).
 *       1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
 *
 * 特性:
 * - 静止时强平滑，减少抖动
 * - 运动时弱平滑，保持响应性
 * - 零延迟，极低计算开销
 *
 * 参数说明:
 * - freq: 采样频率(Hz)，例如30fps则为30
 * - min_cutoff: 最小截止频率(Hz)，控制静止时的平滑强度
 *               值越小越平滑，推荐范围: 0.5-2.0
 *               建议: 0.5(强平滑), 1.0(中等), 2.0(弱平滑)
 * - beta: 速度系数，控制对快速运动的响应度
 *         值越大对运动越敏感，推荐范围: 0.001-0.1
 *         建议: 0.001(迟钝), 0.007(中等), 0.05(敏感)
 * - d_cutoff: 速度滤波的截止频率(Hz)，通常固定为1.0
 */
class OneEuroFilter {
public:
    OneEuroFilter(float freq = 30.0f, float min_cutoff = 1.0f,
                  float beta = 0.007f, float d_cutoff = 1.0f)
        : freq_(freq), min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff),
          last_time_(0), initialized_(false) {}

    /**
     * 滤波函数
     * @param value 原始输入值
     * @param timestamp 时间戳(秒)，如果<0则使用系统时间
     * @return 滤波后的值
     */
    float Filter(float value, float timestamp = -1.0f) {
        // 如果没有提供时间戳，使用系统时间
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<float>(now.time_since_epoch()).count();
        }

        if (!initialized_) {
            initialized_ = true;
            last_time_ = timestamp;
            dx_filter_.Reset();
            x_filter_.Reset();
            return x_filter_.Filter(value, 1.0f);
        }

        // 计算时间间隔
        float dt = timestamp - last_time_;
        last_time_ = timestamp;

        // 避免除零或异常值
        if (dt <= 0 || dt > 1.0f) {
            dt = 1.0f / freq_;
        }

        // 计算速度（变化率）
        float dx = (value - x_filter_.prev_value_) / dt;

        // 对速度进行低通滤波
        float alpha_d = Alpha(d_cutoff_, dt);
        float dx_filtered = dx_filter_.Filter(dx, alpha_d);

        // 根据速度自适应调整截止频率
        // 速度越快，截止频率越高，平滑越弱
        float cutoff = min_cutoff_ + beta_ * std::abs(dx_filtered);

        // 对原始值进行低通滤波
        float alpha = Alpha(cutoff, dt);
        float filtered_value = x_filter_.Filter(value, alpha);

        return filtered_value;
    }

    void Reset() {
        initialized_ = false;
        dx_filter_.Reset();
        x_filter_.Reset();
    }

    // 在线调整参数
    void SetMinCutoff(float min_cutoff) { min_cutoff_ = min_cutoff; }
    void SetBeta(float beta) { beta_ = beta; }
    void SetFreq(float freq) { freq_ = freq; }

private:
    // 计算平滑系数alpha
    // alpha越大，越接近原始值（平滑越弱）
    float Alpha(float cutoff, float dt) {
        float tau = 1.0f / (2.0f * M_PI * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    float freq_;          // 采样频率
    float min_cutoff_;    // 最小截止频率
    float beta_;          // 速度系数
    float d_cutoff_;      // 速度滤波截止频率
    float last_time_;     // 上一帧时间戳
    bool initialized_;    // 是否已初始化

    LowPassFilter x_filter_;   // 值滤波器
    LowPassFilter dx_filter_;  // 速度滤波器
};

/**
 * 关键点滤波器 - 同时处理x和y坐标
 * 为2D点提供独立的x/y滤波
 */
class KeypointFilter {
public:
    KeypointFilter(float freq = 30.0f, float min_cutoff = 1.0f,
                   float beta = 0.007f, float d_cutoff = 1.0f)
        : x_filter_(freq, min_cutoff, beta, d_cutoff),
          y_filter_(freq, min_cutoff, beta, d_cutoff) {}

    /**
     * 滤波2D点
     * @param x 输入/输出x坐标（会被修改）
     * @param y 输入/输出y坐标（会被修改）
     * @param timestamp 时间戳，如果<0则使用系统时间
     */
    void Filter(float& x, float& y, float timestamp = -1.0f) {
        x = x_filter_.Filter(x, timestamp);
        y = y_filter_.Filter(y, timestamp);
    }

    void Reset() {
        x_filter_.Reset();
        y_filter_.Reset();
    }

    void SetParameters(float min_cutoff, float beta) {
        x_filter_.SetMinCutoff(min_cutoff);
        y_filter_.SetMinCutoff(min_cutoff);
        x_filter_.SetBeta(beta);
        y_filter_.SetBeta(beta);
    }

    void SetFreq(float freq) {
        x_filter_.SetFreq(freq);
        y_filter_.SetFreq(freq);
    }

private:
    OneEuroFilter x_filter_;
    OneEuroFilter y_filter_;
};

#endif // ONE_EURO_FILTER_H
