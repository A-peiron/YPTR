#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack( std::vector<float> tlwh_, float score);
	~STrack();

	 std::vector<float> static tlbr_to_tlwh( std::vector<float> &tlbr);
	void static multi_predict( std::vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	 std::vector<float> tlwh_to_xyah( std::vector<float> tlwh_tmp);
	 std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

	// ReID相关方法
	void update_reid_feature(const std::vector<float>& feature);
	float compute_reid_distance(const std::vector<float>& other_feature) const;
	
	// 轨迹稳定性检查
	bool is_track_stable() const;
	bool is_position_reasonable(const STrack& new_track) const;
	bool is_reid_feature_valid(int current_frame, int max_age = 10) const; // ReID特征有效性检查

public:
	bool is_activated;
	int track_id;
	int state;

	 std::vector<float> _tlwh;
	 std::vector<float> tlwh;
	 std::vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

	// ReID特征
	std::vector<float> reid_feature;
	bool has_reid_feature;
	int reid_last_updated_frame; // ReID特征最后更新的帧号

	// 原始检测索引，用于关联keypoints
	int detection_index;

private:
	byte_kalman::KalmanFilter kalman_filter;
};