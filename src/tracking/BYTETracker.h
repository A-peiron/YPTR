#pragma once

#include "STrack.h"

struct Object
{
    int classId;
    float score;
    cv::Rect_<float> box;
    std::vector<float> reid_feature; // ReID特征
    bool has_reid_extracted = false; // 标识是否已提取ReID特征
    int detection_index = -1; // 原始检测索引，用于关联keypoints
};

class BYTETracker
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	 std::vector<STrack> update(const  std::vector<Object>& objects);
    cv::Scalar get_color(int idx);
	
	// 静态调试标志控制
	static void SetDebugMode(bool debug) { debug_enabled = debug; }
	static bool IsDebugEnabled() { return debug_enabled; }

private:
	 std::vector<STrack*> joint_stracks( std::vector<STrack*> &tlista,  std::vector<STrack> &tlistb);
	 std::vector<STrack> joint_stracks( std::vector<STrack> &tlista,  std::vector<STrack> &tlistb);

	 std::vector<STrack> sub_stracks( std::vector<STrack> &tlista,  std::vector<STrack> &tlistb);
	void remove_duplicate_stracks( std::vector<STrack> &resa,  std::vector<STrack> &resb,  std::vector<STrack> &stracksa,  std::vector<STrack> &stracksb);

	void linear_assignment( std::vector< std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		 std::vector< std::vector<int> > &matches,  std::vector<int> &unmatched_a,  std::vector<int> &unmatched_b);
	 std::vector< std::vector<float> > iou_distance( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	 std::vector< std::vector<float> > iou_distance( std::vector<STrack> &atracks,  std::vector<STrack> &btracks);
	 std::vector< std::vector<float> > ious( std::vector< std::vector<float> > &atlbrs,  std::vector< std::vector<float> > &btlbrs);
	
	// ReID距离计算
	 std::vector< std::vector<float> > reid_distance( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks);
	 std::vector< std::vector<float> > fuse_iou_reid_distance( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks);
	 std::vector< std::vector<float> > occlusion_aware_matching( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks);
	
	// 改进的多级关联
	 std::vector< std::vector<float> > adaptive_distance( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks);
	bool is_similar_appearance(STrack* track, STrack& detection, float threshold = 0.3f);

	double lapjv(const  std::vector< std::vector<float> > &cost,  std::vector<int> &rowsol,  std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	float low_match_thresh;      // 低分检测匹配阈值
	float unconfirmed_thresh;    // 未确认轨迹匹配阈值
	int frame_id;
	int max_time_lost;

	 std::vector<STrack> tracked_stracks;
	 std::vector<STrack> lost_stracks;
	 std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
	
	// 静态调试标志
	static bool debug_enabled;
};
