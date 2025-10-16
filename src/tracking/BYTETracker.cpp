#include "BYTETracker.h"
#include <fstream>

// 定义静态调试标志
bool BYTETracker::debug_enabled = false;

BYTETracker::BYTETracker(int frame_rate, int track_buffer)
{
	// 更保守的参数设置 - 专注于减少ID跳变和误关联
	track_thresh = 0.4;   // 稍微提高检测阈值
	high_thresh = 0.7;    // 更高的初始化阈值
	match_thresh = 0.8;   // 更严格的匹配阈值，防止远距离关联
	
	// 非常保守的参数：大幅减少错误关联
	low_match_thresh = 0.6;   // 提高低分检测的匹配阈值
	unconfirmed_thresh = 0.7; // 提高未确认轨迹的匹配阈值
	
	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer * 1.5); // 增加轨迹保持时间
}

BYTETracker::~BYTETracker()
{
}

 std::vector<STrack> BYTETracker::update(const  std::vector<Object>& objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	 std::vector<STrack> activated_stracks;
	 std::vector<STrack> refind_stracks;
	 std::vector<STrack> removed_stracks;
	 std::vector<STrack> lost_stracks;
	 std::vector<STrack> detections;
	 std::vector<STrack> detections_low;

	 std::vector<STrack> detections_cp;
	 std::vector<STrack> tracked_stracks_swap;
	 std::vector<STrack> resa, resb;
	 std::vector<STrack> output_stracks;

	 std::vector<STrack*> unconfirmed;
	 std::vector<STrack*> tracked_stracks;
	 std::vector<STrack*> strack_pool;
	 std::vector<STrack*> r_tracked_stracks;

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(4);
            tlbr_[0] = objects[i].box.x;
            tlbr_[1] = objects[i].box.y;
            tlbr_[2] = objects[i].box.x + objects[i].box.width;
            tlbr_[3] = objects[i].box.y + objects[i].box.height;

			float score = objects[i].score;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);

			// 传递detection_index
			strack.detection_index = objects[i].detection_index;

			// 添加ReID特征 - 检查是否已提取
			if (objects[i].has_reid_extracted && !objects[i].reid_feature.empty()) {
				strack.update_reid_feature(objects[i].reid_feature);
			}
			
			if (score >= track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
			
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU + ReID //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter);

	 std::vector< std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	
	// 保护机制：限制目标数量以防止匈牙利算法性能过度下降
	const int MAX_TARGETS = 100; // 最大目标数量限制
	if (strack_pool.size() > MAX_TARGETS || detections.size() > MAX_TARGETS) {
		if (debug_enabled) {
			printf("[ByteTracker] WARNING: Too many targets (tracks: %d, dets: %d), applying protection\n",
				   (int)strack_pool.size(), (int)detections.size());
		}
		// 简化处理：只保留置信度最高的目标
		if (detections.size() > MAX_TARGETS) {
			std::partial_sort(detections.begin(), detections.begin() + MAX_TARGETS, detections.end(),
							  [](const STrack& a, const STrack& b) { return a.score > b.score; });
			detections.erase(detections.begin() + MAX_TARGETS, detections.end());
		}
		if (strack_pool.size() > MAX_TARGETS) {
			// 保留最近更新的轨迹
			std::partial_sort(strack_pool.begin(), strack_pool.begin() + MAX_TARGETS, strack_pool.end(),
							  [](STrack* a, STrack* b) { return a->frame_id > b->frame_id; });
			strack_pool.erase(strack_pool.begin() + MAX_TARGETS, strack_pool.end());
		}
	}
	
	// 使用简单的IoU+ReID融合 (4:6权重)
	dists = fuse_iou_reid_distance(strack_pool, detections);
	if (!dists.empty() && !dists[0].empty()) {
		dist_size = dists.size();
		dist_size_size = dists[0].size();
	} else {
		// 如果融合距离失败，回退到IoU距离
		dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);
	}

	 std::vector< std::vector<int> > matches;
	 std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	
	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, low_match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	 std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, unconfirmed_thresh, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}

std::vector<std::vector<float>> BYTETracker::reid_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks)
{
	std::vector<std::vector<float>> cost_matrix;
	if (atracks.empty() || btracks.empty()) {
		return cost_matrix;
	}
	
	cost_matrix.resize(atracks.size());
	
	for (int i = 0; i < atracks.size(); i++) {
		cost_matrix[i].resize(btracks.size());
		for (int j = 0; j < btracks.size(); j++) {
			if (atracks[i]->has_reid_feature && btracks[j].has_reid_feature) {
				cost_matrix[i][j] = atracks[i]->compute_reid_distance(btracks[j].reid_feature);
			} else {
				cost_matrix[i][j] = 1.0f; // 最大距离
			}
		}
	}
	
	return cost_matrix;
}

std::vector<std::vector<float>> BYTETracker::fuse_iou_reid_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks)
{
	// 计算IoU距离
	int dist_size = 0, dist_size_size = 0;
	std::vector<std::vector<float>> iou_dists = iou_distance(atracks, btracks, dist_size, dist_size_size);
	
	// 如果没有轨迹或检测，直接返回IoU距离
	if (atracks.empty() || btracks.empty()) {
		return iou_dists;
	}
	
	// 检查是否有任何ReID特征
	bool has_any_reid = false;
	int reid_track_count = 0, reid_det_count = 0;
	
	for (int i = 0; i < atracks.size(); i++) {
		if (atracks[i]->has_reid_feature) {
			has_any_reid = true;
			reid_track_count++;
		}
	}
	for (int j = 0; j < btracks.size(); j++) {
		if (btracks[j].has_reid_feature) {
			has_any_reid = true;
			reid_det_count++;
		}
	}
	
	if (debug_enabled) {
		printf("[ByteTracker] Tracks: %d (ReID: %d), Detections: %d (ReID: %d)\n", 
			   (int)atracks.size(), reid_track_count, (int)btracks.size(), reid_det_count);
	}
	
	// 如果没有任何ReID特征，直接返回IoU距离
	if (!has_any_reid) {
		if (debug_enabled) {
			printf("[ByteTracker] No ReID features found, using IoU only\n");
		}
		return iou_dists;
	}
	
	// 计算ReID距离
	std::vector<std::vector<float>> reid_dists = reid_distance(atracks, btracks);
	
	// 保守的IoU+ReID融合策略 - 加强空间约束，支持间隔ReID提取
	std::vector<std::vector<float>> fused_dists = iou_dists;
	int fused_count = 0;
	int spatial_reject_count = 0;
	int no_reid_pairs = 0;
	
	for (int i = 0; i < atracks.size(); i++) {
		for (int j = 0; j < btracks.size(); j++) {
			float iou_dist = iou_dists[i][j];
			float final_dist = iou_dist; // 默认使用IoU距离
			
			// 只有当双方都有ReID特征时才考虑融合
			if (atracks[i]->has_reid_feature && btracks[j].has_reid_feature) {
				float reid_dist = reid_dists[i][j];
				
				// 计算空间约束
				std::vector<float> track_tlwh = atracks[i]->tlwh;
				std::vector<float> det_tlwh = btracks[j].tlwh;
				
				float track_cx = track_tlwh[0] + track_tlwh[2] / 2.0f;
				float track_cy = track_tlwh[1] + track_tlwh[3] / 2.0f;
				float det_cx = det_tlwh[0] + det_tlwh[2] / 2.0f;
				float det_cy = det_tlwh[1] + det_tlwh[3] / 2.0f;
				
				float center_dist = sqrt(pow(track_cx - det_cx, 2) + pow(track_cy - det_cy, 2));
				float avg_size = (track_tlwh[2] + track_tlwh[3] + det_tlwh[2] + det_tlwh[3]) / 4.0f;
				
				// 简化的融合策略：基于空间距离和轨迹长度的自适应权重
				float max_spatial_dist = avg_size * 2.0f;
				bool spatial_valid = center_dist <= max_spatial_dist;
				
				float reid_weight = 0.0f;
				
				if (spatial_valid) {
					// 基础ReID权重：根据轨迹稳定性调整
					float base_weight = 0.3f;
					
					// 轨迹越稳定，越信任ReID特征
					if (atracks[i]->tracklet_len >= 5) {
						base_weight = 0.4f; // 稳定轨迹增加ReID权重
					}
					
					// 根据ReID质量调整权重
					if (reid_dist < 0.3f) {
						reid_weight = base_weight; // ReID相似度高，使用基础权重
					} else if (reid_dist < 0.6f) {
						reid_weight = base_weight * 0.5f; // ReID相似度中等，降低权重
					} else {
						reid_weight = 0.0f; // ReID相似度低，不使用
					}
					
					// IoU很好时降低ReID权重，避免过度依赖
					if (iou_dist < 0.3f) {
						reid_weight *= 0.7f;
					}
				}
				
				// 简单的线性融合
				final_dist = (1.0f - reid_weight) * iou_dist + reid_weight * reid_dist;
				
				// 对空间约束违反的配对增加惩罚
				if (!spatial_valid) {
					final_dist = std::min(1.0f, final_dist + 0.3f);
					spatial_reject_count++;
				}
				
				if (reid_weight > 0) fused_count++;
			} else {
				// 没有ReID特征的情况，直接使用IoU距离
				no_reid_pairs++;
			}
			
			fused_dists[i][j] = final_dist;
		}
	}
	
	if (debug_enabled) {
		printf("[ByteTracker] Interval ReID fusion: %d pairs with ReID, %d without ReID, rejected %d by spatial constraint\n", 
			   fused_count, no_reid_pairs, spatial_reject_count);
	}
	return fused_dists;
}

// 检查外观相似性
bool BYTETracker::is_similar_appearance(STrack* track, STrack& detection, float threshold)
{
	if (!track->has_reid_feature || !detection.has_reid_feature) {
		return false;
	}
	
	float similarity = 0.0f;
	for (size_t i = 0; i < track->reid_feature.size() && i < detection.reid_feature.size(); i++) {
		similarity += track->reid_feature[i] * detection.reid_feature[i];
	}
	
	return similarity > threshold;
}

// 自适应距离计算 - 改进版本，防止远距离跳变
std::vector<std::vector<float>> BYTETracker::adaptive_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks)
{
	std::vector<std::vector<float>> iou_dists;
	int dist_size = 0, dist_size_size = 0;
	iou_dists = iou_distance(atracks, btracks, dist_size, dist_size_size);
	
	if (atracks.empty() || btracks.empty()) {
		return iou_dists;
	}
	
	std::vector<std::vector<float>> reid_dists = reid_distance(atracks, btracks);
	std::vector<std::vector<float>> adaptive_dists = iou_dists;
	
	for (int i = 0; i < atracks.size(); i++) {
		for (int j = 0; j < btracks.size(); j++) {
			if (atracks[i]->has_reid_feature && btracks[j].has_reid_feature) {
				float iou_dist = iou_dists[i][j];
				float reid_dist = reid_dists[i][j];
				
				// 计算中心点距离
				std::vector<float> track_tlwh = atracks[i]->tlwh;
				std::vector<float> det_tlwh = btracks[j].tlwh;
				float center_x_track = track_tlwh[0] + track_tlwh[2] / 2;
				float center_y_track = track_tlwh[1] + track_tlwh[3] / 2;
				float center_x_det = det_tlwh[0] + det_tlwh[2] / 2;
				float center_y_det = det_tlwh[1] + det_tlwh[3] / 2;
				
				float center_dist = sqrt(pow(center_x_track - center_x_det, 2) + 
										pow(center_y_track - center_y_det, 2));
				
				// 计算框的平均尺寸作为距离阈值
				float avg_size = (track_tlwh[2] + track_tlwh[3] + det_tlwh[2] + det_tlwh[3]) / 4;
				float max_center_dist = avg_size * 3.0f; // 允许的最大中心距离
				
				// 距离太远时，惩罚ReID权重
				float reid_weight = 0.3f; // 基础权重较保守
				
				if (center_dist > max_center_dist) {
					// 距离过远，主要依赖位置
					reid_weight = 0.1f;
				} else if (iou_dist < 0.5f) {
					// IoU较好，平衡使用
					reid_weight = 0.4f;
				} else if (reid_dist < 0.3f && center_dist < max_center_dist * 0.5f) {
					// ReID相似且距离近，稍微增强ReID
					reid_weight = 0.5f;
				}
				
				adaptive_dists[i][j] = (1.0f - reid_weight) * iou_dist + reid_weight * reid_dist;
				
				// 对远距离匹配增加额外惩罚
				if (center_dist > max_center_dist) {
					adaptive_dists[i][j] += 0.3f; // 增加惩罚项
				}
			}
		}
	}
	
	if (debug_enabled) {
		printf("[ByteTracker] Used conservative adaptive distance matching\n");
	}
	return adaptive_dists;
}

// 遮挡感知匹配算法 - 专注于稳定性和遮挡处理
std::vector<std::vector<float>> BYTETracker::occlusion_aware_matching(std::vector<STrack*> &atracks, std::vector<STrack> &btracks)
{
	// 计算基础IoU距离
	int dist_size = 0, dist_size_size = 0;
	std::vector<std::vector<float>> iou_dists = iou_distance(atracks, btracks, dist_size, dist_size_size);
	
	if (atracks.empty() || btracks.empty()) {
		return iou_dists;
	}
	
	// 计算ReID距离
	std::vector<std::vector<float>> reid_dists = reid_distance(atracks, btracks);
	std::vector<std::vector<float>> stable_dists = iou_dists;
	
	// 检测遮挡情况并调整策略
	for (int i = 0; i < atracks.size(); i++) {
		for (int j = 0; j < btracks.size(); j++) {
			float iou_dist = iou_dists[i][j];
			float final_dist = iou_dist;
			
			// 只有在有ReID特征时才考虑融合
			if (atracks[i]->has_reid_feature && btracks[j].has_reid_feature) {
				float reid_dist = reid_dists[i][j];
				
				// 计算空间位置关系
				std::vector<float> track_tlwh = atracks[i]->tlwh;
				std::vector<float> det_tlwh = btracks[j].tlwh;
				
				float track_cx = track_tlwh[0] + track_tlwh[2] / 2.0f;
				float track_cy = track_tlwh[1] + track_tlwh[3] / 2.0f;
				float det_cx = det_tlwh[0] + det_tlwh[2] / 2.0f;
				float det_cy = det_tlwh[1] + det_tlwh[3] / 2.0f;
				
				float center_dist = sqrt(pow(track_cx - det_cx, 2) + pow(track_cy - det_cy, 2));
				float avg_size = (track_tlwh[2] + track_tlwh[3] + det_tlwh[2] + det_tlwh[3]) / 4.0f;
				
				// 保守的ReID权重策略
				float reid_weight = 0.0f;  // 默认不使用ReID
				
				// 只在以下情况使用ReID:
				// 1. IoU很低但距离很近 (可能遮挡)
				// 2. 轨迹稳定且ReID相似度很高
				if (iou_dist > 0.7f && center_dist < avg_size * 2.0f) {
					// 可能的遮挡情况，小心使用ReID
					if (reid_dist < 0.3f && atracks[i]->tracklet_len > 5) {
						reid_weight = 0.2f;  // 低权重
					}
				} else if (iou_dist < 0.5f) {
					// IoU较好的情况下，更小心使用ReID
					if (reid_dist < 0.25f && atracks[i]->tracklet_len > 10) {
						reid_weight = 0.15f;  // 非常低的权重
					}
				}
				
				// 距离约束 - 防止远距离跳变
				float max_allowed_dist = avg_size * 3.0f;  // 更严格的约束
				if (center_dist > max_allowed_dist) {
					reid_weight = 0.0f;  // 完全禁用ReID
					final_dist = 1.0f;   // 设为最大距离
				} else {
					final_dist = (1.0f - reid_weight) * iou_dist + reid_weight * reid_dist;
				}
				
				// 额外的稳定性检查
				if (reid_weight > 0.0f) {
					// 检查速度一致性
					if (atracks[i]->tracklet_len >= 2) {
						float movement = center_dist;
						float expected_max_movement = avg_size * 1.5f;  // 预期最大移动距离
						
						if (movement > expected_max_movement) {
							final_dist += 0.3f;  // 惩罚过大移动
						}
					}
				}
			}
			
			stable_dists[i][j] = std::min(1.0f, std::max(0.0f, final_dist));
		}
	}
	
	if (debug_enabled) {
		printf("[ByteTracker] Occlusion-aware matching: conservative strategy applied\n");
	}
	return stable_dists;
}
