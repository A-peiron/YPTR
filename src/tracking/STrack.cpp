#include "STrack.h"

STrack::STrack( std::vector<float> tlwh_, float score)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;
	track_id = 0;
	state = TrackState::New;

	tlwh.resize(4);
	tlbr.resize(4);

	static_tlwh();
	static_tlbr();
	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	start_frame = 0;

	// 初始化ReID特征
	has_reid_feature = false;
	reid_last_updated_frame = -1;

	// 初始化detection_index
	detection_index = -1;
}

STrack::~STrack()
{
}

void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	 std::vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	 std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	if (frame_id == 1)
	{
		this->is_activated = true;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

void STrack::re_activate(STrack &new_track, int frame_id, bool new_id)
{
	 std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id)
		this->track_id = next_id();

	// 更新ReID特征
	if (new_track.has_reid_feature) {
		this->update_reid_feature(new_track.reid_feature);
	}

	// 更新detection_index
	this->detection_index = new_track.detection_index;
}

void STrack::update(STrack &new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	 std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->state = TrackState::Tracked;
	this->is_activated = true;

	this->score = new_track.score;

	// 更新ReID特征 - 但对稳定轨迹要求更高的相似性
	if (new_track.has_reid_feature) {
		if (!this->has_reid_feature) {
			// 如果轨迹还没有ReID特征，直接采用
			this->update_reid_feature(new_track.reid_feature);
		} else if (is_track_stable()) {
			// 对稳定轨迹，检查特征相似性再决定是否更新
			float similarity = 0.0f;
			std::vector<float> temp_fused_feature = reid_feature; // 临时存储融合结果

			// 在一次循环中同时计算相似性和准备融合特征
			for (size_t i = 0; i < reid_feature.size() && i < new_track.reid_feature.size(); i++) {
				similarity += reid_feature[i] * new_track.reid_feature[i];
				temp_fused_feature[i] = 0.7f * reid_feature[i] + 0.3f * new_track.reid_feature[i];
			}

			// 只有当相似性足够高时才应用融合结果，避免特征漂移
			if (similarity > 0.7f) {
				reid_feature = std::move(temp_fused_feature);
			}
		} else {
			// 不稳定轨迹可以更自由地更新特征
			this->update_reid_feature(new_track.reid_feature);
		}
	}

	// 更新detection_index
	this->detection_index = new_track.detection_index;
}

void STrack::static_tlwh()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

void STrack::static_tlbr()
{
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

 std::vector<float> STrack::tlwh_to_xyah( std::vector<float> tlwh_tmp)
{
	 std::vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

 std::vector<float> STrack::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

 std::vector<float> STrack::tlbr_to_tlwh( std::vector<float> &tlbr)
{
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

void STrack::mark_lost()
{
	state = TrackState::Lost;
}

void STrack::mark_removed()
{
	state = TrackState::Removed;
}

int STrack::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

int STrack::end_frame()
{
	return this->frame_id;
}

void STrack::multi_predict( std::vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		if (stracks[i]->state != TrackState::Tracked)
		{
			stracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
	}
}

void STrack::update_reid_feature(const std::vector<float>& feature)
{
	if (!feature.empty()) {
		reid_feature = feature;
		has_reid_feature = true;
		reid_last_updated_frame = frame_id; // 记录更新帧号
	}
}

float STrack::compute_reid_distance(const std::vector<float>& other_feature) const
{
	if (!has_reid_feature || other_feature.empty() || reid_feature.size() != other_feature.size()) {
		return 1.0f; // 最大距离
	}
	
	// 计算余弦距离 (1 - 余弦相似度)
	float dot_product = 0.0f;
	for (size_t i = 0; i < reid_feature.size(); i++) {
		dot_product += reid_feature[i] * other_feature[i];
	}
	
	// 余弦相似度转余弦距离
	return 1.0f - dot_product;
}

// 检查轨迹是否稳定
bool STrack::is_track_stable() const
{
	// 轨迹需要至少存在3帧且被激活
	return is_activated && tracklet_len >= 3 && state == TrackState::Tracked;
}

// 检查ReID特征是否还有效（考虑时间因素）
bool STrack::is_reid_feature_valid(int current_frame, int max_age) const
{
	if (!has_reid_feature) return false;
	return (current_frame - reid_last_updated_frame) <= max_age;
}

// 检查新位置是否合理
bool STrack::is_position_reasonable(const STrack& new_track) const
{
	if (!is_track_stable()) {
		return true; // 新轨迹不需要约束
	}
	
	// 计算中心点距离
	float this_cx = tlwh[0] + tlwh[2] / 2.0f;
	float this_cy = tlwh[1] + tlwh[3] / 2.0f;
	float new_cx = new_track.tlwh[0] + new_track.tlwh[2] / 2.0f;
	float new_cy = new_track.tlwh[1] + new_track.tlwh[3] / 2.0f;
	
	float center_dist = sqrt(pow(this_cx - new_cx, 2) + pow(this_cy - new_cy, 2));
	float avg_size = (tlwh[2] + tlwh[3] + new_track.tlwh[2] + new_track.tlwh[3]) / 4.0f;
	
	// 对稳定轨迹应用更严格的移动约束
	float max_movement = avg_size * 1.5f; // 相对保守的约束
	
	// 额外检查尺寸变化
	float size_change = abs(tlwh[2] * tlwh[3] - new_track.tlwh[2] * new_track.tlwh[3]) / 
						(tlwh[2] * tlwh[3]);
	
	return center_dist <= max_movement && size_change <= 0.5f; // 50%的尺寸变化限制
}