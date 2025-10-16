
#include "cv_draw.h"

#include "utils/logging.h"

// 在img上画出检测结果
void DrawDetections(cv::Mat &img, const std::vector<Detection> &objects)
{
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto &object : objects)
    {
        cv::rectangle(img, object.box, object.color, 2);
        // class name with confidence
        std::string draw_string = object.className + " " + std::to_string(object.confidence);

        cv::putText(img, draw_string, cv::Point(object.box.x, object.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255, 0, 255), 2);
    }
}

void DrawDetectionsObb(cv::Mat& img, const std::vector<Detection>& objects)
{
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    int i = 0;
    for (const auto &object : objects)
    {
        i++;
        if (i == 30)
        {
            break;
        }
        cv::line(img, object.point1, object.point2, object.color, 2);
        cv::line(img, object.point2, object.point3, object.color, 2);
        cv::line(img, object.point3, object.point4, object.color, 2);
        cv::line(img, object.point4, object.point1, object.color, 2);
        std::string draw_string = object.className + " " + std::to_string(object.confidence);
        cv::putText(img, draw_string, object.point1, cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255, 0, 255), 2);
    }

}

void DrawCocoKps(cv::Mat &img, const std::vector<std::map<int, KeyPoint>> &keypoints)
{
    for (const auto &keypoint : keypoints)
    {
        for (const auto &keypoint_item : keypoint)
        {
            cv::circle(img, cv::Point(keypoint_item.second.x, keypoint_item.second.y), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    // draw skeleton
    // skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    //            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
    //            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    static const std::vector<std::vector<int>> joint_pairs =
        {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

    for (const auto &keypoint : keypoints)
    {
        for (const auto &joint_pair : joint_pairs)
        {
            const auto &joint1 = keypoint.find(joint_pair[0] - 1);
            const auto &joint2 = keypoint.find(joint_pair[1] - 1);
            if (joint1 != keypoint.end() && joint2 != keypoint.end())
            {
                cv::line(img, cv::Point(joint1->second.x, joint1->second.y), cv::Point(joint2->second.x, joint2->second.y),
                         cv::Scalar(0, 255, 255), 2);
            }
        }
    }
}