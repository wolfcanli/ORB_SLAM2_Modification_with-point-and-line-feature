//
// Created by jiajieshi on 22-9-5.
//

#ifndef ORB_SLAM2_LINEEXTRACTOR_H
#define ORB_SLAM2_LINEEXTRACTOR_H

#include <iostream>
#include <chrono>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/line_descriptor.hpp>

using namespace cv::line_descriptor;

namespace ORB_SLAM2 {
    class LineExtractor {
    public:
        LineExtractor();

        void ExtractLineSegment(const cv::Mat& img,
                                std::vector<KeyLine>& key_lines,
                                cv::Mat& line_descriptor,
                                std::vector<Eigen::Vector3d>& keyline_coefficients,
                                int scale = 1.2,
                                int num_octaves = 1);

    private:
        // 把一些破碎的直线合并
        // angle输入角度值
        void KeyLineMerging(std::vector<KeyLine>& key_lines,
                            cv::Mat& descriptors,
                            float angle_th = 10.0,
                            float mp_dist_th = 2.0,
                            float ep_dist_th = 15.0,
                            int desc_dist_th = 100);

        // 计算1中点到2的距离
        float MidPointToLineDistance(KeyLine& line1, KeyLine& line2);

        // 计算1和2的端点最小距离
        float EndPointMinDistance(KeyLine& line1, KeyLine& line2);

        // 计算1和2的主方向角度差
        float AngleDist(KeyLine& line1, KeyLine& line2);

        // 计算两个二进制描述子的汉明距离
        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        // 两点距离
        float PointDistance(Eigen::Vector2d& point1, Eigen::Vector2d& point2);

        // 将lines的线段合并
        void LeastSquaresMergeLines(KeyLine& merge_line, std::vector<KeyLine>& lines);

    };

} // namespace ORB_SLAM2

#endif //ORB_SLAM2_LINEEXTRACTOR_H
