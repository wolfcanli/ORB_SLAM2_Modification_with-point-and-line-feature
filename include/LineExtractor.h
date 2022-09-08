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

    };

} // namespace ORB_SLAM2

#endif //ORB_SLAM2_LINEEXTRACTOR_H
