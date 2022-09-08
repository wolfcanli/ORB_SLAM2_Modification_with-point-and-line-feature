//
// Created by jiajieshi on 22-9-5.
//

#include "LineExtractor.h"

struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.response > b.response );
    }
};

namespace ORB_SLAM2 {
LineExtractor::LineExtractor() {

}

void LineExtractor::ExtractLineSegment(const cv::Mat &img,
                                       std::vector<KeyLine> &key_lines,
                                       cv::Mat &line_descriptor,
                                       std::vector<Eigen::Vector3d> &keyline_coefficients,
                                       int scale,
                                       int num_octaves) {
    cv::Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    cv::Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
//    std::cout << "extract line segments" << std::endl;
    lsd->detect(img, key_lines, scale, num_octaves);

    int lsdNFeatures = 50;
//    std::cout << "filter lines" << std::endl;
    if (key_lines.size() > lsdNFeatures) {
        std::sort(key_lines.begin(), key_lines.end(), sort_lines_by_response());
        key_lines.resize(lsdNFeatures);
        for (int i = 0; i < lsdNFeatures; i++) {
            key_lines[i].class_id = i;
        }
    }

    std::cout << "KeyLines.size() == " << key_lines.size() << std::endl;

    lbd->compute(img, key_lines, line_descriptor);

    // 计算直线方程系数，两端点齐次坐标叉乘即可
    for (std::vector<KeyLine>::iterator it = key_lines.begin(); it != key_lines.end(); it++) {
        Eigen::Vector3d start_point(it->startPointX, it->startPointY, 1.0);
        Eigen::Vector3d end_point(it->endPointX, it->endPointY, 1.0);

        Eigen::Vector3d line_coefficient;
        line_coefficient << start_point.cross(end_point);
        // normalization
        line_coefficient = line_coefficient / sqrt(line_coefficient[0] * line_coefficient[0] + line_coefficient[1] * line_coefficient[1]);
        keyline_coefficients.push_back(line_coefficient);
    }
}


} // namespace ORB_SLAM2