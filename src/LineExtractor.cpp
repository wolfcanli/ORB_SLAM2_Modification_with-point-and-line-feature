//
// Created by jiajieshi on 22-9-5.
//

#include "LineExtractor.h"

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

    int nums_lineFeature = 80;
//    std::cout << "filter lines" << std::endl;
    if (key_lines.size() > nums_lineFeature) {
        // response是线长度与图像宽高最大值的比值
        // 这里相当于是将线段长度从大到小排序，只取前lsdNFeatures个
        std::sort(key_lines.begin(), key_lines.end(),
                  [](const KeyLine& a, const KeyLine& b)-> bool {return (a.response > b.response);});
        key_lines.resize(nums_lineFeature);
        // class id，线特征的id，可以用来聚类，不过这里不聚，只是重新设置id
//        for (int i = 0; i < nums_lineFeature; i++) {
//            key_lines[i].class_id = i;
//        }
    }

    // 上面先筛选一部分，原本提取好几百个，但里面很多短小的线段，也没必要合并
    // 先筛选一部分长一点的，在这部分里面找合并的线段
    // TODO 也不知道稳不稳定
    // 先提取描述子，回头还会删掉重来
//    lbd->compute(img, key_lines, line_descriptor);
//    KeyLineMerging(key_lines, line_descriptor);
//    int nums_KeyLine = 50;
//    if (key_lines.size() > nums_KeyLine) {
//        std::sort(key_lines.begin(), key_lines.end(),
//                  [](const KeyLine& a, const KeyLine& b)-> bool {return (a.response > b.response);});
//        key_lines.resize(nums_KeyLine);
//        for (int i = 0; i < nums_KeyLine; i++) {
//            key_lines[i].class_id = i;
//        }
//    }

//    std::cout << "KeyLines.size() == " << key_lines.size() << std::endl;

    // 这次是合并之后的描述子提取
    lbd->compute(img, key_lines, line_descriptor);


    // 计算直线方程系数，两端点齐次坐标叉乘即可
    for (std::vector<KeyLine>::iterator it = key_lines.begin(); it != key_lines.end(); it++) {
        Eigen::Vector3d start_point(it->startPointX, it->startPointY, 1.0);
        Eigen::Vector3d end_point(it->endPointX, it->endPointY, 1.0);

        Eigen::Vector3d line_coefficient;
        line_coefficient << start_point.cross(end_point);
        // normalization
        line_coefficient.normalize();
        keyline_coefficients.push_back(line_coefficient);
    }
}

/**
 * @brief 把一些破碎的直线合并
 * Reference to 《基于点线综合特征的双目视觉SLAM方法》谢晓佳
 * 主要通过角度，中点到另一条线段距离，线段端点距离最小值
 * @param[in,out] key_lines   需要合并的线特征，线段长度从大到小
 * @param[in] descriptors     描述子
 * @param[in] angle_th        角度差阈值
 * @param[in] mp_dist_th      线段中点到另一条线段距离
 * @param[in] ep_dist_th      线段端点距离最小值
 * @param[in] desc_dist_th    描述子距离
 */
void LineExtractor::KeyLineMerging(std::vector<KeyLine>& key_lines,
                                   cv::Mat& descriptors,
                                   float angle_th, float mp_dist_th,
                                   float ep_dist_th, int desc_dist_th) {
    // KeyLine里的角度是弧度制
    angle_th = angle_th * M_PI / 180.0;
    std::vector<KeyLine> key_lines_merge;

    // 条件判断，能合并的将class id设置成相同的
    // 先全部设成0，然后从1开始计数把
    // TODO 先这么弄
    for (int i = 0; i < key_lines.size(); i++) {
        key_lines[i].class_id = 0;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // TODO 测试了一下有点费时，先这么弄吧
    // id：0 初始，x有匹配线段，-1没有匹配线段
    int id = 1;
    int count = 0;
    for (int i = 0; i < key_lines.size(); i++) {
        // 如果这条线已经被分配了id，跳过
        if (key_lines[i].class_id != 0)
            continue;
        key_lines[i].class_id = id;
        count = 0;
        // 每个keyline都与所有线段判断，满足条件设置成同样的class id
        // 之后如果遇到已经分配id了的就直接跳过
        // TODO n^2时间复杂度肯定慢，先这么弄把
        for (int j = 0; j < key_lines.size(); j++) {
            if (j == i)
                continue;
            KeyLine kl_j = key_lines[j];
            // 已经分配过了，跳过
            if (key_lines[j].class_id != 0 && key_lines[j].class_id < id)
                continue;
            // 角度差
            if (AngleDist(key_lines[i], key_lines[j]) > angle_th)
                continue;
            // 中点到线距离
            if (MidPointToLineDistance(key_lines[i], key_lines[j]) > mp_dist_th)
                continue;
            // 端点最小距离
            if (EndPointMinDistance(key_lines[i], key_lines[j]) > ep_dist_th)
                continue;
            // 描述子距离
            if (DescriptorDistance(descriptors.row(i), descriptors.row(j)) > desc_dist_th)
                continue;

            // 四个条件全都满足，设置id，再自增
            key_lines[j].class_id = id;
            count ++;
        }
        // 如果没找到匹配的线段，则把id设置成-1，方面后面搜索排除
        if (count == 0) {
            key_lines[i].class_id = -1;
        } else {
            id++;
//            std::cout << "key_lines " + std::to_string(i) + " 匹配线段数量 " << count << std::endl;

            // TODO 最小二乘合并，需要重新计算KeyLine中的内容
            // 合并前先找出同样id的KeyLine, id：0 初始，x有匹配线段，-1没有匹配线段


        }
    }



    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "Lines to merge match cost " << ttrack * 1000 << " ms" << std::endl;

}

// 将lines的线段合并
void LineExtractor::LeastSquaresMergeLines(KeyLine& merge_line, std::vector<KeyLine>& lines) {
    // TODO 最小二乘合并曲线，并重新计算KeyLine属性
}

// 计算1中点到2的距离
float LineExtractor::MidPointToLineDistance(KeyLine& line1, KeyLine& line2) {
    Eigen::Vector3d mid_point_i(line1.pt.x, line1.pt.y, 1.0);
    float a = line2.endPointY - line2.startPointY;
    float b = line2.startPointX - line2.endPointX;
    float c = line2.startPointY * (line2.endPointX - line2.startPointX) -
            line2.startPointX * (line2.endPointY - line2.startPointY);
    Eigen::Vector3d coefficient(a, b, c);
    coefficient.normalize();
    float dist_mp_line = mid_point_i.dot(coefficient);

    return dist_mp_line;
}

// 计算1和2的端点最小距离
float LineExtractor::EndPointMinDistance(KeyLine& line1, KeyLine& line2) {
    // line (startPointX startPointY endPointX endPointY)
    Eigen::Vector2d line1_start_point(line1.startPointX, line1.startPointY);
    Eigen::Vector2d line1_end_point(line1.endPointX, line1.endPointY);

    Eigen::Vector2d line2_start_point(line2.startPointX, line2.startPointY);
    Eigen::Vector2d line2_end_point(line2.endPointX, line2.endPointY);

    float dist_line1s_line2s = PointDistance(line1_start_point, line2_start_point);
    float dist_line1s_line2e = PointDistance(line1_start_point, line2_end_point);

    float dist_line1e_line2s = PointDistance(line1_end_point, line2_start_point);
    float dist_line1e_line2e = PointDistance(line1_end_point, line2_end_point);

    // 找出最小值
    float dist_min = std::min(std::min(dist_line1s_line2s, dist_line1s_line2e),
                              std::min(dist_line1e_line2s, dist_line1e_line2e));

    return dist_min;
}

    // 两点距离
float LineExtractor::PointDistance(Eigen::Vector2d& point1, Eigen::Vector2d& point2) {
    return sqrt(pow((point2(0) - point1(0)), 2) + pow((point2(1) - point1(1)), 2));
}

// 计算1和2的主方向角度差
float LineExtractor::AngleDist(KeyLine& line1, KeyLine& line2) {
    return abs(line1.angle - line2.angle);
}

// Bit set count operation from
// Hamming distance：两个二进制串之间的汉明距离，指的是其不同位数的个数
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int LineExtractor::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    // 8*32=256bit
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;        // 相等为0,不等为1
        // 下面的操作就是计算其中bit为1的个数了,这个操作看上面的链接就好
        // 其实我觉得也还阔以直接使用8bit的查找表,然后做32次寻址操作就完成了;不过缺点是没有利用好CPU的字长
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} // namespace ORB_SLAM2