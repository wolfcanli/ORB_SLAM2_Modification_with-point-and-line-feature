//
// Created by jiajieshi on 22-9-5.
//

#ifndef ORB_SLAM2_LINEMATCHER_H
#define ORB_SLAM2_LINEMATCHER_H

#include<vector>
#include<limits.h>
#include<stdint.h>
#include <chrono>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include"KeyFrame.h"
#include"Frame.h"
#include "MapLine.h"

using namespace cv::line_descriptor;

namespace ORB_SLAM2 {
class Frame;
class KeyFrame;
class MapLine;

class LineMatcher {
public:
    /**
     * Constructor
     * @param nnratio  ratio of the best and the second score   最优和次优评分的比例
     * @param checkOri check orientation                        是否检查方向
     */
    LineMatcher(float nnratio=0.6, bool checkOri=true);

    /**
     * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
     * @param[in] a     一个描述子
     * @param[in] b     另外一个描述子
     * @return int      描述子的汉明距离
     */
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    // 这个用来和上一帧匹配
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame);
    // TODO 单元测试用，这里测试可行，先留着，回头删了
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, std::vector<KeyLine>& new_kls, std::vector<std::pair<int, int>>& match_indices);

    // 加一个用来和参考关键帧匹配的
    // vpMapLineMatches CurrentFrame中地图点对应的匹配，NULL表示未匹配
    // TODO 这里为啥特地弄个MapLines的vector，根据Tracking之后的步骤来看，好像可以直接在实现中把MapLine给到当前帧
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *RefFrame, std::vector<MapLine*> &vpMapLineMatches);
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *RefFrame);
    // TODO 单元测试用，这里测试可行，先留着，回头删了
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *RefFrame, std::vector<KeyLine>& new_kls, std::vector<std::pair<int, int>>& match_indices);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    // 这个用来和局部地图匹配
    int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines);
    // TODO 单元测试用，这里测试可行，先留着，回头删了
    int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, std::vector<KeyLine>& new_kls, std::vector<std::pair<int, int>>& match_indices);


    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    // 这个用来重定位的，和上一个关键帧匹配
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapLine*> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    // 这个用来回环检测的
    int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapLine*> &vpLines, std::vector<MapLine*> &vpMatched, int th);

    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                               vector<pair<size_t, size_t>> &vMatchedPairs,
                               const bool bOnlyStereo);

    // Project MapLines into KeyFrame and search for duplicated MapLines
    int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines);




public:
    float mfNNratio;            ///< 最优评分和次优评分的比例
    bool mbCheckOrientation;    ///< 是否检查特征点的方向

    // 匹配阈值
    double angle_threshold_ = 15.0 * M_PI / 180.0;
    double length_threshold_ = 0.45;
    double overlap_threshold_ = 0.5;
    double desc_dist_threshold_ = 45;
    double reproj_error_threshold_ = 45;

    /**
     * @brief Liang-Barsky线段裁剪算法，获取一条线段被一个边界所截取的部分
     * 这里用来处理空间线段的部分观测问题，空间线段投影到图像上之后，将图像内的部分裁剪出来
     * @param[in] line        空间线段起点终点投影到图像上的坐标
     * @param[in&out] new_line    新的被裁剪的坐标
     * @param[in] bounds      图像边界
     * @return  这条空间线段投影是否在图像内，true是，false否
     */
    bool LiangBarsky(Eigen::Vector4d& line, Eigen::Vector4d& new_line, std::vector<float>& bounds);

    // 输入两个KeyLine，输出是否匹配
    // offset是阈值的修改值，四维
    bool LineMatching(KeyLine& kl1, KeyLine& kl2, const cv::Mat& desc1, const cv::Mat& desc2,
                      const std::vector<double>& offset = std::vector<double>(5, 0));

    // 计算两个KeyLine的重叠比例
    bool LineOverLap(KeyLine& kl1, KeyLine& kl2, double& threshold);

    // line2到line1的垂直距离，通常匹配的线段距离不会差太远，但也不一定很近，所以这里阈值设大一点
    double PerpendicularDistance(KeyLine& line1, KeyLine& line2);

    double ReprojectionError(KeyLine& line1, KeyLine& line2);

    // new_line中包含的是新的KL的起点和终点坐标
    // im就是原图
    void UpdateKeyLineData(Eigen::Vector4d& new_line, KeyLine& old_line, cv::Mat& im);


};

// todo 按描述子之间的距离从小到大排序？ 这里存疑，可能是按描述子左匹配的序号进行排序，这个地方需要检查，找用到这个函数的地方
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

} // namespace ORB_SLAM2

#endif //ORB_SLAM2_LINEMATCHER_H
