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
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);
    // 加一个用来和参考关键帧匹配的
    // vpMapLineMatches CurrentFrame中地图点对应的匹配，NULL表示未匹配
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *RefFrame, std::vector<MapLine*> &vpMapLineMatches);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    // 这个用来和局部地图匹配
    int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);

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
