/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "Converter.h"
#include "ORBmatcher.h"

#include "MapLine.h"

#include <mutex>
#include <opencv2/line_descriptor.hpp>

using namespace cv::line_descriptor;

namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;
class MapLine;

/**
 * @brief 关键帧类
 * @detials 关键帧，和普通的Frame不一样，但是可以由Frame来构造; 许多数据会被三个线程同时访问，所以用锁的地方很普遍
 *
 */
class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    // 这里的set,get需要用到锁

    /**
     * @brief 设置当前关键帧的位姿
     * @param[in] Tcw 位姿
     */
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();                  ///< 获取位姿
    cv::Mat GetPoseInverse();           ///< 获取位姿的逆
    cv::Mat GetCameraCenter();          ///< 获取(左目)相机的中心
    cv::Mat GetStereoCenter();          ///< 获取双目相机的中心,这个只有在可视化的时候才会用到
    cv::Mat GetRotation();              ///< 获取姿态
    cv::Mat GetTranslation();           ///< 获取位置

    /**
      * @brief Bag of Words Representation
      * @detials 计算mBowVec，并且将描述子分散在第4层上，即mFeatVec记录了属于第i个node的ni个描述子
      * @see ProcessNewKeyFrame()
      */
    void ComputeBoW();

    // ====================== Covisibility graph functions ============================

    /**
     * @brief 为关键帧之间添加连接
     * @details 更新了mConnectedKeyFrameWeights
     * @param pKF    关键帧
     * @param weight 权重，该关键帧与pKF共同观测到的3d点数量
     */
    void AddConnection(KeyFrame* pKF, const int &weight);
    /**
     * @brief 删除当前关键帧和指定关键帧之间的共视关系
     * @param[in] pKF 要删除的共视关系
     */
    void EraseConnection(KeyFrame* pKF);
    /** @brief 更新图的连接  */
    void UpdateConnections();
    /**
     * @brief 按照权重对连接的关键帧进行排序
     * @detials 更新后的变量存储在mvpOrderedConnectedKeyFrames和mvOrderedWeights中
     */
    void UpdateBestCovisibles();
    /**
     * @brief 得到与该关键帧连接的关键帧(没有排序的)
     * @return 连接的关键帧
     */
    std::set<KeyFrame *> GetConnectedKeyFrames();
    /**
     * @brief 得到与该关键帧连接的关键帧(已按权值排序)
     * @return 连接的关键帧
     */
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    /**
     * @brief 得到与该关键帧连接的前N个关键帧(已按权值排序)
     * NOTICE 如果连接的关键帧少于N，则返回所有连接的关键帧,所以说返回的关键帧的数目其实不一定是N个
     * @param N 前N个
     * @return 连接的关键帧
     */
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    /**
     * @brief 得到与该关键帧连接的权重大于等于w的关键帧
     * @param w 权重
     * @return 连接的关键帧
     */
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
    /**
     * @brief 得到该关键帧与pKF的权重
     * @param  pKF 关键帧
     * @return     权重
     */
    int GetWeight(KeyFrame* pKF);

    // ========================= Spanning tree functions =======================
    /**
     * @brief 添加子关键帧（即和子关键帧具有最大共视关系的关键帧就是当前关键帧）
     * @param[in] pKF 子关键帧句柄
     */
    void AddChild(KeyFrame* pKF);
    /**
     * @brief 删除某个子关键帧
     * @param[in] pKF 子关键帧句柄
     */
    void EraseChild(KeyFrame* pKF);
    /**
     * @brief 改变当前关键帧的父关键帧
     * @param[in] pKF 父关键帧句柄
     */
    void ChangeParent(KeyFrame* pKF);
    /**
     * @brief 获取获取当前关键帧的子关键帧
     * @return std::set<KeyFrame*>  子关键帧集合
     */
    std::set<KeyFrame*> GetChilds();
    /**
     * @brief 获取当前关键帧的父关键帧
     * @return KeyFrame* 父关键帧句柄
     */
    KeyFrame* GetParent();
    /**
     * @brief 判断某个关键帧是否是当前关键帧的子关键帧
     * @param[in] pKF 关键帧句柄
     * @return true
     * @return false
     */
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    /**
     * @brief 给当前关键帧添加回环边，回环边连接了形成闭环关系的关键帧
     * @param[in] pKF  和当前关键帧形成闭环关系的关键帧
     */
    void AddLoopEdge(KeyFrame* pKF);
    /**
     * @brief 获取和当前关键帧形成闭环关系的关键帧
     * @return std::set<KeyFrame*> 结果
     */
    std::set<KeyFrame*> GetLoopEdges();

// ====================== MapPoint observation functions ==================================
    /**
     * @brief Add MapPoint to KeyFrame
     * @param pMP MapPoint
     * @param idx MapPoint在KeyFrame中的索引
     */
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void AddMapLine(MapLine* pML, const size_t &idx);
    /**
     * @brief 由于其他的原因,导致当前关键帧观测到的某个地图点被删除(bad==true)了,这里是"通知"当前关键帧这个地图点已经被删除了
     * @param[in] idx 被删除的地图点索引
     */
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapLineMatch(const size_t &idx);
    /**
     * @brief 由于其他的原因,导致当前关键帧观测到的某个地图点被删除(bad==true)了,这里是"通知"当前关键帧这个地图点已经被删除了
     * @param[in] pMP 被删除的地图点指针
     */
    void EraseMapPointMatch(MapPoint* pMP);
    void EraseMapLineMatch(MapLine* pML);
    /**
     * @brief 地图点的替换
     * @param[in] idx 要替换掉的地图点的索引
     * @param[in] pMP 新地图点的指针
     */
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    void ReplaceMapLineMatch(const size_t &idx, MapLine* pML);
    /**
     * @brief 获取当前帧中的所有地图点
     * @return std::set<MapPoint*> 所有的地图点
     */
    std::set<MapPoint*> GetMapPoints();
    std::set<MapLine*> GetMapLines();
    /**
     * @brief Get MapPoint Matches 获取该关键帧的MapPoints
     */
    std::vector<MapPoint*> GetMapPointMatches();
    std::vector<MapLine*> GetMapLineMatches();
    /**
     * @brief 关键帧中，大于等于minObs的MapPoints的数量
     * @details minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint \n
     * 一个高质量的MapPoint会被多个KeyFrame观测到.
     * @param  minObs 最小观测
     */
    int TrackedMapPoints(const int &minObs);
    int TrackedMapLines(const int &minObs);
    /**
     * @brief 获取获取当前关键帧的具体的某个地图点
     * @param[in] idx id
     * @return MapPoint* 地图点句柄
     */
    MapPoint* GetMapPoint(const size_t &idx);
    MapLine* GetMapLine(const size_t &idx);

    // KeyPoint functions
    /**
     * @brief 获取某个特征点的邻域中的特征点id
     * @param[in] x 特征点坐标
     * @param[in] y 特征点坐标
     * @param[in] r 邻域大小(半径)
     * @return std::vector<size_t> 在这个邻域内找到的特征点索引的集合
     */
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    std::vector<size_t> GetLineFeaturesInArea(const float &x, const float &y, const float &r) const;
    /**
     * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
     * @param  i 第i个keypoint
     * @return   3D点（相对于世界坐标系）
     */
    cv::Mat UnprojectStereo(int i);
    cv::Mat UnprojectStereoLineStart(int i);
    cv::Mat UnprojectStereoLineEnd(int i);

    // Image
    /**
     * @brief 判断某个点是否在当前关键帧的图像中
     * @param[in] x 点的坐标
     * @param[in] y 点的坐标
     * @return true
     * @return false
     */
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    /** @brief 设置当前关键帧不要在优化的过程中被删除  */
    void SetNotErase();
    /** @brief 准备删除当前的这个关键帧,表示不进行回环检测过程;由回环检测线程调用 */
    void SetErase();

    // Set/check bad flag
    /** @brief 真正地执行删除关键帧的操作 */
    void SetBadFlag();
    /** @brief 返回当前关键帧是否已经完蛋了 */
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    /**
     * @brief 评估当前关键帧场景深度，q=2表示中值
     * @param q q=2
     * @return Median Depth
     */
    float ComputeSceneMedianDepth(const int q);

    /// 比较两个int型权重的大小的比较函数
    static bool weightComp( int a, int b)
    {
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2)
    {
        return pKF1->mnId<pKF2->mnId;
    }

    void lineDescriptorMAD(vector<vector<cv::DMatch>> line_matches, double &nn_mad, double &nn12_mad) const;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;
    // 特征线数量
    const int NL;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    // 特征线相关属性
    const std::vector<KeyLine> mvKeyLines;
    const std::vector<KeyLine> mvKeyLinesUn;
    const std::vector<float> mvuRightLineStart;
    const std::vector<float> mvuRightLineEnd;
    const std::vector<float> mvDepthLineStart;
    const std::vector<float> mvDepthLineEnd;
    const cv::Mat mLineDescriptors;
    std::vector<Eigen::Vector3d> mvKeyLineCoefficient; // 特征线直线系数

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
public:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;
    // MapPoints associated to keylines
    std::vector<MapLine*> mvpMapLines;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

// 第一种方式：按照每个特征对应的最小匹配距离进行排序
struct compare_descriptor_by_NN_dist
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b) {
        return ( a[0].distance < b[0].distance );  //从小到大排列
    }
};

// 第二种方式： 按照每个特征对应的最小和第二小之间的差值进行排序，差值大的排在前面
struct compare_descriptor_by_NN12_dist
{
    inline bool operator()(const vector<cv::DMatch>& a, const vector<cv::DMatch>& b) {
        return ((a[1].distance - a[0].distance) < (b[1].distance - b[0].distance) );
    }
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
