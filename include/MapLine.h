//
// Created by jiajieshi on 22-9-5.
//

#ifndef ORB_SLAM2_MAPLINE_H
#define ORB_SLAM2_MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "LineMatcher.h"

#include<opencv2/core/core.hpp>
#include <Eigen/Core>
#include<mutex>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

namespace ORB_SLAM2{

class KeyFrame;
class Map;
class Frame;

class MapLine {
public:
    /**
     * @brief 给定坐标与keyframe构造MapPoint
     * @details 被调用: 双目：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints() \n
     * 单目：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
     * @param[in] Pos       MapPoint的坐标（wrt世界坐标系）
     * @param[in] pRefKF    KeyFrame
     * @param[in] pMap      Map
     */
    MapLine(const Vector6d &Pos, KeyFrame* pRefKF, Map* pMap);
    /**
     * @brief 给定坐标与frame构造MapPoint
     * @detials 被双目：UpdateLastFrame()调用
     * @param[in] Pos       MapPoint的坐标（世界坐标系）
     * @param[in] pMap      Map
     * @param[in] pFrame    Frame
     * @param[in] idxF      MapPoint在Frame中的索引，即对应的特征点的编号
     */
    MapLine(const Vector6d &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    // 设置和获取世界坐标系下的坐标
    void SetWorldPos(const Vector6d &Pos);
    Vector6d GetWorldPos();

    //世界坐标系下地图点被多个相机观测的平均观测方向
    Eigen::Vector3d GetNormal();
    // 获取生成当前地图点的参考关键帧
    KeyFrame* GetReferenceKeyFrame();
    // 获取观测到该地图点的关键帧序列，size_t表示该地图点在这些关键帧中的索引
    std::map<KeyFrame*,size_t> GetObservations();
    // 获取当前地图点被观测的次数
    int Observations();
    /**
     * @brief 添加观测
     *
     * 记录哪些KeyFrame的那个特征点能观测到该MapPoint \n
     * 并增加观测的相机数目nObs，单目+1，双目或者grbd+2
     * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些MapPoints的关键帧是共视关键帧
     * @param[in] pKF KeyFrame,观测到当前地图点的关键帧
     * @param[in] idx MapPoint在KeyFrame中的索引
     */
    void AddObservation(KeyFrame* pKF,size_t idx);
    /**
     * @brief 取消某个关键帧对当前地图点的观测
     * @detials 如果某个关键帧要被删除，那么会发生这个操作
     * @param[in] pKF
     */
    void EraseObservation(KeyFrame* pKF);
    // 获取观测到当前地图点的关键帧在其中的索引
    int GetIndexInKeyFrame(KeyFrame* pKF);
    // 判断该点是否能被某个关键帧看到
    bool IsInKeyFrame(KeyFrame* pKF);

    // 告知所有可以观测到该点的关键帧，该点已经被删除
    void SetBadFlag();
    bool isBad();
    // 在形成闭环时，会更新关键帧与地图点的关系,pMP是要代替this的地图点
    void Replace(MapLine* pML);
    // 获取取代当前地图点的点
    MapLine* GetReplaced();
    /**
     * @brief 增加可视次数
     * @detials Visible表示：
     * \n 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
     * \n 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
     * \n   例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
     *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
     * @param[in] n 要增加的次数
     */
    void IncreaseVisible(int n=1);
    /**
     * @brief Increase Found
     * @detials 能找到该点的帧数+n，n默认为1
     * @param[in] n 增加的个数
     * @see Tracking::TrackLocalMap()
     */
    void IncreaseFound(int n=1);
    // 获取被找到的帧数和被观测到的帧数之间的比值
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    /**
     * @brief 计算具有代表的描述子
     * @detials 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
     * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
     * @see III - C3.3
     */
    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();
    /**
     * @brief 更新平均观测方向以及观测距离范围
     *
     * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量
     * @see III - C2.2 c2.4
     */
    void UpdateNormalAndDepth();
    void UpdateAverageDir();  //这里只更新线段的平均观测方向，ORB中还要更新深度

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
    //呐,如果是从帧中创建的话,会将普通帧的id存放于这里
    const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）

    // 被观测到的相机数目，单目+1，双目或RGB-D则+2
    int nObs;

    // Variables used by the tracking
    float mTrackProjStartX;             ///< 当前地图点投影到某帧上后的坐标
    float mTrackProjStartY;             ///< 当前地图点投影到某帧上后的坐标
    float mTrackProjStartXR;            ///< 当前地图点投影到某帧上后的坐标(右目)

    float mTrackProjEndX;             ///< 当前地图点投影到某帧上后的坐标
    float mTrackProjEndY;             ///< 当前地图点投影到某帧上后的坐标
    float mTrackProjEndXR;            ///< 当前地图点投影到某帧上后的坐标(右目)

    int mnTrackScaleLevel;         ///< 所处的尺度, 由其他的类进行操作 //?
    float mTrackViewCos;           ///< 被追踪到时,那帧相机看到当前地图点的视角
    // TrackLocalMap - SearchByProjection 中决定是否对该点进行投影的变量
    // NOTICE mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影   //? 为什么已经是内点了之后就不需要再进行投影了呢?
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）     //?
    bool mbTrackInView;
    // TrackLocalMap - UpdateLocalPoints 中防止将MapPoints重复添加至mvpLocalMapPoints的标记
    long unsigned int mnTrackReferenceForFrame;

    // TrackLocalMap - SearchLocalPoints 中决定是否进行isInFrustum判断的变量
    // NOTICE mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;

    //REVIEW 下面的....都没看明白
    // Variables used by local mapping
    // local mapping中记录地图点对应当前局部BA的关键帧的mnId。mnBALocalForKF 在map point.h里面也有同名的变量。
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;     ///< 在局部建图线程中使用,表示被用来进行地图点融合的关键帧(存储的是这个关键帧的id)

    // Variables used by loop closing -- 一般都是为了避免重复操作
    /// 标记当前地图点是作为哪个"当前关键帧"的回环地图点(即回环关键帧上的地图点),在回环检测线程中被调用
    long unsigned int mnLoopPointForKF;
    // 如果这个地图点对应的关键帧参与到了回环检测的过程中,那么在回环检测过程中已经使用了这个关键帧修正只有的位姿来修正了这个地图点,那么这个标志位置位
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    // 全局BA优化后(如果当前地图点参加了的话),这里记录优化后的位姿
    cv::Mat mPosGBA;
    // 如果当前点的位姿参与到了全局BA优化,那么这个变量记录了那个引起全局BA的"当前关键帧"的id
    long unsigned int mnBAGlobalForKF;

    ///全局BA中对当前点进行操作的时候使用的互斥量
    static std::mutex mGlobalMutex;

public:
    // 下面空间直线的表示需要修改，用的不是普吕克坐标
    // Position in absolute coordinates
    // 由两端点坐标构成的空间直线的表示
    Vector6d mWorldPos;
    Eigen::Vector3d mStart3d;
    Eigen::Vector3d mEnd3d;

    // 观测到该MapLine的关键帧和该MapLine在该关键帧中的索引
    std::map<KeyFrame*,size_t> mObservations;

    // Mean viewing direction
    // 该线段的平均观测方向，相机到地图线中点的向量
    Eigen::Vector3d mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mLineDescriptor;

    // Reference KeyFrame
    KeyFrame* mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapLine* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM


#endif //ORB_SLAM2_MAPLINE_H
