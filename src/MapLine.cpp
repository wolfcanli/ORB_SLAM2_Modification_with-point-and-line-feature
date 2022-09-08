//
// Created by jiajieshi on 22-9-5.
//

#include "MapLine.h"

namespace ORB_SLAM2{

long unsigned int MapLine::nNextId=0;
mutex MapLine::mGlobalMutex;

MapLine::MapLine(const Vector6d &Pos, KeyFrame *pRefKF, Map* pMap):
        mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
        mpReplaced(static_cast<MapLine*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    mWorldPos = Pos;
    mStart3d = Pos.head(3);
    mEnd3d = Pos.tail(3);
    mNormalVector << 0.0, 0.0, 0.0;

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapLine::MapLine(const Vector6d &Pos, Map* pMap, Frame* pFrame, const int &idxF):
        mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
        mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
        mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    mWorldPos = Pos;
    mStart3d = Pos.head(3);
    mEnd3d = Pos.tail(3);

    cv::Mat Ow = pFrame->GetCameraCenter();
    Eigen::Vector3d o_w(Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2));
    Eigen::Vector3d mid_point = 0.5 * (mStart3d + mEnd3d);

    mNormalVector = mid_point - o_w; // 相机到特征线中点的向量
    mNormalVector.normalize();

    Eigen::Vector3d PC = mid_point - o_w;
    const float dist = PC.norm();

    const int level_line = pFrame->mvKeyLinesUn[idxF].octave;
    const float levelScaleFactor_line = pFrame->mvScaleFactors[level_line];
    const int nLevels_line = pFrame->mnScaleLevels;

    // 计算这两个干啥用的暂时还不清楚
    mfMaxDistance = dist * levelScaleFactor_line;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels_line - 1];

    // 将帧上的特征线的描述子给到地图线
    pFrame->mLineDescriptors.row(idxF).copyTo(mLineDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapLine::SetWorldPos(const Vector6d &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Vector6d MapLine::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3d MapLine::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}

KeyFrame* MapLine::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapLine::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF]=idx;

    // 如果直线的两个端点都有深度
    if(pKF->mvDepthLineStart[idx]>=0 && pKF->mvDepthLineEnd[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

void MapLine::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 如果有被pKF观测到，则删除观测
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF]; // 获取索引
            if(pKF->mvuRightLineStart[idx]>=0 && pKF->mvDepthLineStart[idx] >= 0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapLine::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapLine::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapLine::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapLineMatch(mit->second);
    }

    mpMap->EraseMapLine(this);
}

MapLine* MapLine::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapLine::Replace(MapLine* pML)
{
    if(pML->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pML;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pML->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapLineMatch(mit->second, pML);
            pML->AddObservation(pKF, mit->second);
        }
        else
        {
            pKF->EraseMapLineMatch(mit->second);
        }
    }
    pML->IncreaseFound(nfound);
    pML->IncreaseVisible(nvisible);
    pML->ComputeDistinctiveDescriptors();

    mpMap->EraseMapLine(this);
}

bool MapLine::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapLine::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapLine::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapLine::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

// 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
void MapLine::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    // Step 1 获取该地图点所有有效的观测关键帧信息
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // Step 2 遍历观测到该地图点的所有关键帧，对应的LBD描述子，放到向量vDescriptors中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        // mit->first取观测到该地图点的关键帧
        // mit->second取该地图点在关键帧中的索引
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            // 取对应的描述子向量
            vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // Step 3 计算这些描述子两两之间的距离
    // N表示为一共多少个描述子
    const size_t N = vDescriptors.size();
    // 将Distances表述成一个对称的矩阵
    // float Distances[N][N];
    std::vector<std::vector<float>> Distances(N, std::vector<float>(N, 0));
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i] = 0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = LineMatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    // Step 4 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值
    int BestMedian = INT_MAX; // 记录最小的中值
    int BestIdx = 0; // 最小中值对应的索引
    for(size_t i=0;i<N;i++) {
        // 第i个描述子到其它所有描述子之间的距离
        vector<int> vDists(Distances[i].begin(), Distances[i].end());
        // 排序，从小到大
        sort(vDists.begin(),vDists.end());
        // 获得中值
        int median = vDists[0.5 * (N - 1)];
        // 寻找最小的中值
        if(median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mLineDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapLine::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mLineDescriptor.clone();
}

int MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapLine::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapLine::UpdateNormalAndDepth()
{
    // Step 1 获得观测到该地图点的所有关键帧、坐标等信息
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    Vector6d Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations=mObservations; // 获得观测到该地图点的所有关键帧
        pRefKF=mpRefKF; // 观测到该点的参考关键帧（第一次创建时的关键帧）
        Pos = mWorldPos; // 地图点在世界坐标系中的位置
    }

    if(observations.empty())
        return;

    // Step 2 计算该地图点的平均观测方向
    // 能观测到该地图点的所有关键帧，对该点的观测方向归一化为单位向量，然后进行求和得到该地图点的朝向
    // 初始值为0向量，累加为归一化向量，最后除以总数n
    Eigen::Vector3d normal(0.0, 0.0, 0.0);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        Eigen::Vector3d o_w(Owi.at<double>(0), Owi.at<double>(1), Owi.at<double>(2));
        Eigen::Vector3d mid_point = 0.5 * (Pos.head(3) + Pos.tail(3));

        // 获得地图点和观测到它关键帧的向量并归一化
        Eigen::Vector3d normali = mid_point - o_w;
        normal = normal + normali/normali.norm();
        n++;
    }

    Eigen::Vector3d mid_point = 0.5 * (Pos.head(3) + Pos.tail(3));
    cv::Mat pRefKF_CC = pRefKF->GetCameraCenter();
    Eigen::Vector3d pRefKF_cc(pRefKF_CC.at<double>(0), pRefKF_CC.at<double>(1), pRefKF_CC.at<double>(2));

    Eigen::Vector3d PC = mid_point - pRefKF_cc; // 参考关键帧相机指向地图点的向量（在世界坐标系下的表示）
    const float dist = PC.norm(); // 该点到参考关键帧相机的距离

    const int level = pRefKF->mvKeyLinesUn[observations[pRefKF]].octave; // 观测到该地图点的当前帧的特征点在金字塔的第几层
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level]; // 当前金字塔层对应的尺度因子，scale^n，scale=1.2，n为层数
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔总层数，默认为8

    {
        unique_lock<mutex> lock3(mMutexPos);
        // 使用方法见PredictScale函数前的注释
        mfMaxDistance = dist*levelScaleFactor; // 观测到该点的距离上限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 观测到该点的距离下限
        mNormalVector = normal/n; // 获得地图点平均的观测方向
    }
}


void MapLine::UpdateAverageDir() {
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    Vector6d Pos; //地图线的坐标
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations=mObservations; // 获得观测到该3d点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧?
        Pos = mWorldPos;
    }

    if(observations.empty())
        return;

    Eigen::Vector3d normal(0, 0, 0);
    int n = 0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++) {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        Eigen::Vector3d Ow(Owi.at<float>(0), Owi.at<float>(1), Owi.at<float>(2));
        Eigen::Vector3d middlePos = 0.5*(mWorldPos.head(3)+mWorldPos.tail(3));
        Eigen::Vector3d normali = middlePos - Ow;
        assert(normali.norm() != 0);
        normal = normal + normali/normali.norm();
        n++;
    }

    cv::Mat SP = (cv::Mat_<float>(3,1) << Pos(0), Pos(1), Pos(2));
    cv::Mat EP = (cv::Mat_<float>(3,1) << Pos(3), Pos(4), Pos(5));
    cv::Mat MP = 0.5*(SP+EP);

    cv::Mat CM = MP - pRefKF->GetCameraCenter();  // 参考关键帧相机指向3Dline的向量（在世界坐标系下的表示）
    const float dist = cv::norm(CM);

    const int level = pRefKF->mvKeyLines[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;                           // 观测到该点的距离下限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 观测到该点的距离上限
        assert(n!=0);
        mNormalVector = normal/n;                                        // 获得平均的观测方向
    }
}



float MapLine::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapLine::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

int MapLine::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapLine::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

} //namespace ORB_SLAM2

