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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth),
     N(frame.N), mvKeys(frame.mvKeys), mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mvKeyLines(frame.mvKeyLines), mvKeyLinesUn(frame.mvKeyLinesUn),
     mvuRightLineStart(frame.mvuRightLineStart), mvuRightLineEnd(frame.mvuRightLineEnd),
     mvDepthLineStart(frame.mvDepthLineStart), mvDepthLineEnd(frame.mvDepthLineEnd),
     mLineDescriptors(frame.mLineDescriptors.clone()), mvKeyLineCoefficient(frame.mvKeyLineCoefficient),
     mvpMapLines(frame.mvpMapLines), mvbLineOutlier(frame.mvbLineOutlier) {
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    // 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }
    // 计算假想的基线长度 baseline= mbf/fx
    // 后面要对从RGBD相机输入的特征点,结合相机基线长度,焦距,以及点的深度等信息来计算其在假想的"右侧图像"上的匹配点
    mb = mbf/fx;

    // 将特征点分配到图像网格中
    AssignFeaturesToGrid();
}

// 深度图像构造
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // 同时提取点线特征，双线程
    std::thread extract_point_thread_(&Frame::ExtractORB, this, 0, imGray);
    std::thread extract_line_thread(&Frame::ExtractLine, this, imGray);
    extract_point_thread_.join();
    extract_line_thread.join();

    // ORB extraction
//    ExtractORB(0,imGray);

    N = mvKeys.size();
    NL = mvKeyLines.size();

    if(mvKeys.empty() && mvKeyLines.empty())
        return;

    UndistortKeyPoints();
    UndistortKeyLines();

    // 同时计算线特征的起点和终点对应的右目坐标
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // 线特征对应的地图线
    mvpMapLines = vector<MapLine*>(NL, static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// 将提取的ORB特征点分配到图像网格中
void Frame::AssignFeaturesToGrid()
{
    // Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
    // ? 这里0.5 是为什么？节省空间？
    // FRAME_GRID_COLS = 64，FRAME_GRID_ROWS=48
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    //开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);
    // Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中
    for(int i=0;i<N;i++)
    {
        //从类的成员变量中获取已经去畸变后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];
        //存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            //如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::ExtractLine(const cv::Mat &im) {
    mpLineExtractor->ExtractLineSegment(im, mvKeyLines, mLineDescriptors, mvKeyLineCoefficient);
}


void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

bool Frame::isInFrustum(ORB_SLAM2::MapLine *pML, float viewingCosLimit) {
    pML->mbTrackInView = false;

    // 3D in absolute coordinates
    Vector6d P = pML->GetWorldPos();

    cv::Mat point_start = (cv::Mat_<float>(3, 1) << P(0), P(1), P(2));
    cv::Mat point_end = (cv::Mat_<float>(3, 1) << P(3), P(4), P(5));

    const cv::Mat P_start = mRcw * point_start + mtcw;
    const float &P_start_x = P_start.at<float>(0);
    const float &P_start_y = P_start.at<float>(1);
    const float &P_start_z = P_start.at<float>(2);

    const cv::Mat P_end = mRcw * point_end + mtcw;
    const float &P_end_x = P_end.at<float>(0);
    const float &P_end_y = P_end.at<float>(1);
    const float &P_end_z = P_end.at<float>(2);

    // Check positive depth
    if(P_start_z < 0.0f || P_end_z < 0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz_start = 1.0f / P_start_z;
    const float u_start = fx * P_start_x * invz_start + cx;
    const float v_start = fy * P_start_y * invz_start + cy;

    const float invz_end = 1.0f / P_end_z;
    const float u_end = fx * P_end_x * invz_end + cx;
    const float v_end = fy * P_end_y * invz_end + cy;

    if(u_start < mnMinX || u_start > mnMaxX)
        return false;
    if(v_start < mnMinY || v_start > mnMaxY)
        return false;
    if(u_end < mnMinX || u_end > mnMaxX)
        return false;
    if(v_end < mnMinY || v_end > mnMaxY)
        return false;

    // 计算MapLine到相机中心的距离，并判断是否在尺度变化的范围内
    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();

    const cv::Mat PO = 0.5 * (point_start + point_end) - mOw;
    const float dist = cv::norm(PO);

    if(dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (cv::Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

    const float viewCos = PO.dot(pn) / dist;

    if(viewCos < viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pML->PredictScale(dist,this);

    // Data used by the tracking
    pML->mbTrackInView = true;

    pML->mTrackProjStartX = u_start;
    pML->mTrackProjStartY = v_start;
    pML->mTrackProjStartXR = u_start - mbf * invz_start;

    pML->mTrackProjEndX = u_end;
    pML->mTrackProjEndY = v_end;
    pML->mTrackProjEndXR = u_end - mbf * invz_end;

    pML->mnTrackScaleLevel= nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// 传入参数和上面差不多，变为了两个端点的坐标
// 但是如果两条直线不是严格端点匹配，对比中点显得不合理；其次金字塔层数，他美的不是提取了一层，还比较什么
// TODO 这个策略可以改，当然暴力匹配也是可以的
vector<size_t> Frame::GetLinesInArea(const float &x_s, const float &y_s,
                                     const float &x_e, const float &y_e,
                                     const float &r,
                                     const int minLevel, const int maxLevel) const {
    std::vector<size_t> vLineIndices;
    std::vector<KeyLine> vkl = this->mvKeyLinesUn;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

    for (size_t i = 0; i < vkl.size(); i++) {
        KeyLine keyline = vkl[i];
        // 找到中点距离比较近的
        double distance = (0.5 * (x_s + x_e) - keyline.pt.x) * (0.5 * (x_s + x_e) - keyline.pt.x) +
                (0.5 * (y_s + y_e) - keyline.pt.y) * (0.5 * (y_s + y_e) - keyline.pt.y);
        if (distance > r * r)
            continue;

        // 找到斜率差不多的
        float slope = (y_e - y_s) / (x_e - x_s) - keyline.angle;
        if (slope > r + 0.01)
            continue;

        // 比较金字塔层数
        if (bCheckLevels) {
            if (keyline.octave < minLevel)
                continue;
            ///可见opencv中线的提取也是有尺度的！注意凡是涉及到尺度层的一些操作要保证和ORB中类似，切勿出错
            // cout << "线特征的octave： " << keyline.octave << endl;
            if (maxLevel >= 0 && keyline.octave > maxLevel)
                continue;
        }

        vLineIndices.push_back(i);
    }
    return vLineIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 *
 */
void Frame::ComputeBoW()
{

    // 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
        // 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        mpORBvocabulary->transform(vCurrentDesc,	//当前的描述子vector
                                   mBowVec,			//输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                                   mFeatVec,		//输出，记录node id及其对应的图像 feature对应的索引
                                   4);				//4表示从叶节点向前数的层数
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::UndistortKeyLines() {
    // 对于线特征，矫正起点和终点两个点
    // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
    //变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0) {
        mvKeyLinesUn=mvKeyLines;
        return;
    }

    // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    // N为提取的特征点数量，为满足OpenCV函数输入要求，将NL个特征线的起终点分别保存在NL*2的矩阵中
    // 包括起点和终点
    cv::Mat mat_start(NL,2,CV_32F);
    cv::Mat mat_end(NL,2,CV_32F);
    //将这些点的横纵坐标分别保存
    for(int i=0; i<NL; i++)
    {
        // 起点
        mat_start.at<float>(i,0)=mvKeyLines[i].startPointX;
        mat_start.at<float>(i,1)=mvKeyLines[i].startPointY;
        // 终点
        mat_end.at<float>(i,0)=mvKeyLines[i].endPointX;
        mat_end.at<float>(i,1)=mvKeyLines[i].endPointY;
    }

    // Undistort points
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    //为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    mat_start = mat_start.reshape(2);
    cv::undistortPoints(mat_start, mat_start, mK, mDistCoef, cv::Mat(), mK);
    mat_start = mat_start.reshape(1);

    mat_end = mat_end.reshape(2);
    cv::undistortPoints(mat_end, mat_end, mK, mDistCoef, cv::Mat(), mK);
    mat_end = mat_end.reshape(1);

    // Fill undistorted keypoint vector
    // Step 存储校正后的特征点
    mvKeyLinesUn.resize(NL);
    for(int i=0; i<NL; i++)
    {
        //根据索引获取这个特征线
        //注意之所以这样做而不是直接重新声明一个特征线对象的目的是，能够得到源特征线对象的其他属性
        KeyLine kl = mvKeyLines[i];
        kl.startPointX = mat_start.at<float>(i, 0);
        kl.startPointY = mat_start.at<float>(i, 1);
        kl.endPointX = mat_end.at<float>(i, 0);
        kl.endPointY = mat_end.at<float>(i, 1);

        mvKeyLinesUn[i] = kl;
    }
}

// 计算去畸变图像的便捷
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; // 左上
        mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; // 右上
        mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; // 左下
        mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; // 右下
        mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        // 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        //校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        // 如果畸变参数为0，就直接获得图像边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvuRightLineStart = std::vector<float>(NL, -1);
    mvuRightLineEnd = std::vector<float>(NL, -1);
    mvDepthLineStart = std::vector<float>(NL, -1);
    mvDepthLineEnd = std::vector<float>(NL, -1);


    for(int i=0; i<N; i++) {
        // 获取矫正前后的特征点
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        // 获取矫正前特征点的横纵坐标
        const float &v = kp.pt.y;
        const float &u = kp.pt.x;
        // 从深度图中获取这个点对应的深度
        const float d = imDepth.at<float>(v,u);
        // 如果深度存在，则计算除等效的右图坐标
        // x - mbf / d
        if(d>0) {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }

    // 线特征处理，和上面一样，只不过这里处理两个端点
    for (int i = 0; i < NL; i++) {
        const KeyLine &kl = mvKeyLines[i];
        const KeyLine &klU = mvKeyLinesUn[i];

        const float &v_start = kl.startPointY;
        const float &u_start = kl.startPointX;
        const float &v_end = kl.endPointY;
        const float &u_end = kl.endPointX;

        const float d_start = imDepth.at<float>(v_start, u_start);
        const float d_end = imDepth.at<float>(v_end, u_end);

        if (d_start > 0) {
            mvDepthLineStart[i] = d_start;
            mvuRightLineStart[i] = klU.startPointX - mbf / d_start;
        }
        if (d_end > 0) {
            mvDepthLineEnd[i] = d_end;
            mvuRightLineEnd[i] = klU.endPointX - mbf / d_end;
        }
    }

}
// void Frame::setFrameImage(cv::Mat im)
// {
//   
//   im.copyTo(mimage);
//   
// }
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

cv::Mat Frame::UnprojectStereoLine(const int &i) {
    const float z_start = mvDepthLineStart[i];
    const float z_end = mvDepthLineEnd[i];

    cv::Mat x3Dw_start = (cv::Mat_<float>(3,1) << 0,0,0);
    cv::Mat x3Dw_end = (cv::Mat_<float>(3,1) << 0,0,0);

    if(z_start > 0) {
        const float u = mvKeyLinesUn[i].startPointX;
        const float v = mvKeyLinesUn[i].startPointY;
        const float x = (u - cx) * z_start * invfx;
        const float y = (v - cy) * z_start * invfy;
        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << x, y, z_start);
        x3Dw_start = mRwc * x3Dc_start + mOw;
    }

    if(z_end > 0) {
        const float u = mvKeyLinesUn[i].endPointX;
        const float v = mvKeyLinesUn[i].endPointY;
        const float x = (u - cx) * z_end * invfx;
        const float y = (v - cy) * z_end * invfy;
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << x, y, z_end);
        x3Dw_end = mRwc * x3Dc_end + mOw;
    }

    cv::Mat line3Dw = (cv::Mat_<float>(6, 1) <<
                        x3Dw_start.at<float>(0), x3Dw_start.at<float>(1), x3Dw_start.at<float>(2),
                        x3Dw_end.at<float>(0), x3Dw_end.at<float>(1), x3Dw_end.at<float>(2));

    if (z_start > 0 && z_end > 0) {
        return line3Dw;
    } else {
        return cv::Mat();
    }
}

cv::Mat Frame::UnprojectStereoLineStart(const int &i) {
    const float z_start = mvDepthLineStart[i];

    if(z_start > 0) {
        const float u = mvKeyLinesUn[i].startPointX;
        const float v = mvKeyLinesUn[i].startPointY;
        const float x = (u - cx) * z_start * invfx;
        const float y = (v - cy) * z_start * invfy;
        cv::Mat x3Dc_start = (cv::Mat_<float>(3,1) << x, y, z_start);
        return mRwc * x3Dc_start + mOw;
    }
    else
        return cv::Mat();
}

cv::Mat Frame::UnprojectStereoLineEnd(const int &i) {
    const float z_end = mvDepthLineStart[i];

    if(z_end > 0) {
        const float u = mvKeyLinesUn[i].endPointX;
        const float v = mvKeyLinesUn[i].endPointY;
        const float x = (u - cx) * z_end * invfx;
        const float y = (v - cy) * z_end * invfy;
        cv::Mat x3Dc_end = (cv::Mat_<float>(3,1) << x, y, z_end);
        return mRwc * x3Dc_end + mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
