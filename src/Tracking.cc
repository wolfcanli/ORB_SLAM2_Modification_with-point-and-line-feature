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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include "PointCloudMapping.h"

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetPointCloudMapping(PointCloudMapping* pPointCloudMapping) {
    mpPointCloudMapping = pPointCloudMapping;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // 将RGB或RGBA转化成灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 将深度相机的disparity转化成depth
    // 原本的深度图的像素值和真实的深度值（距离）成一个比例，这个比例是depth map factor，这里是5000
    // 所以下面就是将depth map除以5000，还原真实的深度
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    // 这里两个变量是为了稠密建图用的
 	mImRGB=imRGB;
	mImDepth=imDepth;

    // 构造Frame，这里面同时会提取特征
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    // #重点# 追踪，包括恒速模型追踪，参考帧追踪和局部地图追踪
    Track();

    // 返回估计完成的位姿
    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
    if(mState==NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED) {
        // 初始化
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            // 非单目的初始化同时会生成初始地图
            StereoInitialization();
        else
            // 单目初始化
            MonocularInitialization();

        // 和可视化有关
        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    } else {
        // 系统已经初始化，开始追踪
        bool bOK; // 用来表示追踪是否成功的标志，如果不成功则换种方法
 
        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking) {
            // SLAM模式
            // 局部地图工作，默认运行这里，除非手动设置成追踪模式

            if(mState==OK) {
                // 局部地图可能已经改变了一些前一帧追踪的地图点
                CheckReplacedInLastFrame();

                // 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
                // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
                // mnLastRelocFrameId 上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2) {
                    // 用最近的关键帧来跟踪当前的普通帧
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();
                } else {
                    // 用最近的普通帧来跟踪当前的普通帧
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        //恒速模型跟踪失败，只能根据参考关键帧来跟踪
                        bOK = TrackReferenceKeyFrame();
                }
            } else {
                // 如果跟踪状态不成功,那么就只能重定位了
                // BOW搜索，EPnP求解位姿
                bOK = Relocalization();
            }
        } else {
            // Localization Mode: Local Mapping is deactivated
            // Step 2：只进行跟踪tracking，局部地图不工作
            if(mState==LOST) {
                // Step 2.1 如果跟丢了，只能重定位
                bOK = Relocalization();
            } else {
                // mbVO是mbOnlyTracking为true时的才有的一个变量
                // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常 (注意有点反直觉)
                // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                if(!mbVO) {
                    // Step 2.2 如果跟踪正常，使用恒速模型 或 参考关键帧跟踪
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty()) {
                        bOK = TrackWithMotionModel();
                        // ? 为了和前面模式统一，这个地方是不是应该加上
                        // if(!bOK)
                        //    bOK = TrackReferenceKeyFrame();
                    } else {
                        // 如果恒速模型不被满足,那么就只能够通过参考关键帧来定位
                        bOK = TrackReferenceKeyFrame();
                    }
                } else {
                    // In last frame we tracked mainly "visual odometry" points.
                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    // mbVO为true，表明此帧匹配了很少（小于10）的地图点，要跪的节奏，既做跟踪又做重定位
                    // nmatchesMap + line_nmatchesMap < 15;

                    //MM=Motion Model,通过运动模型进行跟踪的结果
                    bool bOKMM = false;
                    //通过重定位方法来跟踪的结果
                    bool bOKReloc = false;

                    // 运动模型中构造的地图点
                    vector<MapPoint*> vpMPsMM;
                    // 运动模型中构造的地图线
                    std::vector<MapLine*> vpMLsMM;
                    //在追踪运动模型后发现的外点
                    vector<bool> vbOutMM;
                    //在追踪运动模型后发现的特征线的外点
                    vector<bool> vbLineOutMM;
                    //运动模型得到的位姿
                    cv::Mat TcwMM;

                    // Step 2.3 当运动模型有效的时候,根据运动模型计算位姿
                    if(!mVelocity.empty()) {
                        bOKMM = TrackWithMotionModel();

                        // 将恒速模型跟踪结果暂存到这几个变量中，因为后面重定位会改变这些变量
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vpMLsMM = mCurrentFrame.mvpMapLines;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        vbLineOutMM = mCurrentFrame.mvbLineOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }

                    // Step 2.4 使用重定位的方法来得到当前帧的位姿
                    bOKReloc = Relocalization();

                    // Step 2.5 根据前面的恒速模型、重定位结果来更新状态
                    if(bOKMM && !bOKReloc) {
                        // 恒速模型成功、重定位失败，重新使用之前暂存的恒速模型结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvpMapLines = vpMLsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;
                        mCurrentFrame.mvbLineOutlier = vbLineOutMM;

                        //? 疑似bug！这段代码是不是重复增加了观测次数？后面 TrackLocalMap 函数中会有这些操作
                        // 如果当前帧匹配的3D点很少，增加当前可视地图点的被观测次数
                        if(mbVO) {
                            // 更新当前帧的地图点被观测次数
                            for(int i =0; i<mCurrentFrame.N; i++) {
                                //如果这个特征点形成了地图点,并且也不是外点的时候
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    //增加能观测到该地图点的帧数
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }

                            // 更新当前帧的地图线被观测次数
                            for(int i = 0; i < mCurrentFrame.NL; i++) {
                                //如果这个特征点形成了地图点,并且也不是外点的时候
                                if(mCurrentFrame.mvpMapLines[i] && !mCurrentFrame.mvbLineOutlier[i])
                                {
                                    //增加能观测到该地图点的帧数
                                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                                }
                            }
                        }
                    } else if(bOKReloc) {
                        // 只要重定位成功整个跟踪过程正常进行（重定位与跟踪，更相信重定位）
                        mbVO = false;
                    }
                    //有一个成功我们就认为执行成功了
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 将最新的关键帧作为当前帧的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // 在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        if(!mbOnlyTracking) {
            if(bOK)
                bOK = TrackLocalMap();
        } else {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // 重定位成功
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // 更新显示线程中的图像、特征点、地图点等信息
        mpFrameDrawer->Update(this);

        // 只有前面追踪成功后，才考虑是否插入新关键帧
        if(bOK) {
            // Update motion model
            // 跟踪成功，更新恒速运动模型
            if(!mLastFrame.mTcw.empty()) {
                // 更新恒速运动模型 TrackWithMotionModel 中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            } else
                //否则速度为空
                mVelocity = cv::Mat();

            //更新显示中的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // 清除观测不到的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // 清除观测不到的线
            for(int i = 0; i < mCurrentFrame.NL; i++) {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                    if(pML->Observations() < 1) {
                        mCurrentFrame.mvbLineOutlier[i] = false;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // 清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
            // 上一步只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
            mlpTemporalPoints.clear();

            // 同样清除临时的特征线
            for(list<MapLine*>::iterator lit = mlpTemporalLines.begin(), lend =  mlpTemporalLines.end(); lit!=lend; lit++) {
                MapLine* pML = *lit;
                delete pML;
            }
            mlpTemporalLines.clear();

            // Check if we need to insert a new keyframe
            // 检测并插入关键帧，对于双目或RGB-D会产生新的地图点
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
            // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉

            //  Step 9 删除那些在bundle adjustment中检测为outlier的地图点
            for(int i=0; i<mCurrentFrame.N;i++) {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
            for(int i = 0; i < mCurrentFrame.NL; i++) {
                if(mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST) {
            //如果地图中的关键帧信息过少的话,直接重新进行初始化了
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一帧的数据,当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // 记录位姿信息，用于最后保存所有的轨迹
    if(!mCurrentFrame.mTcw.empty())
    {
        // 计算相对姿态Tcr = Tcw * Twr, Twr = Trw^-1
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

/**
 * @brief 双目和rgbd的地图初始化
 *
 * 由于具有深度信息，直接生成MapPoints
 * 一定注意：初始化的时候，凡是具有深度信息统统将特征转化为了地图特征，而没有考虑远近
 * RGBD初始化加入线特征的操作
 */
void Tracking::StereoInitialization()
{
    // 初始化要求当前帧的特征点超过500
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        // 设定初始位姿为单位旋转，0平移
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // 将当前帧构造为初始关键帧
        // mCurrentFrame的数据类型为Frame
        // KeyFrame包含Frame、地图3D点、以及BoW
        // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
        // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
        // 提问: 为什么要指向Tracking中的相应的变量呢? -- 因为Tracking是主线程，是它创建和加载的这些模块
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        // KeyFrame中包含了地图、反过来地图中也包含了KeyFrame，相互包含
        // 在地图中添加该初始关键帧
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        // 为每个特征点构造MapPoint
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            //只有具有正深度的点才会被构造地图点
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // 通过反投影得到该特征点的世界坐标系下3D坐标
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                // 将3D点构造为MapPoint
                MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);

                // 为该MapPoint添加属性：
                // a.观测到该MapPoint的关键帧
                // b.该MapPoint的描述子
                // c.该MapPoint的平均观测方向和深度范围

                // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
                pNewMP->AddObservation(pKFini, i);
                // b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
                pNewMP->ComputeDistinctiveDescriptors();
                // c.更新该MapPoint平均观测方向以及观测距离的范围
                pNewMP->UpdateNormalAndDepth();

                // 在地图中添加该MapPoint
                mpMap->AddMapPoint(pNewMP);
                // 表示该KeyFrame的哪个特征点可以观测到哪个3D点
                pKFini->AddMapPoint(pNewMP, i);

                // 将该MapPoint添加到当前帧的mvpMapPoints中
                // 为当前Frame的特征点与MapPoint之间建立索引
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        // Create MapLines and asscoiate to KeyFrame
        // 为每个特征点构造MapLine
        for(int i = 0; i < mCurrentFrame.NL; i++) {
            float z_start = mCurrentFrame.mvDepthLineStart[i];
            float z_end = mCurrentFrame.mvDepthLineEnd[i];

            if(z_start > 0 && z_end > 0) {
                cv::Mat x3D_start = mCurrentFrame.UnprojectStereoLineStart(i);
                cv::Mat x3D_end = mCurrentFrame.UnprojectStereoLineEnd(i);

                Vector6d worldPos;
                worldPos << x3D_start.at<float>(0), x3D_start.at<float>(1), x3D_start.at<float>(2),
                            x3D_end.at<float>(0), x3D_end.at<float>(1), x3D_end.at<float>(2);

                MapLine* pNewML = new MapLine(worldPos, pKFini, mpMap);

                pNewML->AddObservation(pKFini, i);
                pNewML->ComputeDistinctiveDescriptors();
                pNewML->UpdateNormalAndDepth();

                mpMap->AddMapLine(pNewML);
                pKFini->AddMapLine(pNewML, i);

                mCurrentFrame.mvpMapLines[i] = pNewML;
            }
        }


        cout << "Keypoint size in initial map is " << mpMap->MapPointsInMap() << endl;
        cout << "Keyline size in initial map is " << mpMap->MapLinesInMap() << endl;

        // 在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        // 更新当前帧为上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        //? 这个局部地图点竟然..不在mpLocalMapper中管理?
        // 我现在的想法是，这个点只是暂时被保存在了 Tracking 线程之中， 所以称之为 local
        // 初始化之后，通过双目图像生成的地图点，都应该被认为是局部地图点
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLine(mvpLocalMapLines);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        //追踪成功
        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++) {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP) {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep) {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }

    // TODO 加入线特征
    for(int i = 0; i < mLastFrame.NL; i++) {
        MapLine* pML = mLastFrame.mvpMapLines[i];

        if(pML) {
            MapLine* pRep = pML->GetReplaced();
            if(pRep) {
                mLastFrame.mvpMapLines[i] = pRep;
            }
        }
    }
}

/*
 * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪
 *
 * Step 1：将当前普通帧的描述子转化为BoW向量
 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
 * Step 4: 通过优化3D-2D的重投影误差来获得位姿
 * Step 5：剔除优化后的匹配点中的外点
 * @return 如果匹配数超10，返回true
 *
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // TODO 加入线特征
    // TODO 点线词袋模型没有弄，这里估计暂时用不上了
    // Compute Bag of Words vector
    // Step 1：将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    LineMatcher line_matcher(0.7, true);
    vector<MapLine*> vpMapLineMatches;

    // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
    int nmatches = matcher.SearchByBoW(
            mpReferenceKF,          //参考关键帧
            mCurrentFrame,          //当前帧
            vpMapPointMatches);     //存储匹配关系

    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    // TODO 这里就还是用一样的方法来匹配
    int line_nmatches = line_matcher.SearchByProjection(mCurrentFrame, mpReferenceKF);

    // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
    // 或者直线匹配数量与当前帧检测出的直线数量的比值小于指定值
    // TODO 这里也可以改
    if(nmatches < 15 || line_nmatches < 10)
        return false;

    // Step 3:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // Step 4:通过优化3D-2D的重投影误差来获得位姿
//    Optimizer::PoseOptimization(&mCurrentFrame);
    Optimizer::PoseOptimizationWithLines(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除优化后的匹配点中的外点
    //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //如果对应到的某个特征点是外点
            if(mCurrentFrame.mvbOutlier[i])
            {
                //清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                //匹配的内点计数++
                nmatchesMap++;
        }
    }

    // Step 5：剔除地图线中的外点
    int line_nmatchesMap = 0;
    for(int i = 0; i < mCurrentFrame.NL; i++) {
        if(mCurrentFrame.mvpMapLines[i]) {
            if(mCurrentFrame.mvbLineOutlier[i]) {
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i] = false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                line_nmatchesMap--;
            } else if(mCurrentFrame.mvpMapLines[i]->Observations() > 0) {
                // 累加成功匹配到的地图点数目
                line_nmatchesMap++;
            }
        }
    }

    // Step 6：点和线的匹配总数超过20个即认为匹配成功
    // 可以改
//    return nmatchesMap >= 10;
    return nmatchesMap >= 10 && line_nmatchesMap >= 10;

}


/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints.
 *        这个函数在TrackingWithMotionModel中会用到
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints,这些点加入TemprolMapPoints
 * 这些地图点或线都是临时的，用完就删掉，不会添加到Map中去，目的是增加匹配信息，利于位姿估计
 * 函数最终的效果是为上一帧增加了一些新的地图点匹配 mLastFrame.mvpMapPoints[i]=pNewMP
 * 同时做了记录 mlpTemporalPoints.push_back(pNewMP)
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
    // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    // ref_keyframe 到 lastframe的位姿变换
    cv::Mat Tlr = mlRelativeFramePoses.back();

    // 将上一帧的世界坐标系下的位姿计算出来
    // l:last, r:reference, w:world
    // Tlw = Tlr*Trw
    mLastFrame.SetPose(Tlr*pRef->GetPose());

    // 如果上一帧为关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR)
        return;

// ================================== 为上一帧创建临时地图点 ===============================
    // Step 2：对于双目或rgbd相机，为上一帧生成新的临时地图点
    // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);

    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    // 如果上一帧中没有有效深度的点,那么就直接退出
    if(vDepthIdx.empty())
        return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.2：从中找出不是地图点的部分
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
            // 反投影到世界坐标系中
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(
                    x3D,            // 世界坐标系坐标
                    mpMap,          // 跟踪的全局地图
                    &mLastFrame,    // 存在这个特征点的帧(上一帧)
                    i);             // 特征点id

            // 加入上一帧的地图点中
            mLastFrame.mvpMapPoints[i]=pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            // 因为从近到远排序，记录其中不需要创建地图点的个数
            nPoints++;
        }

        // Step 2.4：如果地图点质量不好，停止创建地图点
        // 停止新增临时地图点必须同时满足以下条件：
        // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
        // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }

    // TODO 这里就是单纯的把上一帧的线特征转化成空间线段（如果可以的话）
// ================================== 为上一帧创建临时地图线 ===============================
    // Step 3：对于双目或rgbd相机，为上一帧生成新的临时地图线
    // <<起点深度，终点深度>, 索引>
    vector<pair<pair<float, float>, int>> vDepthIdxLine;
    vDepthIdxLine.reserve(mLastFrame.NL);

    for(int i=0; i<mLastFrame.NL;i++) {
        float z_start = mLastFrame.mvDepthLineStart[i];
        float z_end = mLastFrame.mvDepthLineEnd[i];

        if(z_start > 0 && z_end > 0) {
            vDepthIdxLine.push_back(make_pair(make_pair(z_start, z_end), i));
        }
    }

    // 如果上一帧中没有有效深度的点,那么就直接退出
    if(vDepthIdxLine.empty())
        return;

    // 起点终点的最大值，从小到大排序
    std::sort(vDepthIdxLine.begin(),vDepthIdxLine.end(),
              [](const pair<pair<float,float>, int>& a, const pair<pair<float,float>, int>& b)->bool{
                  float aDepthMax = max(a.first.first, a.first.second);
                  float bDepthMax = max(b.first.first, b.first.second);
                  return aDepthMax < bDepthMax;});

    int nLines = 0;
    for(size_t j = 0; j < vDepthIdxLine.size(); j++) {
        int i = vDepthIdxLine[j].second; // mvKeyLines中的索引

        bool bCreateNew = false;

        // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        MapLine* pML = mLastFrame.mvpMapLines[i];
        if(!pML) {
            bCreateNew = true;
        } else if(pML->Observations() < 1) {
            // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
            bCreateNew = true;
        }

        if(bCreateNew) {
            // 将线特征转化成空间线段
            // 并计算其普吕克坐标（在MapLine构造函数中计算，这里只需要给起点和终点三维坐标即可）
            cv::Mat x3D_start = mLastFrame.UnprojectStereoLineStart(i);
            cv::Mat x3D_end = mLastFrame.UnprojectStereoLineEnd(i);

            Vector6d worldPos;
            worldPos << x3D_start.at<float>(0), x3D_start.at<float>(1), x3D_start.at<float>(2),
                        x3D_end.at<float>(0), x3D_end.at<float>(1), x3D_end.at<float>(2);

            MapLine* pNewML = new MapLine(worldPos, mpMap, &mLastFrame, i);
            mLastFrame.mvpMapLines[i] = pNewML;
            mlpTemporalLines.push_back(pNewML);
            nLines++;
        } else {
            // 因为从近到远排序，记录其中不需要创建地图点的个数
            // 这个主要为了下面一步，前面按照端点深度最大值升序排列
            // 如果nLines大，说明线段距离已经比较远了
            nLines++;
        }

        // Step 2.4：如果地图点质量不好，停止创建地图点
        // 停止新增临时地图点必须同时满足以下条件：
        // 1、起点端点的深度已经超过了设定的深度阈值（35倍基线）
        // 2、nLines已经超过100个点，说明距离比较远了，可能不准确，停掉退出
        if(max(vDepthIdxLine[j].first.first, vDepthIdxLine[j].first.second) > mThDepth && nLines > 45)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    // TODO 加入线特征
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.9, true);
    LineMatcher line_matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points
    // step 1：对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
    // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
    // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
    // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
    UpdateLastFrame();

    // Step 2：根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    // 清空当前帧的地图线
    fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    // Project points seen in previous frame
    // 设置特征匹配过程中的搜索半径
    int th;
    if(mSensor!=System::STEREO)
        th=15;//单目
    else
        th=7;//双目

    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
    // Step 3：线匹配这里用的K近邻匹配，所以后面两个参数在里面没用上
    // TODO 与前一帧的匹配单独测试基本凑合，现在放进SLAM中，局部地图匹配也可以用类似的方法
    int line_nmatches = line_matcher.SearchByProjection(mCurrentFrame, mLastFrame);

    std::cout << "Nums of point matches = " << nmatches << std::endl;
    std::cout << "Nums of line matches = " << line_nmatches << std::endl;

    // If few matches, uses a wider window search
    // 如果匹配点太少，则扩大搜索半径再来一次
    // 线特征匹配点太少扩大搜索已经包含在上面的匹配方法里了
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR); // 2*th
    }

    // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
    // 或者直线匹配数量与当前帧检测出的直线数量的比值小于指定值
    // TODO 这里也可以改
    if(nmatches < 20 || line_nmatches < 15)
        return false;

    // Optimize frame pose with all matches
    // Step 4：利用3D-2D投影关系，优化当前帧位姿
    // TODO 加入线特征的图优化
//    Optimizer::PoseOptimization(&mCurrentFrame);
    Optimizer::PoseOptimizationWithLines(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除地图点中外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                // 累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }

    // Step 5：剔除地图线中的外点
    int line_nmatchesMap = 0;
    for(int i = 0; i < mCurrentFrame.NL; i++) {
        if(mCurrentFrame.mvpMapLines[i]) {
            if(mCurrentFrame.mvbLineOutlier[i]) {
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i] = false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                line_nmatchesMap--;
            } else if(mCurrentFrame.mvpMapLines[i]->Observations() > 0) {
                // 累加成功匹配到的地图点数目
                line_nmatchesMap++;
            }
        }
    }

    if(mbOnlyTracking)
    {
        // 纯定位模式下：如果成功追踪的地图点非常少,那么这里的mbVO标志就会置位
//        mbVO = nmatchesMap < 10;
//        return nmatches > 20;
        mbVO = nmatchesMap < 15 || line_nmatchesMap < 10;
        return nmatchesMap > 20 || line_nmatchesMap > 15;
    }

    // Step 6：点和线的匹配总数超过20个即认为匹配成功
    // 可以改
//    return nmatchesMap >= 10;
    return nmatchesMap >= 10 || line_nmatchesMap >= 15;
}

bool Tracking::TrackLocalMap()
{
    // TODO 加入线特征
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();


    // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchLocalPoints();
    // 筛选局部地图中新增的在视野范围内的地图线，并与当前帧匹配
    SearchLocalLines();

    // Optimize Pose
    // 在这个函数之前，在 Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化，
    // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
//    Optimizer::PoseOptimization(&mCurrentFrame);
    Optimizer::PoseOptimizationWithLines(&mCurrentFrame);

    mnMatchesInliers = 0;
    // Update MapPoints Statistics
    // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 地图点存在
        {
            if(!mCurrentFrame.mvbOutlier[i]) // 地图点不是外点
            {
                // 找到该点的帧数mnFound 加 1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                    // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                // 如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
                // ?单目就不管吗
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // TODO 加入线特征观测信息
    mnLineMatchesInliers = 0;
    for(int i = 0; i < mCurrentFrame.NL; i++) {
        if(mCurrentFrame.mvpMapLines[i]) {
            // MapLine存在
            if(!mCurrentFrame.mvbLineOutlier[i]) {
                // MapLine不是外点
                // 找到该点的帧数mnFound 加 1
                mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                if(!mbOnlyTracking) {
                    // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                    // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                    if(mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                        mnLineMatchesInliers++;
                } else
                    mnLineMatchesInliers++;
            } else if(mSensor==System::STEREO)
                // 如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
                // ?单目就不管吗
                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
    // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
    // TODO 加入线特征的重定位还没写，写完后改这里
    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers + mnLineMatchesInliers < 60)
        return false;

    //如果是正常的状态话只要跟踪的地图点大于30个就认为成功了，地图线大于20个
    // TODO 20个会不会太多了
    if(mnMatchesInliers < 30 && mnLineMatchesInliers < 20)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    // TODO 加入线的条件
    // Step 1：纯VO模式下不插入关键帧
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    // 获取当前地图中的关键帧数目
    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧

    // 地图点的最小观测次数
    int nMinObs = 3;
    if(nKFs <= 2)
        nMinObs=2;
    // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    // 同样，获取MapLines中大于观测数的MapLines
    int nRefLineMatches = mpReferenceKF->TrackedMapLines(nMinObs);

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
    int nNonTrackedClose = 0; //双目或RGB-D中没有跟踪到的近点
    int nTrackedClose = 0; //双目或RGB-D中成功跟踪的近点（三维点）
    if(mSensor!=System::MONOCULAR) {
        for(int i =0; i<mCurrentFrame.N; i++) {
            // 深度值在有效范围内
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth) {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }
    // 双目或RGBD情况下：跟踪到的地图点中近点太少 同时 没有跟踪到的三维点太多，可以插入关键帧了
    // 单目时，为false
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // 线特征也同样操作，以空间线段端点深度最大值来判断
    int nLineNonTrackedClose = 0;
    int nLineTrackedClose = 0;
    if (mSensor != System::MONOCULAR) {
        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (max(mCurrentFrame.mvDepthLineStart[i], mCurrentFrame.mvDepthLineEnd[i]) > 0 &&
                max(mCurrentFrame.mvDepthLineStart[i], mCurrentFrame.mvDepthLineEnd[i]) < mThDepth) {
                if (mCurrentFrame.mvpMapLines[i] && !mCurrentFrame.mvbLineOutlier[i]) {
                    nLineTrackedClose++;
                } else {
                    nLineNonTrackedClose++;
                }
            }
        }
    }

    // Step 7：决策是否需要插入关键帧
    // Thresholds
    // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.75f;
    // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if(nKFs<2)
        thRefRatio = 0.4f;
    //单目情况下插入关键帧的频率很高
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // Step 7.3：满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    // Step 7.4：在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        // Step 7.6：local mapping空闲时可以直接插入，不空闲的时候要根据情况插入
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    //队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    //队列中缓冲的关键帧数目太多,暂时不能插入
                    return false;
            }
            else
                //对于单目情况,就直接无法插入关键帧了
                //? 为什么这里对单目情况的处理不一样?
                //回答：可能是单目关键帧相对比较密集
                return false;
        }
    }
    else
        //不满足上面的条件,自然不能插入关键帧
        return false;
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 *
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    // 如果局部建图线程关闭了,就无法插入关键帧
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // Step 1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // Step 2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 这段代码和 Tracking::UpdateLastFrame 中的那一部分代码功能相同
    // Step 3：对于双目或rgbd摄像头，为当前帧生成新的地图点；单目无操作
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // Step 3.1：得到当前帧有深度值的特征点（不一定是地图点）
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // 第一个元素是深度,第二个元素是对应的特征点的id
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // Step 3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // Step 3.3：从中找出不是地图点的生成临时地图点
            // 处理的近点的个数
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                // 如果需要就新建地图点，这里的地图点不是临时的，是全局地图中新建地图点，用于跟踪
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    // 因为从近到远排序，记录其中不需要创建地图点的个数
                    nPoints++;
                }

                // Step 3.4：停止新建地图点必须同时满足以下条件：
                // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
                // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }

        // TODO 加入线特征
// ================================== 从关键帧创建新的地图线 ===============================
        // <<起点深度，终点深度>, 索引>
        vector<pair<pair<float, float>, int>> vDepthIdxLine;
        vDepthIdxLine.reserve(mCurrentFrame.NL);

        for(int i = 0; i < mCurrentFrame.NL; i++) {
            float z_start = mCurrentFrame.mvDepthLineStart[i];
            float z_end = mCurrentFrame.mvDepthLineEnd[i];

            if(z_start > 0 && z_end > 0) {
                // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
                vDepthIdxLine.push_back(make_pair(make_pair(z_start, z_end), i));
            }
        }

        // 如果上一帧中没有有效深度的点,那么就直接退出
        if(!vDepthIdxLine.empty()) {
            // 起点终点的最大值，从小到大排序
            std::sort(vDepthIdxLine.begin(),vDepthIdxLine.end(),
                      [](const pair<pair<float,float>, int>& a, const pair<pair<float,float>, int>& b)->bool{
                        float aDepthMax = max(a.first.first, a.first.second);
                        float bDepthMax = max(b.first.first, b.first.second);
                        return aDepthMax < bDepthMax;});

            // Step 3.3：从中找出不是地图点的生成临时地图点
            // 处理的近点的个数
            int nLines = 0;
            for(size_t j = 0; j < vDepthIdxLine.size(); j++) {
                int i = vDepthIdxLine[j].second;

                bool bCreateNew = false;

                // 如果当前帧中对应的地图线是空的，或者创建后没有被观测到，则创建新的地图线
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(!pML)
                    bCreateNew = true;
                else if(pML->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                }

                // 如果需要就新建地图线，这里的地图线不是临时的，是全局地图中新建地图线，用于跟踪
                if(bCreateNew) {
                    cv::Mat x3D_start = mCurrentFrame.UnprojectStereoLineStart(i);
                    cv::Mat x3D_end = mCurrentFrame.UnprojectStereoLineEnd(i);
                    Vector6d worldPos;
                    worldPos << x3D_start.at<float>(0), x3D_start.at<float>(1), x3D_start.at<float>(2),
                            x3D_end.at<float>(0), x3D_end.at<float>(1), x3D_end.at<float>(2);

                    MapLine* pNewML = new MapLine(worldPos, pKF, mpMap);

                    // 这些添加属性的操作是每次创建MapLine后都要做的
                    pNewML->AddObservation(pKF, i);
                    pKF->AddMapLine(pNewML, i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateNormalAndDepth();
                    mpMap->AddMapLine(pNewML);

                    mCurrentFrame.mvpMapLines[i] = pNewML;
                    nLines++;
                } else {
                    // 因为从近到远排序，记录其中不需要创建地图点的个数
                    nLines++;
                }

                // Step 3.4：停止新建地图点必须同时满足以下条件：
                // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
                // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
                if(max(vDepthIdxLine[j].first.first, vDepthIdxLine[j].first.second) > mThDepth && nLines > 45)
                    break;
            }
        }
    }

    // Step 4：插入关键帧
    // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
    mpLocalMapper->InsertKeyFrame(pKF);

    // 插入好了，允许局部建图停止
    mpLocalMapper->SetNotStop(false);
    mpPointCloudMapping->InsertKeyFrame(pKF, this->mImRGB, this->mImDepth);

    // 当前帧成为新的关键帧，更新
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1(被当前帧观测了)
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点在后面搜索匹配时不被投影，因为已经有匹配了
                pMP->mbTrackInView = false;
            }
        }
    }

    // 准备进行投影匹配的点的数目
    int nToMatch=0;

    // Project points in frame and check its visibility
    // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // 跳过坏点
        if(pMP->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        // 判断地图点是否在在当前帧视野内
        if(mCurrentFrame.IsInFrustum(pMP, 0.5))
        {
            // 观测到该点的帧数加1
            pMP->IncreaseVisible();
            // 只有在视野范围内的地图点才参与之后的投影匹配
            nToMatch++;
        }
    }

    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)   //RGBD相机输入的时候,搜索的阈值会变得稍微大一些
            th=3;

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        // 投影匹配得到更多的匹配关系
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::SearchLocalLines() {
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++) {
        MapLine* pML = *vit;
        if(pML) {
            if(pML->isBad()) {
                *vit = static_cast<MapLine*>(NULL);
            } else {
                pML->IncreaseVisible();
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                pML->mbTrackInView = false;
            }
        }
    }

    // 准备进行投影匹配的点的数目
    int nToMatch=0;

    // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
    for(vector<MapLine*>::iterator vit=mvpLocalMapLines.begin(), vend=mvpLocalMapLines.end(); vit!=vend; vit++) {
        MapLine* pML = *vit;
        // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
        if(pML->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // 跳过坏点
        if(pML->isBad())
            continue;
        // 判断MapLines是否在当前帧视野内，包括部分观测
        if(mCurrentFrame.IsInFrustum(pML, 0.5)) {
            // 观测到或部分观测到该MapLine的帧数+1
            pML->IncreaseVisible();
            // 只有在视野范围内的地图点才参与之后的投影匹配
            nToMatch++;
        }
    }

    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if(nToMatch > 0) {
        // TODO 这个的阈值可以该，原来是针对点的，当然也可以考虑采用别的局部地图线匹配策略
        LineMatcher line_matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th=5;
        // 投影匹配得到更多的匹配关系
        line_matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines);
    }
}

void Tracking::UpdateLocalMap()
{
    // TODO 加入线特征
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLine(mvpLocalMapLines);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalLines();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalLines() {
    // 和上面哪个UpdateLocalPoints基本一样
    mvpLocalMapLines.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++) {
        KeyFrame* pKF = *itKF;
        const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();

        for(vector<MapLine*>::const_iterator itMP=vpMLs.begin(), itEndMP=vpMLs.end(); itMP!=itEndMP; itMP++) {
            MapLine* pML = *itMP;
            if(!pML)
                continue;
            if(pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if(!pML->isBad()) {
                mvpLocalMapLines.push_back(pML);
                pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @details 重定位过程
 * @return true
 * @return false
 *
 * Step 1：计算当前帧特征点的词袋向量
 * Step 2：找到与当前帧相似的候选关键帧
 * Step 3：通过BoW进行匹配
 * Step 4：通过EPnP算法估计姿态
 * Step 5：通过PoseOptimization对姿态进行优化求解
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization()
{
    // TODO 重定位暂时不用线特征
    // Compute Bag of Words Vector
    // Step 1：计算当前帧特征点的词袋向量
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：用词袋找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    // 如果没有候选关键帧，则退出
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    //每个关键帧的解算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    //每个关键帧和当前帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    //有效的候选关键帧数目
    int nCandidates=0;

    // Step 3：遍历所有的候选关键帧，通过词袋进行快速匹配，用匹配结果初始化PnP Solver
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches，nmatches表示匹配的数目
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            // 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 如果匹配数目够用，用匹配结果初始化EPnPsolver
                // 为什么用EPnP? 因为计算复杂度低，精度高
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                        0.99,   //用于计算RANSAC迭代次数理论值的概率
                        10,     //最小内点数, 但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                        300,    //最大迭代次数
                        4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                        0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                        5.991); // 自由度为2的卡方检验的阈值,程序中还会根据特征点所在的图层对这个阈值进行缩放
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 这里的 P4P RANSAC是Epnp，每次迭代需要4个点
    // 是否已经找到相匹配的关键帧的标志
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // Step 4: 通过一系列操作,直到找到能够匹配上的关键帧
    // 为什么搞这么复杂？答：是担心误闭环
    while(nCandidates>0 && !bMatch)
    {
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            // 忽略放弃的
            if(vbDiscarded[i])
                continue;

            //内点标记
            vector<bool> vbInliers;

            //内点数
            int nInliers;

            // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            bool bNoMore;

            // Step 4.1：通过EPnP算法估计姿态，迭代5次
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // bNoMore 为true 表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                //  Step 4.2：如果EPnP 计算出了位姿，对内点进行BA优化
                Tcw.copyTo(mCurrentFrame.mTcw);

                // EPnP 里RANSAC后的内点的集合
                set<MapPoint*> sFound;

                const int np = vbInliers.size();
                //遍历所有内点
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // 如果优化之后的内点数目不多，跳过了当前候选关键帧,但是却没有放弃当前帧的重定位
                if(nGood<10)
                    continue;

                // 删除外点对应的地图点
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
                    int nadditional = matcher2.SearchByProjection(
                            mCurrentFrame,          //当前帧
                            vpCandidateKFs[i],      //关键帧
                            sFound,                 //已经找到的地图点集合，不会用于PNP
                            10,                     //窗口阈值，会乘以金字塔尺度
                            100);                   //匹配的ORB描述子距离应该小于这个阈值

                    // 如果通过投影过程新增了比较多的匹配特征点对
                    if(nadditional+nGood>=50)
                    {
                        // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下, 最后垂死挣扎
                        // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
                        // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
                        if(nGood>30 && nGood<50)
                        {
                            // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(
                                    mCurrentFrame,          //当前帧
                                    vpCandidateKFs[i],      //候选的关键帧
                                    sFound,                 //已经找到的地图点，不会用于PNP
                                    3,                      //新的窗口阈值，会乘以金字塔尺度
                                    64);                    //匹配的ORB描述子距离应该小于这个阈值

                            // Final optimization
                            // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                //更新地图点
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                            //如果还是不能够满足就放弃了
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
                if(nGood>=50)
                {
                    bMatch = true;
                    // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                    break;
                }
            }
        }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    // 折腾了这么久还是没有匹配上，重定位失败
    if(!bMatch)
    {
        return false;
    }
    else
    {
        // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
        // 记录成功重定位帧的id，防止短时间多次重定位
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
