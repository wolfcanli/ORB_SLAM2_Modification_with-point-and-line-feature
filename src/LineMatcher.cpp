//
// Created by jiajieshi on 22-9-5.
//

#include "LineMatcher.h"

namespace ORB_SLAM2 {
// 要用到的一些阈值


// 构造函数,参数默认值为0.6,true
LineMatcher::LineMatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{

}

// Bit set count operation from
// Hamming distance：两个二进制串之间的汉明距离，指的是其不同位数的个数
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int LineMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
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


/**
 * @brief 当前帧和前一帧的线特征匹配
 * 这个用来和上一普通帧匹配，匹配方法很多，后续可以改，这里先用普通的K近邻匹配
 * TODO 使用别的匹配方法
 * 对于LastFrame中的MapLine，将其投影到CurrentFrame中
 * Reference to 《基于点线综合特征的双目视觉SLAM方法》谢晓佳
 * 需要考虑线段被部分观测到的情况
 * Step 1 对于前一帧的每一个MapLine，将端点投影到当前帧上
 * Step 2 投影之后分三种情况
 *   Step 2.1 两端点都在相机后方，跳过
 *   Step 2.2 一个在前，一个在后。求直线与图像平面的交点，以这个交点作为图像内线段的端点
 *            端点计算 Xik = Xsk + lambda * (Xsk - Xek)
 *            其中Xik[2] = 0，可以计算除lambda，从而计算除Xik[0], Xik[1]
 *            但端点也有可能跑到图像边界外面，应该也需要裁剪
 *   Step 2.2 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
 *            保留图像边界内的部分。Liang-Barsky线段裁剪算法
 * Step 3 成功投影之后，在投影线周围指定区域内搜索线特征
 *        点特征容易判断是否在区域内，线特征可能只有部分在内，应该可以判断某个端点在内即可
 *        以此来获取候选线特征
 * Step 4 匹配候选线特征
 *        匹配策略：角度差、长度比值、重叠长度、描述子距离
 *        匹配数量较少时可以增大阈值
 *        但好像搜索范围没有减少，上面只是搜索条件，但是线特征以什么标准来限定范围呢？
 * @param[in] CurrentFrame       当前帧
 * @param[in] LastFrame          上一帧
 * @param[in] th                 搜索范围
 * @param[in] bMono              是否是单目相机，不过这里好像也不用，回头删了
 * @return int                   成功匹配的数目
 */
int LineMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int line_nmatches = 0;

//    cv::Mat Tcw_current = CurrentFrame.mTcw;
//    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
//    Tcw(0, 0) = Tcw_current.at<double>(0, 0);
//    Tcw(0, 1) = Tcw_current.at<double>(0, 1);
//    Tcw(0, 2) = Tcw_current.at<double>(0, 2);
//    Tcw(0, 3) = Tcw_current.at<double>(0, 3);
//
//    Tcw(1, 0) = Tcw_current.at<double>(1, 0);
//    Tcw(1, 1) = Tcw_current.at<double>(1, 1);
//    Tcw(1, 2) = Tcw_current.at<double>(1, 2);
//    Tcw(1, 3) = Tcw_current.at<double>(1, 3);
//
//    Tcw(2, 0) = Tcw_current.at<double>(2, 0);
//    Tcw(2, 1) = Tcw_current.at<double>(2, 1);
//    Tcw(2, 2) = Tcw_current.at<double>(2, 2);
//    Tcw(2, 3) = Tcw_current.at<double>(2, 3);
//
//    for (int i = 0; i < LastFrame.mvpMapLines.size(); i++) {
//        // Step 1 对于前一帧的每一个MapLine，将端点投影到当前帧上
//        MapLine* pML = LastFrame.mvpMapLines[i];
//        Eigen::Vector3d Xw_start = pML->mStart3d;
//        Eigen::Vector3d Xw_end = pML->mEnd3d;
//
//        Eigen::Vector3d Xc_start = Tcw * Xw_start;
//        Eigen::Vector3d Xc_end = Tcw * Xw_start;
//
//        // Step 2 投影之后分三种情况
//
//        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
//            // Step 2.1 两端点都在相机后方，跳过
//            continue;
//        }
//        if (Xc_start[2] < 0 || Xc_end[2] < 0) {
//            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
//            // 然后把前面那个端点留下
//            // 先求交点
//            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
//            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
//            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);
//
//            Eigen::Vector3d Xc_cross(x_c_cross, y_c_cross, 0); // 交点
//
//
//        }
//    }


    cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);

    // Matches. Each matches[i] is k or less matches for the same query descriptor.
    std::vector<std::vector<cv::DMatch>> line_matches;
    bfm->knnMatch(LastFrame.mLineDescriptors, CurrentFrame.mLineDescriptors, line_matches, 2);

    for(size_t i = 0;i < line_matches.size();i++)
    {
        const cv::DMatch& bestMatch = line_matches[i][0];
        const cv::DMatch& betterMatch = line_matches[i][1];
        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < 0.75)
            line_nmatches++;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "CurrentFrame and LastFrame match time: " << ttrack * 1000 << " ms" << std::endl;

    return line_nmatches;
}

// 加一个用来和参考关键帧匹配的
int LineMatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *RefFrame, std::vector<MapLine*> &vpMapLineMatches) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    int line_nmatches = 0;
    cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
    vpMapLineMatches = vector<MapLine*>(CurrentFrame.NL, static_cast<MapLine*>(NULL));
    // 获取参考关键帧的地图点
    const vector<MapLine*> vpMapLinesKF = RefFrame->GetMapLineMatches();

    // Matches. Each matches[i] is k or less matches for the same query descriptor.
    std::vector<std::vector<cv::DMatch>> line_matches;
    bfm->knnMatch(RefFrame->mLineDescriptors, CurrentFrame.mLineDescriptors, line_matches, 2);

    for(size_t i = 0;i < line_matches.size();i++) {
        const cv::DMatch& bestMatch = line_matches[i][0];
        const cv::DMatch& betterMatch = line_matches[i][1];

        MapLine* pML = vpMapLinesKF[bestMatch.queryIdx];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < 0.75) {
            // 用来给CurrentFrame的地图线
            vpMapLineMatches[bestMatch.trainIdx] = pML;

            line_nmatches++;
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "CurrentFrame and RefFrame match time: " << ttrack * 1000 << " ms" << std::endl;

    return line_nmatches;
}

/**
 * @brief 这个用来和局部地图匹配。把局部地图线投影到当前帧中，方法和前面相邻帧匹配一样
 * @param[in] F                   当前帧
 * @param[in] vpMapLines          局部地图线，在执行这个函数前通过局部关键帧构建的
 * @param[in] th                  搜索范围
 * @return int                    成功匹配的数目
*/
int LineMatcher::SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th) {
    // TODO 与局部地图的地图线匹配
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    int line_nmatches = 0;

    // 如果 th！=1 (RGBD 相机或者刚刚进行过重定位), 需要扩大范围搜索
    const bool bFactor = (th != 1.0);

    // Step 1 遍历有效的局部地图点
    for(size_t iML = 0; iML < vpMapLines.size(); iML++) {
        MapLine* pML = vpMapLines[iML];

        // 判断该点是否要投影
        if(!pML->mbTrackInView)
            continue;

        if(pML->isBad())
            continue;

        // 通过距离预测的金字塔层数，该层数相对于当前的帧
        const int &nPredictedLevel = pML->mnTrackScaleLevel;



        // The size of the window will depend on the viewing direction
        // Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
        float r = 0;
        if(pML->mTrackViewCos > 0.998)
            r = 2.5;
        else
            r = 4.0;

        // 如果需要扩大范围搜索，则乘以阈值th
        if(bFactor)
            r *= th;

        // Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
//        Frame::GetLinesInArea(const float &x_s, const float &y_s,
//        const float &x_e, const float &y_e,
//        const float &r,
//        const int minLevel, const int maxLevel)
        const std::vector<size_t> vLineIndices =
                F.GetLinesInArea(pML->mTrackProjStartX, pML->mTrackProjStartY,
                                 pML->mTrackProjEndX, pML->mTrackProjEndY,
                                 r * F.mvScaleFactors[nPredictedLevel],
                                 nPredictedLevel - 1, nPredictedLevel);

        // 没找到候选的,就放弃对当前点的匹配
        if(vLineIndices.empty())
            continue;

        const cv::Mat MLdescriptor = pML->GetDescriptor();

        // 最优的次优的描述子距离和index
        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1 ;

        // Get best and second matches with near keypoints
        // Step 4 寻找候选匹配点中的最佳和次佳匹配点
        for(vector<size_t>::const_iterator vit=vLineIndices.begin(), vend=vLineIndices.end(); vit!=vend; vit++) {
            const size_t idx = *vit;

            // 如果Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
            if(F.mvpMapLines[idx])
                if(F.mvpMapLines[idx]->Observations() > 0)
                    continue;

            //如果是双目数据
            if(F.mvuRight[idx]>0) {
                //计算在X轴上的投影误差
                const float er_start = fabs(pML->mTrackProjStartXR - F.mvuRightLineStart[idx]);
                const float er_end = fabs(pML->mTrackProjEndXR - F.mvuRightLineEnd[idx]);
                //超过阈值,说明这个点不行,丢掉.
                //这里的阈值定义是以给定的搜索范围r为参考,然后考虑到越近的点(nPredictedLevel越大), 相机运动时对其产生的影响也就越大,
                //因此需要扩大其搜索空间.
                //当给定缩放倍率为1.2的时候, mvScaleFactors 中的数据是: 1 1.2 1.2^2 1.2^3 ...
                if(er_start > r*F.mvScaleFactors[nPredictedLevel] || er_end > r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &desc_l = F.mLineDescriptors.row(idx);

            // 计算地图点和候选投影点的描述子距离
            const int dist = DescriptorDistance(MLdescriptor, desc_l);

            // 寻找描述子距离最小和次小的特征点和索引
            if(dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeyLinesUn[idx].octave;
                bestIdx = idx;
            } else if(dist < bestDist2) {
                bestLevel2 = F.mvKeyLinesUn[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        // Step 5 筛选最佳匹配点
        // 最佳匹配距离还需要满足在设定阈值内
        if(bestDist <= 100) {
            // 条件1：bestLevel==bestLevel2 表示 最佳和次佳在同一金字塔层级
            // 条件2：bestDist>mfNNratio*bestDist2 表示最佳和次佳距离不满足阈值比例。理论来说 bestDist/bestDist2 越小越好
            if(bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                continue;

            //保存结果: 为Frame中的特征点增加对应的MapPoint
            F.mvpMapLines[bestIdx]=pML;
            line_nmatches++;
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "Match local map costs " << ttrack * 1000 << " ms" << std::endl;

    return line_nmatches;
}

/// Matching to triangulate new MapPoints. Check Epipolar Constraint.
// todo 这个函数也需要改！类似于ORB，比如检查旋转一致性
// 函数返回的是两个关键帧之间特征匹配的索引号对
int LineMatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                        vector<pair<size_t, size_t>> &vMatchedPairs,
                                        const bool bOnlyStereo) {
    vMatchedPairs.clear();
    int nmatches = 0;
    cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
    cv::Mat ldesc1, ldesc2;
    vector<vector<cv::DMatch>> lmatches;

    ldesc1 = pKF1->mLineDescriptors;
    ldesc2 = pKF2->mLineDescriptors;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    pKF1->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.1;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for(int i=0; i<lmatches.size(); i++)
    {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12>nn12_dist_th)
        {
            vMatchedPairs.push_back(make_pair(qdx, tdx));
            nmatches++;
        }
    }

    return nmatches;
}

/// Project MapPoints into KeyFrame and search for duplicated MapPoints.
int LineMatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines)
{
    cv::Mat Rcw = pKF->GetRotation();  //关键帧的旋转和平移
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter(); //关键帧的相机光心位置

    int nFused=0;

    cv::Mat lineDesc = pKF->mLineDescriptors;   //待Fuse的关键帧的描述子，这是关键帧上线特征的所有描述子？
    const int nMLs = vpMapLines.size();  // 所有地图线的数量

    //遍历所有的MapLines
    for(int i=0; i<nMLs; i++)
    {
        MapLine* pML = vpMapLines[i];

        if(!pML)  // 地图线为空跳过
            continue;

        if(pML->isBad() || pML->IsInKeyFrame(pKF))  //第二个条件：关键帧在地图线的Observation中，map类型
            continue;
#if 0
        Vector6d LineW = pML->GetWorldPos();
    cv::Mat LineSW = (Mat_<float>(3,1) << LineW(0), LineW(1), LineW(2));
    cv::Mat LineSC = Rcw*LineSW + tcw;
    cv::Mat LineEW = (Mat_<float>(3,1) << LineW(3), LineW(4), LineW(5));
    cv::Mat LineEC = Rcw*LineEW + tcw;

    //Depth must be positive
    if(LineSC.at<float>(2)<0.0f || LineEC.at<float>(2)<0.0f)
        continue;

    // 获取起始点在图像上的投影坐标
    const float invz1 = 1/LineSC.at<float>(2);
    const float x1 = LineSC.at<float>(0)*invz1;
    const float y1 = LineSC.at<float>(1)*invz1;

    const float u1 = fx*x1 + cx;
    const float v1 = fy*y1 + cy;

    // 获取终止点在图像上的投影坐标
    const float invz2 = 1/LineEC.at<float>(2);
    const float x2 = LineEC.at<float>(0)*invz2;
    const float y2 = LineEC.at<float>(1)*invz2;

    const float u2 = fx*x2 + cx;
    const float v2 = fy*y2 + cy;
#endif
        cv::Mat CurrentLineDesc = pML->mLineDescriptor; //MapLine[i]对应的线特征描述子,这应该是线特征的最优描述子

#if 0
        // 采用暴力匹配法,knnMatch
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    vector<vector<DMatch>> lmatches;
    bfm->knnMatch(CurrentLineDesc, lineDesc, lmatches, 2);  //当前地图线的描述子和关键帧上线的所有描述子进行匹配
    double nn_dist_th, nn12_dist_th;
    pKF->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.1;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for(int i=0; i<lmatches.size(); i++)
    {
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12>nn12_dist_th)    //找到了pKF中对应ML
        {
            MapLine* pMLinKF = pKF->GetMapLine(tdx);
            if(pMLinKF)
            {
                if(!pMLinKF->isBad())
                {
                    if(pMLinKF->Observations()>pML->Observations())
                        pML->Replace(pMLinKF);
                    else
                        pMLinKF->Replace(pML);
                }
            }
            nFused++;
        }
    }
#elif 1

        //todo 下面这部分代码似乎有问题呢。。参考下ORB中的代码
        // 使用暴力匹配法
        cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
        vector<cv::DMatch> lmatches;

        // 一个描述子和很多个描述子进行匹配，输出的是一维数组
        matcher->match ( CurrentLineDesc, lineDesc, lmatches );

        double max_dist = 0;
        double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < CurrentLineDesc.rows; i++ )  //todo CurrentLineDesc.row?这应该是一个特征线的描述子啊？ 修改为lmatches.size()看看
        {
            double dist = lmatches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // "good" matches (i.e. whose distance is less than 2*min_dist ) todo 这样不就变成了一个地图线特征对应很多个关键帧上的线特征了吗？
        std::vector<cv::DMatch> good_matches;
        for( int i = 0; i < CurrentLineDesc.rows; i++ ) {
            if( lmatches[i].distance < 1.5*min_dist ) {
                int tdx = lmatches[i].trainIdx;
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if(pMLinKF) {
                    if(!pMLinKF->isBad()) {
                        if(pMLinKF->Observations()>pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }

#else
        cout << "CurrentLineDesc.empty() = " << CurrentLineDesc.empty() << endl;
        cout << "lineDesc.empty() = " << lineDesc.empty() << endl;
        cout << CurrentLineDesc << endl;
        if(CurrentLineDesc.empty() || lineDesc.empty())
            continue;

        // 采用Flann方法
        FlannBasedMatcher flm;
        vector<DMatch> lmatches;
        flm.match(CurrentLineDesc, lineDesc, lmatches);

        double max_dist = 0;
        double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        cout << "CurrentLineDesc.rows = " << CurrentLineDesc.rows << endl;
        for( int i = 0; i < CurrentLineDesc.rows; i++ )
        { double dist = lmatches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // "good" matches (i.e. whose distance is less than 2*min_dist )
        std::vector< DMatch > good_matches;
        for( int i = 0; i < CurrentLineDesc.rows; i++ )
        {
            if( lmatches[i].distance < 2*min_dist )
            {
                int tdx = lmatches[i].trainIdx;
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                    {
                        if(pMLinKF->Observations()>pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }
#endif
    }
    return nFused;
}


} // namespace ORB_SLAM2