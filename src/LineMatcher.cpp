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
 * TODO 使用别的方法匹配上一帧
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
 *   Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
 *            保留图像边界内的部分。Liang-Barsky线段裁剪算法
 * Step 3 成功投影之后，在投影线周围指定区域内搜索线特征
 *        点特征容易判断是否在区域内，线特征可能只有部分在内，应该可以判断某个端点在内即可
 *        以此来获取候选线特征
 *        // TODO step3没有做
 * Step 4 匹配候选线特征
 *        匹配策略：角度差、长度比值、重叠长度、描述子距离
 *        匹配数量较少时可以增大阈值
 *
 * @param[in] CurrentFrame       当前帧
 * @param[in] LastFrame          上一帧
 * @param[in] th                 搜索范围，这里暂时也用不上
 * @param[in] bMono              是否是单目相机，不过这里好像也不用，回头删了
 * @return int                   成功匹配的数目
 */
int LineMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 获取当前帧位姿
    // 这里的初始位姿，是上一帧的位姿，来自恒速模型的初始值
    cv::Mat Tcw_current = CurrentFrame.mTcw;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw(0, 0) = Tcw_current.at<float>(0, 0);
    Tcw(0, 1) = Tcw_current.at<float>(0, 1);
    Tcw(0, 2) = Tcw_current.at<float>(0, 2);
    Tcw(0, 3) = Tcw_current.at<float>(0, 3);

    Tcw(1, 0) = Tcw_current.at<float>(1, 0);
    Tcw(1, 1) = Tcw_current.at<float>(1, 1);
    Tcw(1, 2) = Tcw_current.at<float>(1, 2);
    Tcw(1, 3) = Tcw_current.at<float>(1, 3);

    Tcw(2, 0) = Tcw_current.at<float>(2, 0);
    Tcw(2, 1) = Tcw_current.at<float>(2, 1);
    Tcw(2, 2) = Tcw_current.at<float>(2, 2);
    Tcw(2, 3) = Tcw_current.at<float>(2, 3);

    std::vector<float> bounds = {CurrentFrame.mnMinX, CurrentFrame.mnMinY,
                                 CurrentFrame.mnMaxX, CurrentFrame.mnMaxY};

    // 存储新的裁剪后的KeyLine，后面用来匹配的，需要和mvpMapLines索引相同
    std::vector<KeyLine> new_KeyLines;
    // 这个用来记录新创建的KL，在LastFrame.mvpMapLines中的索引，用于后面给当前帧传递数据用
    std::vector<int> new_kl_index;
    cv::Mat new_descriptors;
    new_KeyLines.reserve(LastFrame.mvpMapLines.size());

    // 先遍历投影
    for (int i = 0; i < LastFrame.mvpMapLines.size(); i++) {
        // Step 1 对于前一帧的每一个MapLine，将端点投影到当前帧上
        MapLine* pML = LastFrame.mvpMapLines[i];
        if (!pML)
            continue;
        if (LastFrame.mvbLineOutlier[i])
            continue;
        if (pML->isBad())
            continue;
        Eigen::Vector3d Xw_start = pML->mStart3d;
        Eigen::Vector3d Xw_end = pML->mEnd3d;
        // 计算相机坐标下的坐标值
        Eigen::Vector3d Xc_start = Tcw * Xw_start;
        Eigen::Vector3d Xc_end = Tcw * Xw_end;

        // Step 2 投影之后分三种情况
        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
            // Step 2.1 两端点都在相机后方，跳过
            continue;
        }
        if (Xc_start[2] < 0.0 || Xc_end[2] < 0.0) {
            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
            // 然后把前面那个端点留下
            // 先求交点
            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);

            Eigen::Vector3d P_c_cross(x_c_cross, y_c_cross, 0); // 交点

            Eigen::Vector4d line_proj;

            if (Xc_start[2] < 0.0) {
                // 如果是起点在后面，则把原来的起点替换成交点
//                pML->SetStartPos(P_c_cross);
                float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
                float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(x_c_cross, y_c_cross, u_end, v_end);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);
                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            } else if (Xc_end[2] < 0.0) {
                // 如果是终点在后面，则把原来的终点替换成交点
//                pML->SetEndPos(P_c_cross);
                float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
                float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(u_start, v_start, x_c_cross, y_c_cross);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);
                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            }
        }
        if (Xc_start[2] > 0.0 && Xc_end[2] > 0.0) {
            // Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
            // 保留图像边界内的部分。Liang-Barsky线段裁剪算法
            float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
            float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;

            float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
            float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;

            Eigen::Vector4d line_proj(u_start, v_start, u_end, v_end);

            // 下面这套可以照搬
            Eigen::Vector4d new_line_proj;
            bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
            if (ok) {
                // 线段成功裁剪
                // 修改原来的KeyLine
                KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                new_KeyLines.push_back(kl_to_change);
                new_kl_index.push_back(i);
                new_descriptors.push_back(pML->mLineDescriptor.clone());
            } else {
                continue;
            }
        }
    }

    // 全部投影完后再匹配
    int line_nmatches = 0;
    bool match_ok;
    std::vector<std::pair<int, int>> match_indices;

    for (int j = 0; j < CurrentFrame.NL; j++) {
        // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
        if(CurrentFrame.mvpMapLines[j])
            if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                continue;
        for (int i = 0; i < new_KeyLines.size(); i++) {
            // 输入两个KeyLine，输出是否匹配
            match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                    new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j));
            if (match_ok) {
                // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                MapLine* pML_from_last = LastFrame.mvpMapLines[new_kl_index[i]];
                CurrentFrame.mvpMapLines[j] = pML_from_last;
                line_nmatches++;
            } else {
                continue;
            }
        }
    }

    if (line_nmatches * 1.0 / CurrentFrame.NL < 0.2) {
        // 匹配数量不够，放大阈值再来
        std::vector<double> offset = {10.0, -0.1, -0.1, 5, 10};
        line_nmatches = 0;
        fill(CurrentFrame.mvpMapLines.begin(), CurrentFrame.mvpMapLines.end(), static_cast<ORB_SLAM2::MapLine*>(NULL));

        for (int j = 0; j < CurrentFrame.NL; j++) {
            // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
            if(CurrentFrame.mvpMapLines[j])
                if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                    continue;
            for (int i = 0; i < new_KeyLines.size(); i++) {
                // 输入两个KeyLine，输出是否匹配
                match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                        new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j),
                                        offset);
                if (match_ok) {
                    // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                    MapLine* pML_from_last = LastFrame.mvpMapLines[new_kl_index[i]];
                    CurrentFrame.mvpMapLines[j] = pML_from_last;
                    line_nmatches++;
                } else {
                    continue;
                }
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "CurrentFrame and LastFrame match time: " << ttrack * 1000 << " ms" << std::endl;
    std::cout << "CurrentFrame has matched " << line_nmatches << " MapLines from LastFrame" << std::endl;

    return line_nmatches;
}

// TODO  单元测试用，后续删掉
int LineMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, std::vector<KeyLine>& new_kls, std::vector<std::pair<int, int>>& match_indices) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 获取当前帧位姿
    // 这里的初始位姿，是上一帧的位姿，来自恒速模型的初始值
    cv::Mat Tcw_current = CurrentFrame.mTcw;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw(0, 0) = Tcw_current.at<float>(0, 0);
    Tcw(0, 1) = Tcw_current.at<float>(0, 1);
    Tcw(0, 2) = Tcw_current.at<float>(0, 2);
    Tcw(0, 3) = Tcw_current.at<float>(0, 3);

    Tcw(1, 0) = Tcw_current.at<float>(1, 0);
    Tcw(1, 1) = Tcw_current.at<float>(1, 1);
    Tcw(1, 2) = Tcw_current.at<float>(1, 2);
    Tcw(1, 3) = Tcw_current.at<float>(1, 3);

    Tcw(2, 0) = Tcw_current.at<float>(2, 0);
    Tcw(2, 1) = Tcw_current.at<float>(2, 1);
    Tcw(2, 2) = Tcw_current.at<float>(2, 2);
    Tcw(2, 3) = Tcw_current.at<float>(2, 3);

    std::vector<float> bounds = {CurrentFrame.mnMinX, CurrentFrame.mnMinY,
                                 CurrentFrame.mnMaxX, CurrentFrame.mnMaxY};

    // 存储新的裁剪后的KeyLine，后面用来匹配的，需要和mvpMapLines索引相同
    std::vector<KeyLine> new_KeyLines;
    // 这个用来记录新创建的KL，在LastFrame.mvpMapLines中的索引，用于后面给当前帧传递数据用
    std::vector<int> new_kl_index;
    cv::Mat new_descriptors;
    new_KeyLines.reserve(LastFrame.mvpMapLines.size());

    // 先遍历投影
    for (int i = 0; i < LastFrame.mvpMapLines.size(); i++) {
        // Step 1 对于前一帧的每一个MapLine，将端点投影到当前帧上
        MapLine* pML = LastFrame.mvpMapLines[i];
        if (!pML)
            continue;
        if (LastFrame.mvbLineOutlier[i])
            continue;
        if (pML->isBad())
            continue;
        Eigen::Vector3d Xw_start = pML->mStart3d;
        Eigen::Vector3d Xw_end = pML->mEnd3d;
        // 计算相机坐标下的坐标值
        Eigen::Vector3d Xc_start = Tcw * Xw_start;
        Eigen::Vector3d Xc_end = Tcw * Xw_end;

        // Step 2 投影之后分三种情况
        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
            // Step 2.1 两端点都在相机后方，跳过
            continue;
        }
        if (Xc_start[2] < 0.0 || Xc_end[2] < 0.0) {
            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
            // 然后把前面那个端点留下
            // 先求交点
            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);

            Eigen::Vector3d P_c_cross(x_c_cross, y_c_cross, 0); // 交点

            Eigen::Vector4d line_proj;

            if (Xc_start[2] < 0.0) {
                // 如果是起点在后面，则把原来的起点替换成交点
//                pML->SetStartPos(P_c_cross);
                float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
                float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(x_c_cross, y_c_cross, u_end, v_end);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);

                    new_kls.push_back(kl_to_change);

                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            } else if (Xc_end[2] < 0.0) {
                // 如果是终点在后面，则把原来的终点替换成交点
//                pML->SetEndPos(P_c_cross);
                float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
                float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(u_start, v_start, x_c_cross, y_c_cross);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);

                    new_kls.push_back(kl_to_change);

                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            }
        }
        if (Xc_start[2] > 0.0 && Xc_end[2] > 0.0) {
            // Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
            // 保留图像边界内的部分。Liang-Barsky线段裁剪算法
            float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
            float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;

            float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
            float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;

            Eigen::Vector4d line_proj(u_start, v_start, u_end, v_end);

            // 下面这套可以照搬
            Eigen::Vector4d new_line_proj;
            bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
            if (ok) {
                // 线段成功裁剪
                // 修改原来的KeyLine
                KeyLine kl_to_change = LastFrame.mvKeyLinesUn[i];
                UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                new_KeyLines.push_back(kl_to_change);

                new_kls.push_back(kl_to_change);

                new_kl_index.push_back(i);
                new_descriptors.push_back(pML->mLineDescriptor.clone());
            } else {
                continue;
            }
        }
    }

    // 全部投影完后再匹配
    int line_nmatches = 0;
    bool match_ok;

    for (int j = 0; j < CurrentFrame.NL; j++) {
        for (int i = 0; i < new_KeyLines.size(); i++) {
            // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
            if(CurrentFrame.mvpMapLines[j])
                if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                    continue;

            // 输入两个KeyLine，输出是否匹配
            match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                    new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j));
            if (match_ok) {
                // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                MapLine* pML_from_last = LastFrame.mvpMapLines[new_kl_index[i]];
                CurrentFrame.mvpMapLines[j] = pML_from_last;

                match_indices.push_back(std::pair<int, int>(i, j));

                line_nmatches++;
            } else {
                continue;
            }
        }
    }

    if (line_nmatches * 1.0 / CurrentFrame.NL < 0.2) {
        // 匹配数量不够，放大阈值再来
        std::vector<double> offset = {10.0, -0.1, -0.1, 5, 10};
        line_nmatches = 0;
        fill(CurrentFrame.mvpMapLines.begin(), CurrentFrame.mvpMapLines.end(), static_cast<ORB_SLAM2::MapLine*>(NULL));
        match_indices.clear();

        for (int j = 0; j < CurrentFrame.NL; j++) {
            for (int i = 0; i < new_KeyLines.size(); i++) {
                // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
                if(CurrentFrame.mvpMapLines[j])
                    if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                        continue;

                // 输入两个KeyLine，输出是否匹配
                match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                        new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j),
                                        offset);
                if (match_ok) {
                    // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                    MapLine* pML_from_last = LastFrame.mvpMapLines[new_kl_index[i]];
                    CurrentFrame.mvpMapLines[j] = pML_from_last;

                    match_indices.push_back(std::pair<int, int>(i, j));

                    line_nmatches++;
                } else {
                    continue;
                }
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "CurrentFrame and LastFrame match time: " << ttrack * 1000 << " ms" << std::endl;
    std::cout << "CurrentFrame has matched " << line_nmatches << " MapLines from LastFrame" << std::endl;

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

int LineMatcher::SearchByProjection(ORB_SLAM2::Frame &CurrentFrame, ORB_SLAM2::KeyFrame *RefFrame) {
    // TODO 与参考关键帧的匹配没有做单元测试，不过应该和上一帧匹配差不多
    // 与参考关键帧匹配，匹配步骤和与上一帧匹配一样
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 获取当前帧位姿
    cv::Mat Tcw_current = CurrentFrame.mTcw;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw(0, 0) = Tcw_current.at<float>(0, 0);
    Tcw(0, 1) = Tcw_current.at<float>(0, 1);
    Tcw(0, 2) = Tcw_current.at<float>(0, 2);
    Tcw(0, 3) = Tcw_current.at<float>(0, 3);

    Tcw(1, 0) = Tcw_current.at<float>(1, 0);
    Tcw(1, 1) = Tcw_current.at<float>(1, 1);
    Tcw(1, 2) = Tcw_current.at<float>(1, 2);
    Tcw(1, 3) = Tcw_current.at<float>(1, 3);

    Tcw(2, 0) = Tcw_current.at<float>(2, 0);
    Tcw(2, 1) = Tcw_current.at<float>(2, 1);
    Tcw(2, 2) = Tcw_current.at<float>(2, 2);
    Tcw(2, 3) = Tcw_current.at<float>(2, 3);

    std::vector<float> bounds = {CurrentFrame.mnMinX, CurrentFrame.mnMinY,
                                 CurrentFrame.mnMaxX, CurrentFrame.mnMaxY};

    // 存储新的裁剪后的KeyLine，后面用来匹配的，需要和mvpMapLines索引相同
    std::vector<KeyLine> new_KeyLines;
    // 这个用来记录新创建的KL，在mvpMapLines中的索引，用于后面给当前帧传递数据用
    std::vector<int> new_kl_index;
    cv::Mat new_descriptors;
    new_KeyLines.reserve(RefFrame->mvpMapLines.size());

    // 先遍历投影
    for (int i = 0; i < RefFrame->mvpMapLines.size(); i++) {
        // Step 1 对于前一帧的每一个MapLine，将端点投影到当前帧上
        MapLine* pML = RefFrame->mvpMapLines[i];
        if (!pML)
            continue;

        Eigen::Vector3d Xw_start = pML->mStart3d;
        Eigen::Vector3d Xw_end = pML->mEnd3d;
        // 计算相机坐标下的坐标值
        Eigen::Vector3d Xc_start = Tcw * Xw_start;
        Eigen::Vector3d Xc_end = Tcw * Xw_end;

        // Step 2 投影之后分三种情况
        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
            // Step 2.1 两端点都在相机后方，跳过
            continue;
        }
        if (Xc_start[2] < 0.0 || Xc_end[2] < 0.0) {
            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
            // 然后把前面那个端点留下
            // 先求交点
            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);

            Eigen::Vector3d P_c_cross(x_c_cross, y_c_cross, 0); // 交点

            Eigen::Vector4d line_proj;

            if (Xc_start[2] < 0.0) {
                // 如果是起点在后面，则把原来的起点替换成交点
//                pML->SetStartPos(P_c_cross);
                float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
                float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(x_c_cross, y_c_cross, u_end, v_end);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = RefFrame->mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);
                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            } else if (Xc_end[2] < 0.0) {
                // 如果是终点在后面，则把原来的终点替换成交点
//                pML->SetEndPos(P_c_cross);
                float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
                float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;
                line_proj = Eigen::Vector4d(u_start, v_start, x_c_cross, y_c_cross);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine kl_to_change = RefFrame->mvKeyLinesUn[i];
                    UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                    new_KeyLines.push_back(kl_to_change);
                    new_kl_index.push_back(i);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            }
        }
        if (Xc_start[2] > 0.0 && Xc_end[2] > 0.0) {
            // Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
            // 保留图像边界内的部分。Liang-Barsky线段裁剪算法
            float u_start = CurrentFrame.fx * Xc_start[0] / Xc_start[2] + CurrentFrame.cx;
            float v_start = CurrentFrame.fy * Xc_start[1] / Xc_start[2] + CurrentFrame.cy;

            float u_end = CurrentFrame.fx * Xc_end[0] / Xc_end[2] + CurrentFrame.cx;
            float v_end = CurrentFrame.fy * Xc_end[1] / Xc_end[2] + CurrentFrame.cy;

            Eigen::Vector4d line_proj(u_start, v_start, u_end, v_end);

            // 下面这套可以照搬
            Eigen::Vector4d new_line_proj;
            bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
            if (ok) {
                // 线段成功裁剪
                // 修改原来的KeyLine
                KeyLine kl_to_change = RefFrame->mvKeyLinesUn[i];
                UpdateKeyLineData(new_line_proj, kl_to_change, CurrentFrame.im_gray_);

                new_KeyLines.push_back(kl_to_change);
                new_kl_index.push_back(i);
                new_descriptors.push_back(pML->mLineDescriptor.clone());
            } else {
                continue;
            }
        }
    }

    // 全部投影完后再匹配
    int line_nmatches = 0;
    bool match_ok;

    for (int j = 0; j < CurrentFrame.NL; j++) {
        // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
        if(CurrentFrame.mvpMapLines[j])
            if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                continue;
        for (int i = 0; i < new_KeyLines.size(); i++) {
            // 输入两个KeyLine，输出是否匹配
            match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                    new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j));
            if (match_ok) {
                // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                MapLine* pML_from_last = RefFrame->mvpMapLines[new_kl_index[i]];
                CurrentFrame.mvpMapLines[j] = pML_from_last;
                line_nmatches++;
            } else {
                continue;
            }
        }
    }

    if (line_nmatches * 1.0 / CurrentFrame.NL < 0.2) {
        // 匹配数量不够，放大阈值再来
        std::vector<double> offset = {10.0, -0.1, -0.1, 5, 10};
        line_nmatches = 0;
        fill(CurrentFrame.mvpMapLines.begin(), CurrentFrame.mvpMapLines.end(), static_cast<ORB_SLAM2::MapLine*>(NULL));

        for (int j = 0; j < CurrentFrame.NL; j++) {
            // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
            if(CurrentFrame.mvpMapLines[j])
                if(CurrentFrame.mvpMapLines[j]->Observations()>0)
                    continue;
             for (int i = 0; i < new_KeyLines.size(); i++) {
                // 输入两个KeyLine，输出是否匹配
                match_ok = LineMatching(new_KeyLines[i], CurrentFrame.mvKeyLinesUn[j],
                                        new_descriptors.row(i), CurrentFrame.mLineDescriptors.row(j),
                                        offset);
                if (match_ok) {
                    // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                    MapLine* pML_from_last = RefFrame->mvpMapLines[new_kl_index[i]];
                    CurrentFrame.mvpMapLines[j] = pML_from_last;
                    line_nmatches++;
                } else {
                    continue;
                }
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Track reference keyframe time cost: " << ttrack * 1000 << " ms" << std::endl;
    std::cout << "CurrentFrame has matched " << line_nmatches << " MapLines from ReferenceFrame" << std::endl;

    return line_nmatches;
}



/**
 * @brief 当前帧和局部地图的线特征匹配
 * 方法和当前帧上一帧匹配差不多
 * TODO 使用别的方法匹配局部地图
 * 需要考虑线段被部分观测到的情况
 * Step 1 遍历有效局部MapLines，输入的局部MapLines未必全都能被观测或部分观测到
 *        在Tracking::SearchLocalLines中会判断观测情况并对
 *        pML->mbTrackInView赋值，true表示有被观测或部分观测到
 * Step 2 把局部MapLines投影到当前帧
 *        投影之后分三种情况
 *   Step 2.1 两端点都在相机后方，跳过，这个应该在上一步里就剔除了
 *   Step 2.2 一个在前，一个在后。求直线与图像平面的交点，以这个交点作为图像内线段的端点
 *            端点计算 Xik = Xsk + lambda * (Xsk - Xek)
 *            其中Xik[2] = 0，可以计算除lambda，从而计算除Xik[0], Xik[1]
 *            但端点也有可能跑到图像边界外面，应该也需要裁剪
 *   Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
 *            保留图像边界内的部分。Liang-Barsky线段裁剪算法
 * Step 3 成功投影之后，在投影线周围指定区域内搜索线特征
 *        点特征容易判断是否在区域内，线特征可能只有部分在内，应该可以判断某个端点在内即可
 *        以此来获取候选线特征
 *        // TODO step3没有做
 * Step 4 匹配候选线特征
 *        匹配策略：角度差、长度比值、重叠长度、描述子距离
 *        匹配数量较少时可以增大阈值
 *
 * @param[in] F                   当前帧
 * @param[in] vpMapLines          局部地图线，在执行这个函数前通过局部关键帧构建的
 * @param[in] th                  搜索范围
 * @return int                    成功匹配的数目
*/
int LineMatcher::SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 获取当前帧位姿，局部地图追踪是在初始位姿估计完成的情况下做的
    // 所以这里每一帧都有一个初始位姿
    cv::Mat Tcw_current = F.mTcw;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw(0, 0) = Tcw_current.at<float>(0, 0);
    Tcw(0, 1) = Tcw_current.at<float>(0, 1);
    Tcw(0, 2) = Tcw_current.at<float>(0, 2);
    Tcw(0, 3) = Tcw_current.at<float>(0, 3);

    Tcw(1, 0) = Tcw_current.at<float>(1, 0);
    Tcw(1, 1) = Tcw_current.at<float>(1, 1);
    Tcw(1, 2) = Tcw_current.at<float>(1, 2);
    Tcw(1, 3) = Tcw_current.at<float>(1, 3);

    Tcw(2, 0) = Tcw_current.at<float>(2, 0);
    Tcw(2, 1) = Tcw_current.at<float>(2, 1);
    Tcw(2, 2) = Tcw_current.at<float>(2, 2);
    Tcw(2, 3) = Tcw_current.at<float>(2, 3);

    std::vector<float> bounds = {F.mnMinX, F.mnMinY,
                                 F.mnMaxX, F.mnMaxY};

    // 存储局部地图投影到图像上的KeyLines，以及描述子和索引
    // 下面3个变量size相同，每个位置都是相对应的
    std::vector<KeyLine> new_KeyLines;
    cv::Mat new_descriptors;
    // 这个用来记录新创建的KL，在vpMapLines中的索引，用于后面给当前帧传递数据用
    std::vector<int> new_kl_index;
    new_KeyLines.reserve(F.mvpMapLines.size());

    // Step 1 遍历有效的局部MapLines，需要处理部分观测情况
    for(size_t iML = 0; iML < vpMapLines.size(); iML++) {
        MapLine* pML = vpMapLines[iML];
        // 不需要投影跳过
        if(!pML->mbTrackInView)
            continue;
        // 坏点跳过
        if(pML->isBad())
            continue;

        // 对于有效地图点，需要处理部分观测
        Eigen::Vector3d Xw_start = pML->mStart3d;
        Eigen::Vector3d Xw_end = pML->mEnd3d;
        // 计算相机坐标下的坐标值
        Eigen::Vector3d Xc_start = Tcw * Xw_start;
        Eigen::Vector3d Xc_end = Tcw * Xw_end;

        // Step 2 投影之后分三种情况
        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
            // Step 2.1 两端点都在相机后方，跳过
            // 这种情况在这里应该不会有，但保险起见还是判断一下
            continue;
        }
        if (Xc_start[2] < 0.0 || Xc_end[2] < 0.0) {
            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
            // 然后把前面那个端点留下
            // 先求交点
            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);

            Eigen::Vector3d P_c_cross(x_c_cross, y_c_cross, 0); // 交点

            Eigen::Vector4d line_proj;

            if (Xc_start[2] < 0.0) {
                // 如果是起点在后面，则把原来的起点替换成交点
//                pML->SetStartPos(P_c_cross);
                float u_end = F.fx * Xc_end[0] / Xc_end[2] + F.cx;
                float v_end = F.fy * Xc_end[1] / Xc_end[2] + F.cy;
                line_proj = Eigen::Vector4d(x_c_cross, y_c_cross, u_end, v_end);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    KeyLine proj_kl;
                    UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                    new_KeyLines.push_back(proj_kl);
                    new_kl_index.push_back(iML);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            } else if (Xc_end[2] < 0.0) {
                // 如果是终点在后面，则把原来的终点替换成交点
//                pML->SetEndPos(P_c_cross);
                float u_start = F.fx * Xc_start[0] / Xc_start[2] + F.cx;
                float v_start = F.fy * Xc_start[1] / Xc_start[2] + F.cy;
                line_proj = Eigen::Vector4d(u_start, v_start, x_c_cross, y_c_cross);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine proj_kl;
                    UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                    new_KeyLines.push_back(proj_kl);
                    new_kl_index.push_back(iML);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            }
        }
        if (Xc_start[2] > 0.0 && Xc_end[2] > 0.0) {
            // Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
            // 保留图像边界内的部分。Liang-Barsky线段裁剪算法
            float u_start = F.fx * Xc_start[0] / Xc_start[2] + F.cx;
            float v_start = F.fy * Xc_start[1] / Xc_start[2] + F.cy;

            float u_end = F.fx * Xc_end[0] / Xc_end[2] + F.cx;
            float v_end = F.fy * Xc_end[1] / Xc_end[2] + F.cy;

            Eigen::Vector4d line_proj(u_start, v_start, u_end, v_end);

            // 下面这套可以照搬
            Eigen::Vector4d new_line_proj;
            bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
            if (ok) {
                // 线段成功裁剪
                // 这里新建一个kl
                KeyLine proj_kl;
                UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                new_KeyLines.push_back(proj_kl);
                new_kl_index.push_back(iML);
                new_descriptors.push_back(pML->mLineDescriptor.clone());
            } else {
                continue;
            }
        }
    }

    // 全部投影完后再匹配
    int line_nmatches = 0;
    bool match_ok;
    for (int j = 0; j < F.NL; j++) {
        // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
        if(F.mvpMapLines[j])
            if(F.mvpMapLines[j]->Observations()>0)
                continue;
        for (int i = 0; i < new_KeyLines.size(); i++) {
            // 输入两个KeyLine，输出是否匹配
            match_ok = LineMatching(new_KeyLines[i], F.mvKeyLinesUn[j],
                                    new_descriptors.row(i), F.mLineDescriptors.row(j));
            if (match_ok) {
                // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                MapLine* pML_from_last = vpMapLines[new_kl_index[i]];
                F.mvpMapLines[j] = pML_from_last;
                line_nmatches++;
            } else {
                continue;
            }
        }
    }

    if (line_nmatches * 1.0 / F.NL < 0.2) {
        // 匹配数量不够，放大阈值再来
        std::vector<double> offset = {10.0, -0.1, -0.1, 5, 10};
        line_nmatches = 0;
        fill(F.mvpMapLines.begin(), F.mvpMapLines.end(), static_cast<ORB_SLAM2::MapLine*>(NULL));

        for (int j = 0; j < F.NL; j++) {
            // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
            if(F.mvpMapLines[j])
                if(F.mvpMapLines[j]->Observations()>0)
                    continue;
            for (int i = 0; i < new_KeyLines.size(); i++) {
                // 输入两个KeyLine，输出是否匹配
                match_ok = LineMatching(new_KeyLines[i], F.mvKeyLinesUn[j],
                                        new_descriptors.row(i), F.mLineDescriptors.row(j),
                                        offset);
                if (match_ok) {
                    // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                    MapLine* pML_from_last = vpMapLines[new_kl_index[i]];
                    F.mvpMapLines[j] = pML_from_last;
                    line_nmatches++;
                } else {
                    continue;
                }
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Match local map costs " << ttrack * 1000 << " ms" << std::endl;
    std::cout << "Local map has matched " << line_nmatches << " MapLines" << std::endl;

    return line_nmatches;
}

int LineMatcher::SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, std::vector<KeyLine>& new_kls, std::vector<std::pair<int, int>>& match_indices) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 获取当前帧位姿，局部地图追踪是在初始位姿估计完成的情况下做的
    // 所以这里每一帧都有一个初始位姿
    cv::Mat Tcw_current = F.mTcw;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw(0, 0) = Tcw_current.at<float>(0, 0);
    Tcw(0, 1) = Tcw_current.at<float>(0, 1);
    Tcw(0, 2) = Tcw_current.at<float>(0, 2);
    Tcw(0, 3) = Tcw_current.at<float>(0, 3);

    Tcw(1, 0) = Tcw_current.at<float>(1, 0);
    Tcw(1, 1) = Tcw_current.at<float>(1, 1);
    Tcw(1, 2) = Tcw_current.at<float>(1, 2);
    Tcw(1, 3) = Tcw_current.at<float>(1, 3);

    Tcw(2, 0) = Tcw_current.at<float>(2, 0);
    Tcw(2, 1) = Tcw_current.at<float>(2, 1);
    Tcw(2, 2) = Tcw_current.at<float>(2, 2);
    Tcw(2, 3) = Tcw_current.at<float>(2, 3);

    std::vector<float> bounds = {F.mnMinX, F.mnMinY,
                                 F.mnMaxX, F.mnMaxY};

    // 存储局部地图投影到图像上的KeyLines，以及描述子和索引
    // 下面3个变量size相同，每个位置都是相对应的
    std::vector<KeyLine> new_KeyLines;
    cv::Mat new_descriptors;
    // 这个用来记录新创建的KL，在vpMapLines中的索引，用于后面给当前帧传递数据用
    std::vector<int> new_kl_index;
    new_KeyLines.reserve(F.mvpMapLines.size());

    // Step 1 遍历有效的局部MapLines，需要处理部分观测情况
    for(size_t iML = 0; iML < vpMapLines.size(); iML++) {
        MapLine* pML = vpMapLines[iML];
        // 不需要投影跳过
        if(!pML->mbTrackInView)
            continue;
        // 坏点跳过
        if(pML->isBad())
            continue;

        // 对于有效地图点，需要处理部分观测
        Eigen::Vector3d Xw_start = pML->mStart3d;
        Eigen::Vector3d Xw_end = pML->mEnd3d;
        // 计算相机坐标下的坐标值
        Eigen::Vector3d Xc_start = Tcw * Xw_start;
        Eigen::Vector3d Xc_end = Tcw * Xw_end;

        // Step 2 投影之后分三种情况
        if (Xc_start[2] < 0 && Xc_end[2] < 0) {
            // Step 2.1 两端点都在相机后方，跳过
            // 这种情况在这里应该不会有，但保险起见还是判断一下
            continue;
        }
        if (Xc_start[2] < 0.0 || Xc_end[2] < 0.0) {
            // Step 2.2 一个在前，一个在后。求直线与图像平面的交点
            // 然后把前面那个端点留下
            // 先求交点
            double lambda = -1.0 * Xc_start[2] / (Xc_start[2] - Xc_end[2]);
            double x_c_cross = Xc_start[0] + lambda * (Xc_start[0] - Xc_end[0]);
            double y_c_cross = Xc_start[1] + lambda * (Xc_start[1] - Xc_end[1]);

            Eigen::Vector3d P_c_cross(x_c_cross, y_c_cross, 0); // 交点

            Eigen::Vector4d line_proj;

            if (Xc_start[2] < 0.0) {
                // 如果是起点在后面，则把原来的起点替换成交点
//                pML->SetStartPos(P_c_cross);
                float u_end = F.fx * Xc_end[0] / Xc_end[2] + F.cx;
                float v_end = F.fy * Xc_end[1] / Xc_end[2] + F.cy;
                line_proj = Eigen::Vector4d(x_c_cross, y_c_cross, u_end, v_end);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    KeyLine proj_kl;
                    UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                    new_KeyLines.push_back(proj_kl);

                    new_kls.push_back(proj_kl);

                    new_kl_index.push_back(iML);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            } else if (Xc_end[2] < 0.0) {
                // 如果是终点在后面，则把原来的终点替换成交点
//                pML->SetEndPos(P_c_cross);
                float u_start = F.fx * Xc_start[0] / Xc_start[2] + F.cx;
                float v_start = F.fy * Xc_start[1] / Xc_start[2] + F.cy;
                line_proj = Eigen::Vector4d(u_start, v_start, x_c_cross, y_c_cross);

                Eigen::Vector4d new_line_proj;
                bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
                if (ok) {
                    // 线段成功裁剪
                    // 修改原来的KeyLine
                    KeyLine proj_kl;
                    UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                    new_KeyLines.push_back(proj_kl);

                    new_kls.push_back(proj_kl);

                    new_kl_index.push_back(iML);
                    new_descriptors.push_back(pML->mLineDescriptor.clone());
                } else {
                    continue;
                }
            }
        }
        if (Xc_start[2] > 0.0 && Xc_end[2] > 0.0) {
            // Step 2.3 两个端点都在前方。这时候未必两个端点都在图像边界内，将直线裁剪，
            // 保留图像边界内的部分。Liang-Barsky线段裁剪算法
            float u_start = F.fx * Xc_start[0] / Xc_start[2] + F.cx;
            float v_start = F.fy * Xc_start[1] / Xc_start[2] + F.cy;

            float u_end = F.fx * Xc_end[0] / Xc_end[2] + F.cx;
            float v_end = F.fy * Xc_end[1] / Xc_end[2] + F.cy;

            Eigen::Vector4d line_proj(u_start, v_start, u_end, v_end);

            // 下面这套可以照搬
            Eigen::Vector4d new_line_proj;
            bool ok = LiangBarsky(line_proj, new_line_proj, bounds);
            if (ok) {
                // 线段成功裁剪
                // 这里新建一个kl
                KeyLine proj_kl;
                UpdateKeyLineData(new_line_proj, proj_kl, F.im_gray_);

                new_KeyLines.push_back(proj_kl);

                new_kls.push_back(proj_kl);

                new_kl_index.push_back(iML);
                new_descriptors.push_back(pML->mLineDescriptor.clone());
            } else {
                continue;
            }
        }
    }

    std::cout << "New KeyLines size = " << new_KeyLines.size() << std::endl;
    std::cout << "new_kl_index size = " << new_kl_index.size() << std::endl;


    // 全部投影完后再匹配
    int line_nmatches = 0;
    bool match_ok;
    for (int j = 0; j < F.NL; j++) {
        // 如果该KeyLine已经有MapLine了，且有帧观测到它，跳过
        if(F.mvpMapLines[j])
            if(F.mvpMapLines[j]->Observations()>0)
                continue;
        for (int i = 0; i < new_KeyLines.size(); i++) {
            // 输入两个KeyLine，输出是否匹配
            match_ok = LineMatching(new_KeyLines[i], F.mvKeyLinesUn[j],
                                    new_descriptors.row(i), F.mLineDescriptors.row(j));
            if (match_ok) {
                // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                MapLine* pML_from_last = vpMapLines[new_kl_index[i]];
                F.mvpMapLines[j] = pML_from_last;

                match_indices.push_back(std::pair<int, int>(i, j));

                line_nmatches++;
            } else {
                continue;
            }
        }
    }

    if (line_nmatches <= 0.2 * F.NL) {
        // 匹配数量不够，放大阈值再来
        std::vector<double> offset = {10.0, -0.1, -0.1, 5, 10};
        line_nmatches = 0;
        fill(F.mvpMapLines.begin(), F.mvpMapLines.end(), static_cast<ORB_SLAM2::MapLine*>(NULL));
        match_indices.clear();

        for (int j = 0; j < F.NL; j++) {
            if(F.mvpMapLines[j])
                if(F.mvpMapLines[j]->Observations()>0)
                    continue;
            for (int i = 0; i < new_KeyLines.size(); i++) {
                match_ok = LineMatching(new_KeyLines[i], F.mvKeyLinesUn[j],
                                        new_descriptors.row(i), F.mLineDescriptors.row(j),
                                        offset);
                if (match_ok) {
                    // 匹配成功，根据new_kl_index中的索引，对CurrentFrame.mvpMapLines赋值
                    MapLine* pML_from_last = vpMapLines[new_kl_index[i]];
                    F.mvpMapLines[j] = pML_from_last;

                    match_indices.push_back(std::pair<int, int>(i, j));

                    line_nmatches++;
                } else {
                    continue;
                }
            }
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    std::cout << "Match local map costs " << ttrack * 1000 << " ms" << std::endl;
    std::cout << "Local map has matched " << line_nmatches << " MapLines" << std::endl;

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

/**
 * @brief Liang-Barsky线段裁剪算法，获取一条线段被一个边界所截取的部分
 * 这里用来处理空间线段的部分观测问题，空间线段投影到图像上之后，将图像内的部分裁剪出来
 * @param[in] line        空间线段起点终点投影到图像上的坐标
 * @param[in&out] new_line    新的被裁剪的坐标
 * @param[in] bounds      图像边界
 * @return  这条空间线段投影是否在图像内，true是，false否
 */
bool LineMatcher::LiangBarsky(Eigen::Vector4d& line, Eigen::Vector4d& new_line, std::vector<float>& bounds) {
    Eigen::Vector2d start(line(0), line(1));
    Eigen::Vector2d end(line(2), line(3));

    // 先计算四个p和四个q
    double p[4];
    double q[4];
    p[0] = start(0) - end(0);
    p[1] = end(0) - start(0);
    p[2] = start(1) - end(1);
    p[3] = end(1) - start(1);

    q[0] = start(0) - bounds[0];
    q[1] = bounds[2] - start(0);
    q[2] = start(1) - bounds[1];
    q[3] = bounds[3] - start(1);

    if (p[0] == 0) {
        // 表示线是竖直的
        if (q[0] <= 0 || q[2] <= 0) {
            // 在边界外，或边界上
            return false;
        }
    }
    if (p[2] == 0) {
        // 表示线是水平的
        if (q[2] >= 0 || q[3] >= 0) {
            return false;
        }
    }

    // 计算u值
    double u[4];
    for (int i = 0; i < 4; i++) {
        u[i] = q[i] / p[i];
    }
    // 在 p < 0 里面找出最大的u
    // 在 p > 0 里面找出最小的u
    double u_min = 0, u_max = 1;
    for (int i = 0; i < 4; i++) {
        if (p[i] < 0)
        {
            if (u_min < u[i])
                u_min = u[i];
        }
        else
        {
            if (u_max > u[i])
                u_max = u[i];
        }
    }

    // TODO 大部分的线都在图像内，当然这里也不是没有作用
    if (u_max >= u_min) {
        // 计算新的端点
        double new_x1, new_x2, new_y1, new_y2;
        new_x1 = start(0) + round(u_min * (end(0) - start(0)));
        new_line(0) = new_x1;

        new_y1 = start(1) + round(u_min * (end(1) - start(1)));
        new_line(1) = (new_y1);

        new_x2 = start(0) + round(u_max * (end(0) - start(0)));
        new_line(2) = (new_x2);

        new_y2 = start(1) + round(u_max * (end(1) - start(1)));
        new_line(3) = (new_y2);
        return true;
    } else {
        return false;
    }
}

// 输入两个KeyLine，输出是否匹配
bool LineMatcher::LineMatching(KeyLine& kl1, KeyLine& kl2, const cv::Mat& desc1, const cv::Mat& desc2,
                               const std::vector<double>& offset) {
    // TODO 阈值选的不好，总是匹配不成功
    double angle_offset = offset[0];
    double length_offset = offset[1];
    double overlap_offset = offset[2];
    double desc_offset = offset[3];
    double perpend_dist_offset = offset[4];

    // TODO 我怎么感觉可以考虑先比较哪个，比如先比较距离和角度，直接在相对位置方面筛选一部分
    // 先比较描述子，外观一致性检测，主要用描述子的汉明距离描述
    if (DescriptorDistance(desc1, desc2) > desc_dist_threshold_ + desc_offset) {
        return false;
    }

    // 下面是几何一致性检测，主要是角度差、长度比值、重叠长度
    if (abs(kl1.angle - kl2.angle) > angle_threshold_ + angle_offset * M_PI / 180.0) {
        return false;
    }

//    if (PerpendicularDistance(kl1, kl2) > perpend_dist_threshold_ + perpend_dist_offset) {
//        return false;
//    }
    // 垂直距离换成重投影距离公式，即起点和终点到另一条线段的距离

    if (min(kl1.lineLength, kl2.lineLength) / max(kl1.lineLength, kl2.lineLength) < length_threshold_ + length_offset) {
        return false;
    }

    double th = overlap_threshold_ + overlap_offset;
    if (! LineOverLap(kl1, kl2, th)) {
        return false;
    }

    // 下面这个作为误匹配的判断，主要因为调试的时候发现有些距离很远的线都被匹配上了
    // TODO 线特征误匹配剔除
    if (ReprojectionError(kl1, kl2) > reproj_error_threshold_) {
        return false;
    }

    return true;
}

// 计算两个KeyLine的重叠比例
// 重叠满足输出true，否则false
bool LineMatcher::LineOverLap(KeyLine& kl1, KeyLine& kl2, double& threshold) {
    // 重叠区域比例
    // 投影到x轴和y轴上，比较轴上的重叠区域与较短长度的比值
    double d1_x = abs(kl1.startPointX - kl1.endPointX);
    double d2_x = abs(kl2.startPointX - kl2.endPointX);
    double min_x = min(min(kl1.startPointX, kl1.endPointX), min(kl2.startPointX, kl2.endPointX));
    double max_x = max(max(kl1.startPointX, kl1.endPointX), max(kl2.startPointX, kl2.endPointX));

    double d1_y = abs(kl1.startPointY - kl1.endPointY);
    double d2_y = abs(kl2.startPointY - kl2.endPointY);
    double min_y = min(min(kl1.startPointY, kl1.endPointY), min(kl2.startPointY, kl2.endPointY));
    double max_y = max(max(kl1.startPointY, kl1.endPointY), max(kl2.startPointY, kl2.endPointY));

    // TODO 如果线段竖直或水平，那就只能比较y或x方向，如果是一般情况，那只要某一个方向满足要求几何
    // 如果有一线段是竖直的，就只看y投影，如果水平，就看x投影
    if (d1_x == 0 || d2_x == 0) {
        // 比较y投影，这个时候y投影重叠部分确实可以有效判断
        if ((d1_y + d2_y - max_y + min_y) / min(d1_y, d2_y) >= threshold) {
            return true;
        }
    }
    if (d1_y == 0 || d2_y == 0) {
        // 比较x投影，同样，这个时候y投影重叠部分确实可以有效判断
        if ((d1_x + d2_x - max_x + min_x) / min(d1_x, d2_x) >= threshold) {
            return true;
        }
    }
    // 大多数情况都是下面的
    // x轴没有重叠，y轴有可能重叠，反过来也一样
    // 如果某一个方向上重叠了，另一个方向就不用判断了
    // TODO 但如果两条线在x轴上重叠，但y轴相距很远怎么算？所以不重叠的时候要限制非重叠长度不能太大
    if ((d1_x + d2_x - max_x + min_x) / min(d1_x, d2_x) >= threshold) {
        // 如果x方向上重叠超过50%，且y方向上也重叠或者非重叠区域较小，则认为这两条线段重叠
        if (d1_y + d2_y + min_y >= max_y) {
            // y向重叠
            return true;
        } else {
            // y向不重叠，但非重叠区域小
            if (max_y - min_y - d1_y - d2_y < 0.3 * min(d1_y, d2_y)) {
                return true;
            }
        }
    } else if ((d1_x + d2_x - max_x + min_x) / min(d1_x, d2_x) < threshold
                && (max_x - min_x - d1_x - d2_x) < 0.3 * min(d1_x, d2_x)) {
        // 如果x方向上重叠少，则比较y，也重叠超过50%则认为两条线重叠
        if ((d1_y + d2_y - max_y + min_y) / min(d1_y, d2_y) >= threshold) {
            return true;
        }
    }

    return false;
}

double LineMatcher::PerpendicularDistance(cv::line_descriptor::KeyLine &line1, cv::line_descriptor::KeyLine &line2) {
    Eigen::Vector3d line1_start(line1.startPointX, line1.startPointY, 1);
    Eigen::Vector3d line1_end(line1.endPointX, line1.endPointY, 1);

    Eigen::Vector3d line2_start(line2.startPointX, line2.startPointY, 1);
    Eigen::Vector3d line2_end(line2.endPointX, line2.endPointY, 1);

    Eigen::Vector3d line1_coefficient = line1_start.cross(line1_end);

    double dist_line2_start = line2_start.dot(line1_coefficient) /
            sqrt(pow(line1_coefficient[0], 2) + pow(line1_coefficient[1], 2));

    double dist_line2_end = line2_end.dot(line1_coefficient) /
                          sqrt(pow(line1_coefficient[0], 2) + pow(line1_coefficient[1], 2));

    return min(dist_line2_start, dist_line2_end);
}

double LineMatcher::ReprojectionError(KeyLine &line1, KeyLine &line2) {
    Eigen::Vector3d line1_start(line1.startPointX, line1.startPointY, 1);
    Eigen::Vector3d line1_end(line1.endPointX, line1.endPointY, 1);

    Eigen::Vector3d line2_start(line2.startPointX, line2.startPointY, 1);
    Eigen::Vector3d line2_end(line2.endPointX, line2.endPointY, 1);

    Eigen::Vector3d line1_coefficient = line1_start.cross(line1_end);

    double dist_line2_start = line2_start.dot(line1_coefficient) /
                              sqrt(pow(line1_coefficient[0], 2) + pow(line1_coefficient[1], 2));

    double dist_line2_end = line2_end.dot(line1_coefficient) /
                            sqrt(pow(line1_coefficient[0], 2) + pow(line1_coefficient[1], 2));
    Eigen::Vector2d dist_21(dist_line2_start, dist_line2_end);

    return dist_21.norm();
}


// new_line中包含的是新的KL的起点和终点坐标
// im就是原图
void LineMatcher::UpdateKeyLineData(Eigen::Vector4d& new_line, KeyLine& old_line, cv::Mat& im) {
        old_line.startPointX = new_line[0];
        old_line.startPointY = new_line[1];
        old_line.endPointX = new_line[2];
        old_line.endPointY = new_line[3];
        old_line.sPointInOctaveX = new_line[0];
        old_line.sPointInOctaveY = new_line[1];
        old_line.ePointInOctaveX = new_line[2];
        old_line.ePointInOctaveY = new_line[3];
        // 修改中点坐标
        old_line.pt = cv::Point2f((old_line.endPointX + old_line.startPointX) / 2, (old_line.endPointY + old_line.startPointY) / 2);
        // 修改长度和像素数
        old_line.lineLength = float(sqrt(pow(old_line.startPointX - old_line.endPointX, 2) + pow(old_line.startPointY - old_line.endPointY, 2)));
        cv::LineIterator li(im,
                            cv::Point2f(old_line.startPointX, old_line.startPointY),
                            cv::Point2f(old_line.endPointX, old_line.endPointY));
        old_line.numOfPixels = li.count;
        // 修改角度
        old_line.angle = atan2((old_line.endPointY - old_line.startPointY), (old_line.endPointX - old_line.startPointX));
        // 修改最小包含区域
        old_line.size = (old_line.endPointX - old_line.startPointX) * (old_line.endPointY - old_line.startPointY);
        // 修改强度
        old_line.response = old_line.lineLength / max(im.cols, im.rows);
}


} // namespace ORB_SLAM2