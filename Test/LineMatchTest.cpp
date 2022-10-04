//
// Created by jiajieshi on 22-9-5.
//

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <memory>

#include<opencv2/core/core.hpp>
#include <opencv2/line_descriptor.hpp>

#include "System.h"
#include "LineExtractor.h"
#include "Frame.h"
#include "MapLine.h"
#include "Map.h"
#include "LineMatcher.h"

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main()
{
    std::vector<cv::line_descriptor::KeyLine> mvKeyLines;
    std::vector<cv::line_descriptor::KeyLine> mvKeyLinesUn;
    std::vector<float> mvDepthLineStart; // 特征线起点坐标深度
    std::vector<float> mvDepthLineEnd; // 特征线终点坐标深度

    cv::Mat mLineDescriptors;
    std::vector<Eigen::Vector3d> mvKeyLineCoefficient; // 特征线直线系数
    std::vector<bool> mvbLineOutlier;

    std::string setting_dir = "../Examples/RGB-D/dataset.yaml";

    cv::FileStorage fSettings(setting_dir, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << setting_dir << endl;
        exit(-1);
    }
    std::string tum_yaml_dir;
    std::string dataset_dir;

    fSettings["DatasetDir"] >> dataset_dir;
    std::string associate_txt_dir = dataset_dir + "/associate.txt";

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(associate_txt_dir);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    ifstream fin("../results/CameraTrajectory.txt");
    if (!fin) {
        cerr << "请在有groundtruth.txt的目录下运行此程序" << endl;
        return 1;
    }
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    while (!fin.eof()) {
        double data[8] = {0};
        for (auto &d:data)
            fin >> d;

        Eigen::Isometry3d pose(Eigen::Quaterniond(data[7], data[4], data[5], data[6]));
        pose.pretranslate(Eigen::Vector3d(data[1], data[2], data[3]));

        poses.push_back(pose);
    }

    ifstream fin2("../Examples/RGB-D/associate_with_groundtruth.txt");
    if (!fin2) {
        cerr << "请在有groundtruth.txt的目录下运行此程序" << endl;
        return 1;
    }
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> gt_poses;
    while(!fin2.eof()) {
        string s;
        getline(fin2,s);
        if(!s.empty()) {
            double data[7] = {0};
            stringstream ss;
            ss << s;
            double t, x, y, z, qx, qy, qz, qw;
            string sRGB, sD;
            ss >> t;
            ss >> sRGB;
            ss >> t;
            ss >> sD;
            ss >> t;
            ss >> x;
            data[0] = x;
            ss >> y;
            data[1] = y;
            ss >> z;
            data[2] = z;
            ss >> qx;
            data[3] = qx;
            ss >> qy;
            data[4] = qy;
            ss >> qz;
            data[5] = qz;
            ss >> qw;
            data[6] = qw;

            Eigen::Isometry3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]));
            pose.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));

            gt_poses.push_back(pose);


        }
    }


    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    ORB_SLAM2::LineExtractor* mpLineExtractor;
    ORB_SLAM2::Frame mCurrentFrame, LastFrame;

    ORB_SLAM2::ORBVocabulary* mpVocabulary = new ORB_SLAM2::ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile("../Vocabulary/ORBvoc.txt");
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << "../Vocabulary/ORBvoc.txt" << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    ///相机的内参数矩阵
    cv::Mat mK;
    ///相机的去畸变参数
    cv::Mat mDistCoef;
    ///相机的基线长度 * 相机的焦距
    float mbf;
    float mThDepth;

    cv::FileStorage fSettings2("../Examples/RGB-D/TUM1.yaml", cv::FileStorage::READ);
    float fx = fSettings2["Camera.fx"];
    float fy = fSettings2["Camera.fy"];
    float cx = fSettings2["Camera.cx"];
    float cy = fSettings2["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings2["Camera.k1"];
    DistCoef.at<float>(1) = fSettings2["Camera.k2"];
    DistCoef.at<float>(2) = fSettings2["Camera.p1"];
    DistCoef.at<float>(3) = fSettings2["Camera.p2"];
    const float k3 = fSettings2["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings2["Camera.bf"];
    mThDepth = mbf*(float)fSettings2["ThDepth"]/fx;

    int nFeatures = fSettings2["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings2["ORBextractor.scaleFactor"];
    int nLevels = fSettings2["ORBextractor.nLevels"];
    int fIniThFAST = fSettings2["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings2["ORBextractor.minThFAST"];
    ORB_SLAM2::ORBextractor* mpORBextractorLeft;
    mpORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    ORB_SLAM2::Map* mpMap = new ORB_SLAM2::Map();
    ORB_SLAM2::LineMatcher line_matcher(0.9, true);

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++) {
        std::cout << "Current loop id " << ni << std::endl;

        // Read image and depthmap from file
        imRGB = cv::imread(string(dataset_dir)+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(dataset_dir)+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(dataset_dir) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        cv::Mat imGray;
        // 将RGB或RGBA转化成灰度图
        if(imRGB.channels() == 3) {
            cvtColor(imRGB, imGray, CV_RGB2GRAY);
        } else if(imRGB.channels() == 4) {
            cvtColor(imRGB, imGray, CV_RGBA2GRAY);
        }

        // 将深度相机的disparity转化成depth
        // 原本的深度图的像素值和真实的深度值（距离）成一个比例，这个比例是depth map factor，这里是5000
        // 所以下面就是将depth map除以5000，还原真实的深度
        imD.convertTo(imD, CV_32F, 1.0 / 5000.0);

        mCurrentFrame = ORB_SLAM2::Frame(imGray, imD, tframe, mpORBextractorLeft,mpVocabulary,mK,mDistCoef,mbf,mThDepth);
        std::cout << "Keylines size = " << mCurrentFrame.mvKeyLines.size() << std::endl;

//        Eigen::Isometry3d pose_current = poses[ni]; // estimated pose
        Eigen::Isometry3d pose_current = gt_poses[ni]; // groundtruth
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
//        std::cout << "Tcw = " << std::endl;
//        std::cout << Tcw << std::endl;

        Tcw.at<float>(0, 0) = pose_current.rotation()(0, 0);
//        std::cout << pose_current.rotation()(0, 0) << std::endl;
        Tcw.at<float>(0, 1) = pose_current.rotation()(0, 1);
        Tcw.at<float>(0, 2) = pose_current.rotation()(0, 2);

        Tcw.at<float>(1, 0) = pose_current.rotation()(1, 0);
//        std::cout << pose_current.rotation()(1, 0) << std::endl;
        Tcw.at<float>(1, 1) = pose_current.rotation()(1, 1);
        Tcw.at<float>(1, 2) = pose_current.rotation()(1, 2);

        Tcw.at<float>(2, 0) = pose_current.rotation()(2, 0);
        Tcw.at<float>(2, 1) = pose_current.rotation()(2, 1);
        Tcw.at<float>(2, 2) = pose_current.rotation()(2, 2);
//        std::cout << pose_current.rotation()(2, 2) << std::endl;

        Tcw.at<float>(0, 3) = pose_current.translation()(0);
        Tcw.at<float>(1, 3) = pose_current.translation()(1);
        Tcw.at<float>(2, 3) = pose_current.translation()(2);

        mCurrentFrame.SetPose(Tcw);

        if (ni == 0) {
            // 第一张跳过
            LastFrame = ORB_SLAM2::Frame(mCurrentFrame);
            continue;
        }

        // 第一张跳过
        LastFrame = ORB_SLAM2::Frame(mCurrentFrame);

        // 在当前图像里把当前图像检测出来的KeyLine（绿色）和上一帧投影裁剪的KeyLine（红色）绘制出来
        // 这里只画出匹配的线对
        std::vector<KeyLine> current_kls; // from mCurrentFrame.mvKeyLines
        std::vector<KeyLine> last_kls; // from mCurrentFrame.mvLastKeyLines
        std::vector<std::pair<int, int>> match_indices = mCurrentFrame.mvMatchIndex;
        // 通过当前帧储存的MapLine来找到上一帧的匹配
        for (int k = 0; k < match_indices.size(); k++) {
            std::pair<int, int> pair_index = match_indices[k];
            last_kls.push_back(mCurrentFrame.mvLastKeyLines[pair_index.first]);
            current_kls.push_back(mCurrentFrame.mvKeyLinesUn[pair_index.second]);
        }
        cv::Mat outImg = imRGB.clone();
        // 当前帧的
        cv::line_descriptor::drawKeylines(outImg, current_kls, outImg, cv::Scalar(0, 255, 0));
        std::cout << "current_kls size = " << current_kls.size() << std::endl;
        // Last Frame
        cv::line_descriptor::drawKeylines(outImg, last_kls, outImg, cv::Scalar(0, 0, 255));
        std::cout << "last_kls size = " << last_kls.size() << std::endl;
        // 在匹配的线对起点之间画一条线，用蓝色表示
        for (int l = 0; l < current_kls.size(); l++) {
            float last_x = last_kls[l].startPointX;
            float last_y = last_kls[l].startPointY;

            float curr_x = current_kls[l].startPointX;
            float curr_y = current_kls[l].startPointY;

            cv::line(outImg, cv::Point2i(last_x, last_y), cv::Point2i(curr_x, curr_y), cv::Scalar(255, 0, 0), 1);
        }


//        cv::Mat outImg = imRGB.clone();
//        // 当前帧的
//        cv::line_descriptor::drawKeylines(outImg, mCurrentFrame.mvKeyLinesUn, outImg, cv::Scalar(0, 255, 0));
//        std::cout << "mCurrentFrame.mvKeyLinesUn size = " << mCurrentFrame.mvKeyLinesUn.size() << std::endl;
//        // Last Frame
//        cv::line_descriptor::drawKeylines(outImg, mCurrentFrame.mvLastKeyLines, outImg, cv::Scalar(0, 0, 255));
//        std::cout << "mCurrentFrame.mvLastKeyLines size = " << mCurrentFrame.mvLastKeyLines.size() << std::endl;
//
//        cv::Mat outImg2 = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_8UC3);
//        // 当前帧的
//        cv::line_descriptor::drawKeylines(outImg2, mCurrentFrame.mvKeyLinesUn, outImg2, cv::Scalar(0, 255, 0));
//        std::cout << "mCurrentFrame.mvKeyLinesUn size = " << mCurrentFrame.mvKeyLinesUn.size() << std::endl;
//        // Last Frame
//        cv::line_descriptor::drawKeylines(outImg2, mCurrentFrame.mvLastKeyLines, outImg2, cv::Scalar(0, 0, 255));
//        std::cout << "mCurrentFrame.mvLastKeyLines size = " << mCurrentFrame.mvLastKeyLines.size() << std::endl;


        cv::imshow("image with lines", outImg);
//        cv::imshow("image with lines(no background)", outImg2);
        cv::waitKey(200);
    }

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
