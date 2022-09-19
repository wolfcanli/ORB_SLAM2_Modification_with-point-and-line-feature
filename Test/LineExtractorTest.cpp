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

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    ORB_SLAM2::LineExtractor* mpLineExtractor;
    ORB_SLAM2::Frame mCurrentFrame;

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

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++) {
//        std::cout << "Current loop id " << ni << std::endl;

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

        cv::Mat outImg;
        cv::line_descriptor::drawKeylines(imRGB, mCurrentFrame.mvKeyLines, outImg, cv::Scalar(0, 255, 0));
        cv::imshow("image with lines", outImg);
        cv::waitKey(0);
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
