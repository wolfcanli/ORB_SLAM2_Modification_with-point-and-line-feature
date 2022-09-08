//
// Created by jiajieshi on 22-9-5.
//

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include<opencv2/core/core.hpp>
#include <opencv2/line_descriptor.hpp>

#include "System.h"
#include "LineExtractor.h"


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

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++) {
        mvKeyLines.clear();
        mLineDescriptors = cv::Mat();
        mvKeyLineCoefficient.clear();
//        std::cout << "Current loop id " << ni << std::endl;

        // Read image and depthmap from file
        imRGB = cv::imread(string(dataset_dir)+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_GRAYSCALE);
        imD = cv::imread(string(dataset_dir)+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(dataset_dir) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
        mpLineExtractor->ExtractLineSegment(imRGB,
                                            mvKeyLines,
                                            mLineDescriptors,
                                            mvKeyLineCoefficient);
        std::cout << "Keylines size = " << mvKeyLines.size() << std::endl;

        cv::Mat outImg;
        cv::line_descriptor::drawKeylines(imRGB, mvKeyLines, outImg, cv::Scalar::all( -1 ));
        cv::imshow("image with lines", outImg);
        cv::waitKey(10);
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
