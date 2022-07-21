/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "KeyFrame.h"
#include "Converter.h"

#include <condition_variable>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/features/normal_3d.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/compression/compression_profiles.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>


namespace ORB_SLAM2 {
class PointCloudMapping {
public:
    PointCloudMapping();

    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void shutdown();
    void Run();
    void getGlobalCloudMap(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &outputMap);
    void reset();
    void SavePcdFile(const std::string& filename);

protected:
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr generatePointCloud(cv::Mat& color, cv::Mat& depth);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr globalMap;
    shared_ptr<thread> viewerThread;
    
    bool shutDownFlag = false;
    mutex shutDownMutex;

    condition_variable keyFrameUpdated;
    mutex keyFrameUpdateMutex;
    
    // data to generate point clouds
    vector<KeyFrame*> keyframes;
    vector<cv::Mat> colorImgs, depthImgs;
    cv::Mat   depthImg,colorImg,mpose;

    vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> mvPointClouds;
    vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> mvPointCloudsForMatch;
    vector<cv::Mat> mvPosePointClouds;
    unsigned long int mpointcloudID=0;
    
    mutex keyframeMutex;
    uint16_t lastKeyframeSize =0;
    
    double resolution;
    pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
    pcl::VoxelGrid<pcl::PointXYZRGBA> voxelForMatch;
    float cx = 0;
    float cy = 0;
    float fx = 0;
    float fy = 0;

    void computeTranForTwoPiontCloud(pcl::PointCloud<pcl::PointXYZRGBA> ::Ptr &P1,
                                     pcl::PointCloud<pcl::PointXYZRGBA> ::Ptr &P2, Eigen::Isometry3d&  T );

};
}
#endif // POINTCLOUDMAPPING_H
