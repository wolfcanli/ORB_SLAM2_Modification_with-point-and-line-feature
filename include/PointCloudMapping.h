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
#include <list>

#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>


namespace ORB_SLAM2 {

class PointCloudMapping {
public:
    PointCloudMapping();
    ~PointCloudMapping();

    void Run();

    void InsertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    void SetGlobalUpdateFlag();

    void RequestFinish();

    void Reset();
    void SavePcdFile(const std::string& filename);

public:
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr GeneratePointCloud(cv::Mat& color, cv::Mat& depth);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr GeneratePointCloud(cv::Mat& color, cv::Mat& depth, cv::Mat& pose);
    void GeneratePointCloud2(cv::Mat& color, cv::Mat& depth, cv::Mat& pose);

    bool finish_flag_;

    bool is_loop_;
    std::mutex loop_mutex_;

    std::vector<KeyFrame*> keyframes_vector_;
    std::vector<cv::Mat> color_imgs_vector_, depth_imgs_vector_;

    std::list<KeyFrame*> keyframes_;
    std::list<cv::Mat> color_imgs_, depth_imgs_;
    std::list<KeyFrame*> keyframes_seg_;
    std::list<cv::Mat> color_imgs_seg_, depth_imgs_seg_;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr global_map_; // 普通点云

    std::mutex keyframe_mutex_;
    std::mutex finish_mutex_;
    std::mutex point_cloud_mutex_;

    condition_variable keyframe_update_condi_var_;

    std::shared_ptr<std::thread> point_cloud_thread_;

    float cx = 0;
    float cy = 0;
    float fx = 0;
    float fy = 0;
};
}
#endif // POINTCLOUDMAPPING_H
