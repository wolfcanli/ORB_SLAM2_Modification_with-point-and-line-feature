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

#include "PointCloudMapping.h"
 
namespace ORB_SLAM2 {
PointCloudMapping::PointCloudMapping(): finish_flag_(false), is_loop_(false) {
    global_map_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();

    point_cloud_thread_ = std::make_shared<std::thread>(&PointCloudMapping::Run, this);
}

PointCloudMapping::~PointCloudMapping() {
    point_cloud_thread_->join();
}

void PointCloudMapping::RequestFinish() {
    {
        std::unique_lock<mutex> lck(finish_mutex_);
        finish_flag_ = true;
    } // finish_mutex_
}

void PointCloudMapping::InsertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth) {
    unique_lock<mutex> lck(keyframe_mutex_);

    keyframes_.emplace_back(kf);
    color_imgs_.emplace_back(color.clone());
    depth_imgs_.emplace_back(depth.clone());

    if(cx == 0 || cy == 0 || fx == 0 || fy == 0) {
        cx = kf->cx;
        cy = kf->cy;
        fx = kf->fx;
        fy = kf->fy;
    }

    keyframes_vector_.emplace_back(kf);
    color_imgs_vector_.emplace_back(color);
    depth_imgs_vector_.emplace_back(depth);

    keyframe_update_condi_var_.notify_one();
    std::cout << "color_imgs size = " << color_imgs_.size() << std::endl;
    std::cout <<"receive a keyframe, id = "<< kf->mnId << std::endl;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloudMapping::GeneratePointCloud(cv::Mat& color, cv::Mat& depth) {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_camera(new pcl::PointCloud<pcl::PointXYZRGBA>());

    for (int m = 0; m < depth.rows; m += 3) {
        for (int n = 0; n < depth.cols; n += 3) {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 10)
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m - cy) * p.z / fy;

            p.b = color.ptr<uchar>(m)[n * 3];
            p.g = color.ptr<uchar>(m)[n * 3 + 1];
            p.r = color.ptr<uchar>(m)[n * 3 + 2];

            pc_camera->points.push_back(p);
        }
    }
    return pc_camera;
}


pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloudMapping::GeneratePointCloud(cv::Mat& color, cv::Mat& depth, cv::Mat& pose) {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_camera(new pcl::PointCloud<pcl::PointXYZRGBA>());

    for (int m = 0; m < depth.rows; m += 3) {
        for (int n = 0; n < depth.cols; n += 3) {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 10)
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m - cy) * p.z / fy;

            p.b = color.ptr<uchar>(m)[n * 3];
            p.g = color.ptr<uchar>(m)[n * 3 + 1];
            p.r = color.ptr<uchar>(m)[n * 3 + 2];

            pc_camera->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = Converter::toSE3Quat(pose);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc_world(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::transformPointCloud(*pc_camera, *pc_world, T.inverse().matrix());

    pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
    voxel.setLeafSize(0.01f, 0.01f, 0.01f);
    voxel.setInputCloud(pc_world);
    voxel.filter(*pc_camera);
    pc_world->is_dense = true;

    return pc_camera;
}

void PointCloudMapping::GeneratePointCloud2(cv::Mat& color, cv::Mat& depth, cv::Mat& pose) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcd_camera(new pcl::PointCloud<pcl::PointXYZRGBA>());
    for(size_t v = 1; v < color.rows ; v+=3){
        for(size_t u = 1; u < color.cols ; u+=3){
            float d = depth.ptr<float>(v)[u];
            if(d <0.01 || d>10){
                continue;
            }

            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( u - cx) * p.z / fx;
            p.y = ( v - cy) * p.z / fy;

            p.b = color.ptr<uchar>(v)[u*3];
            p.g = color.ptr<uchar>(v)[u*3+1];
            p.r = color.ptr<uchar>(v)[u*3+2];

            pcd_camera->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = Converter::toSE3Quat( pose );
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcd_world(new pcl::PointCloud<pcl::PointXYZRGBA>());
    // tmp为转换到世界坐标系下的点云
    pcl::transformPointCloud(*pcd_camera, *pcd_world, T.inverse().matrix());

    pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
    voxel.setLeafSize( 0.01, 0.01, 0.01);
    voxel.setInputCloud(pcd_world);
    voxel.filter(*pcd_camera);
    pcd_camera->is_dense = true;
    *global_map_ += *pcd_camera;

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "Time cost " << t << std::endl;
}

void PointCloudMapping::Run() {
    pcl::visualization::CloudViewer viewer("Dense Pointcloud Viewer");

    while(true) {
        KeyFrame* kf_;
        cv::Mat color_img_, depth_img_;
        if (is_loop_) {
            // 清空地图并重建
            global_map_->clear();
            for (int i = 0; i < keyframes_vector_.size(); i++) {
                cv::Mat pose = keyframes_vector_[i]->GetPose();
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p = GeneratePointCloud(color_imgs_vector_[i], depth_imgs_vector_[i], pose);
                *global_map_ += *p;
            }
            is_loop_ = false;
        } else {
            {
                std::unique_lock<std::mutex> locker(keyframe_mutex_);
                while(keyframes_.empty() && !finish_flag_){
                    // 阻塞当前线程，直到insertKeyFrame中的notify_one()唤醒
                    // 阻塞条件，keyframes_中没有存储的kf同时线程没有被shut down
                    keyframe_update_condi_var_.wait(locker);
                }

                if (!(depth_imgs_.size() == color_imgs_.size() && keyframes_.size() == color_imgs_.size())) {
                    // 三个list数据对应
                    continue;
                }

                if (finish_flag_ && color_imgs_.empty() && depth_imgs_.empty() && keyframes_.empty()) {
                    // 接受结束信号，并且list中的图像全都处理完之后结束
                    break;
                }

                kf_ = keyframes_.front();
                color_img_ = color_imgs_.front();
                depth_img_ = depth_imgs_.front();
                keyframes_.pop_front();
                color_imgs_.pop_front();
                depth_imgs_.pop_front();
            } // keyframe_mutex_

            {
                std::unique_lock<std::mutex> locker(point_cloud_mutex_);
                cv::Mat pose = kf_->GetPose();
//            GeneratePointCloud2(color_img_, depth_img_, pose);
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p = GeneratePointCloud(color_img_, depth_img_, pose);
                *global_map_ += *p;
            } // point_cloud_mutex_
        }

        viewer.showCloud(global_map_);
        std::cout << "show point cloud, size=" << global_map_->points.size() << std::endl;
    }
}

void PointCloudMapping::SavePcdFile(const std::string &filename) {
    // 保存点云pcd文件到filename位置
    pcl::io::savePCDFile(filename, *global_map_);
}

} // namespace ORB_SLAM2
