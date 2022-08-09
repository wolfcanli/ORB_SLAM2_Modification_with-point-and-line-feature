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
PointCloudMapping::PointCloudMapping() {
    this->resolution = 0.005;
    voxel.setLeafSize(resolution, resolution, resolution);
    voxelForMatch.setLeafSize(0.1f, 0.1f, 0.1f);
    global_map_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();

    lastKeyframeSize = 0;
}

void PointCloudMapping::Reset() {
      mvPosePointClouds.clear();
      mvPointClouds.clear();
      mpointcloudID=0;
}

void PointCloudMapping::shutdown() {
    {
        std::unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
    }
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth) {
    unique_lock<mutex> lck(keyframeMutex);
    cv::Mat T =kf->GetPose();
    mvPosePointClouds.push_back(T.clone());
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    if(cx == 0 || cy == 0 || fx == 0 || fy == 0) {
        cx = kf->cx;
        cy = kf->cy;
        fx = kf->fx;
        fy = kf->fy;
    }
    keyFrameUpdated.notify_one();
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloudMapping::generatePointCloud(cv::Mat& color, cv::Mat& depth, cv::Mat& pose) {
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>());

    for ( int m=0; m<depth.rows; m+=3 ) {
        for ( int n=0; n<depth.cols; n+=3 ) {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m -cy) * p.z / fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = Converter::toSE3Quat(pose);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::transformPointCloud(*tmp, *cloud1, T.inverse().matrix());

    voxel.setInputCloud(cloud1);
    voxel.filter(*tmp);
    cloud1->is_dense = true;

    return cloud1;
}


pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloudMapping::generatePointCloud( cv::Mat& color, cv::Mat& depth) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>());
    
    for ( int m=0; m<depth.rows; m+=3 ) {
        for ( int n=0; n<depth.cols; n+=3 ) {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m -cy) * p.z / fy;  
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    // rotation the pointcloud and stiching 
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity() ;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::transformPointCloud(*tmp, *cloud1, T.inverse().matrix());
    pcl::transformPointCloud(*tmp, *cloud2, T.inverse().matrix()); //专用于点云匹配

    cloud1->is_dense = false;
    voxel.setInputCloud(cloud1);
    voxel.filter(*tmp);
    cloud1->swap(*tmp);

    cloud2->is_dense = false;
    voxelForMatch.setInputCloud(cloud2);
    voxelForMatch.filter(*tmp);
    cloud2->swap(*tmp);

    mvPointClouds.push_back(cloud1);
    mvPointCloudsForMatch.push_back(cloud2);
    mpointcloudID++;
    cout<<"generate point cloud from  kf-ID:"<<mpointcloudID<<", size="<<cloud1->points.size() << std::endl;

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>( t2-t1 );
    std::cout<<"   cost time: " << time_used.count() * 1000 << " ms ." << std::endl;

    return cloud1;
}

void PointCloudMapping::Run() {
    pcl::visualization::CloudViewer viewer("Run");

    while(1) {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag) {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0,i=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N =mvPosePointClouds.size();
        }
        for(i=lastKeyframeSize; i<N ; i++) {
            cv::Mat pose = keyframes[i]->GetPose();
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p = generatePointCloud(colorImgs[i], depthImgs[i], pose);
            *global_map_ += *p;
        }
        lastKeyframeSize = i;
        viewer.showCloud(global_map_);
        cout << "show global map, size=" << global_map_->points.size() << endl;
    }

}


void PointCloudMapping::RunNoSegmentation() {
    pcl::visualization::CloudViewer viewer("RunNoSegmentation");

    while(1) {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag) {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated
        size_t N=0,i=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N =mvPosePointClouds.size();
        }
        for(i=lastKeyframeSize; i<N ; i++) {
            //PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
            //*global_map_ += *p;
            if((mvPosePointClouds.size() != colorImgs.size()) ||
               (mvPosePointClouds.size()!= depthImgs.size()) ||
               (depthImgs.size() != colorImgs.size())) {
                cout<<" depthImgs.size != colorImgs.size()  "<<endl;
                continue;
            }
            cout<<"i: "<<i<<"  mvPosePointClouds.size(): "<<mvPosePointClouds.size()<<endl;

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tem_cloud1(new pcl::PointCloud<pcl::PointXYZRGBA>());
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tem_cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>());
            tem_cloud1 = generatePointCloud(colorImgs[i], depthImgs[i]);
            
            Eigen::Isometry3d T_c2w =ORB_SLAM2::Converter::toSE3Quat( mvPosePointClouds[i]);
            
            Eigen::Isometry3d T_cw= Eigen::Isometry3d::Identity();
            if(mvPointClouds.size() > 1) {
                Eigen::Isometry3d T_c1w =ORB_SLAM2::Converter::toSE3Quat( mvPosePointClouds[i-1]);
                Eigen::Isometry3d T_c1c2 = T_c1w*T_c2w.inverse();// T_c1_c2

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tem_match_cloud1 =mvPointCloudsForMatch[i-1];
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tem_match_cloud2 =mvPointCloudsForMatch[i];

                // 计算cloud1 cloud2 之间的相对旋转变换
                //pcl::transformPointCloud( *tem_match_cloud2, *cloud, T_c1c2.matrix());
                computeTranForTwoPiontCloud(tem_match_cloud1, tem_match_cloud2, T_c1c2);
                    
                T_cw = T_c1c2 * T_c1w;
            }
 
 		pcl::transformPointCloud( *tem_cloud1, *tem_cloud2, T_c2w.inverse().matrix());
 		//pcl::transformPointCloud( *tem_cloud1, *tem_cloud2,T_cw.inverse().matrix());
		
 		*global_map_ += *tem_cloud2;
	}
	lastKeyframeSize = i;
	viewer.showCloud(global_map_);
    cout << "show global map, size=" << global_map_->points.size() << endl;
	}

}

void PointCloudMapping::computeTranForTwoPiontCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &P1,
                                                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &P2,
                                                    Eigen::Isometry3d& T) {
    Eigen::Matrix<float, 4, 4> Tcw;
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> t; 

    //计算曲面法线和曲率
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>());
    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::PointNormal> norm_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    //norm_est.setSearchMethod (tree);
    norm_est.setKSearch (30);
    norm_est.setInputCloud (P1);
    norm_est.compute (*points_with_normals_src);
    pcl::copyPointCloud (*P1, *points_with_normals_src);
    norm_est.setInputCloud (P2);
    norm_est.compute (*points_with_normals_tgt);
    pcl::copyPointCloud (*P2, *points_with_normals_tgt);
//     
//     //举例说明我们自定义点的表示（以上定义）
//     MyPointRepresentation point_representation;
//     //调整'curvature'尺寸权重以便使它和x, y, z平衡
//     float alpha[4] = {1.0, 1.0, 1.0, 1.0};
//     point_representation.setRescaleValues (alpha);
//     //
    // 配准
    pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
    reg.setTransformationEpsilon (1e-6);
    //将两个对应关系之间的(src<->tgt)最大距离设置为10厘米
    //注意：根据你的数据集大小来调整
    reg.setMaxCorrespondenceDistance (0.1);  
    //     //设置点表示
    //    // reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
    reg.setInputCloud (points_with_normals_src);
    reg.setInputTarget (points_with_normals_tgt);

    //     //在一个循环中运行相同的最优化并且使结果可视化
    //     Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
    pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = points_with_normals_src;
    reg.setMaximumIterations (2);
    reg.align (*reg_result);

    Tcw = reg.getFinalTransformation();
    R(0,0)=Tcw(0,0);R(0,1)=Tcw(0,1);R(0,2)=Tcw(0,2);
    R(1,0)=Tcw(1,0);  R(1,1)=Tcw(1,1);  R(1,2)=Tcw(1,2);
    R(2,0)=Tcw(2,0);  R(2,1)=Tcw(2,1);  R(2,2)=Tcw(2,2);

    t(0,0)=Tcw(0,3);t(1,0)=Tcw(1,3); t(2,0)=Tcw(2,3);

    T.rotate(R);
    T.pretranslate(t);

}

void PointCloudMapping::RunSegmentation() {
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cout << "Python init fail!" << std::endl;
    }
//    import_array();

    PyRun_SimpleString("import sys");
//    PyRun_SimpleString("basepath = os.getcwd()");
    PyRun_SimpleString("sys.path.append('../deeplabv2')");

    PyObject* pModule = nullptr;
    PyObject* pArg = nullptr;
    PyObject* pFunc = nullptr;

    pModule = PyImport_ImportModule("inference");

    pFunc= PyObject_GetAttrString(pModule, "init");   // 这里是要调用的函数名
    PyObject_CallObject(pFunc, pArg);

    pcl::visualization::CloudViewer viewer("PointCloud Segmentation Viewer");

    while(true) {

    }


}


void PointCloudMapping::getGlobalCloudMap(pcl::PointCloud<pcl::PointXYZRGBA> ::Ptr &outputMap) {
	   unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );   
	   outputMap= global_map_;
}

void PointCloudMapping::SavePcdFile(const std::string &filename) {
    pcl::io::savePCDFile(filename, *global_map_);
}

} // namespace ORB_SLAM2
