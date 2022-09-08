//
// Created by jiajieshi on 22-9-7.
//

#ifndef ORB_SLAM2_TYPES_LINE_EXPMAP_H
#define ORB_SLAM2_TYPES_LINE_EXPMAP_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

using namespace Eigen;
using namespace g2o;

namespace ORB_SLAM2 {
class  EdgeLineProjectXYZOnlyPose: public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeLineProjectXYZOnlyPose(){}

    bool read(std::istream& is) {
        for (int i=0; i<=3; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool write(std::ostream& os) const {
        for (int i=0; i<=3; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    void computeError() {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        Vector3d obs(_measurement);
        // v1->estimate().map(Xw) 相当于Tcw * Pw
        Vector2d proj(cam_project(v1->estimate().map(Xw)));
        _error(0) = obs(0) * proj(0) + obs(1) * proj(1) + obs(2);
    }

    bool isDepthPositive() {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        return (v1->estimate().map(Xw))(2)>0.0;
    }


    virtual void linearizeOplus() {
        VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        double lx = _measurement(0); // 直线系数a
        double ly = _measurement(1); // 直线系数b

        _jacobianOplusXi(0,0) = -fy*ly - fx*lx*x*y*invz_2 - fy*ly*y*y*invz_2;
        _jacobianOplusXi(0,1) = fx*lx + fx*lx*x*x*invz_2 + fy*ly*x*y*invz_2;
        _jacobianOplusXi(0,2) = -fx*lx*y*invz + fy*ly*x*invz;

        _jacobianOplusXi(0,3) = fx*lx*invz;
        _jacobianOplusXi(0.4) = fy*ly*invz;
        _jacobianOplusXi(0,5) = -(fx*lx*x+fy*ly*y)*invz_2;
    }

    Vector2d cam_project(const Vector3d & trans_xyz) const {
        const float invz = 1.0f / trans_xyz[2];
        Vector2d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        return res;
    }

    Vector3d Xw;
    double fx, fy, cx, cy;
};

/**
 * 线段端点和相机位姿之间的边，线段端点的类型仍然采用g2o中的3D坐标类型
*/
class EdgeLineProjectXYZ : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeLineProjectXYZ() {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v1 = static_cast<VertexSE3Expmap *>(_vertices[0]);
        const VertexSBAPointXYZ* v2 = static_cast<VertexSBAPointXYZ*>(_vertices[1]);
        //这里是取0还是1，我觉得要看Optimizer.c文件中怎么为误差边添加顶点的.todo 这里是重大bug吗！! 应该是0对应地图点，1对应关键帧位姿
        /** optimizer.cpp 549行附近这样写的，可见误差边上0对应线的端点,1对应的是关键帧位姿，所以作者上面0 1位置应该颠倒了吧！！！
         *
         *  EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
         */

        Vector3d obs = _measurement;    //线段所在直线参数
        Vector2d proj = cam_project(v1->estimate().map(Xw));
        //todo BUG！误差边的两个顶点都是优化变量，计算误差的时候两个顶点的更新应该都会用到！ 参考types_six_dof_expmap.h第108行！  卧草，怎么处处有bug……
        //要改啊！！
        _error(0) = obs(0) * proj(0) + obs(1)*proj(1) + obs(2); //误差还是1维的
    }

    virtual void linearizeOplus()
    {
        // 位姿顶点
        VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
        SE3Quat T(vj->estimate());
        Vector3d xyz_trans = T.map(Xw);    //线段端点的世界坐标系转换到相机坐标系下  //todo sigh...这里也错了！不会用到Xw,请见types_six_dof_expmap.cpp中的Xw，这个变量只有在OnlyPose误差边中用到，因为空间点的位置是固定的。而这里空间点的位置要用估计 VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ*>(-vertices[0]), Xw = vi->estimate();  (tsde.c源文件112行)

        // 线段端点顶点
//        VertexLinePointXYZ *vj = static_cast<VertexLinePointXYZ *>(_vertices[1]);
        VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
        Vector3d xyz = vi->estimate();

        double x = xyz_trans[0];  //所以这些地方都要改，或者用xyz
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz * invz;

        double lx = _measurement(0);
        double ly = _measurement(1);

        // 3*6 jacobian
        // 1.这是最开始推导雅克比时，有负号的
//        _jacobianOplusXj(0,0) = fy*ly + fx*lx*x*y*invz_2 + fy*ly*y*y*invz_2;
//        _jacobianOplusXj(0,1) = -fx*lx - fx*lx*x*x*invz_2 - fy*ly*x*y*invz_2;
//        _jacobianOplusXj(0,2) = fx*lx*y*invz - fy*ly*x*invz;
//        _jacobianOplusXj(0,3) = -fx*lx*invz;
//        _jacobianOplusXj(0.4) = -fy*ly*invz;
//        _jacobianOplusXj(0,5) = (fx*lx*x+fy*ly*y)*invz_2;


        // 雅克比没有负号的。 注意这里的下标，只有位姿优化的时候是Xi，既有位姿又有地图点，点是Xi，位姿是Xj
        // 这一部分和EdgeLineProjectXYZOnlyPose中是一样的
        _jacobianOplusXj(0,0) = -fy*ly - fx*lx*x*y*invz_2 - fy*ly*y*y*invz_2;
        _jacobianOplusXj(0,1) = fx*lx + fx*lx*x*x*invz_2 + fy*ly*x*y*invz_2;
        _jacobianOplusXj(0,2) = -fx*lx*y*invz + fy*ly*x*invz;
        _jacobianOplusXj(0,3) = fx*lx*invz;
        _jacobianOplusXj(0.4) = fy*ly*invz;
        _jacobianOplusXj(0,5) = -(fx*lx*x+fy*ly*y)*invz_2;

        Matrix<double, 3, 3, Eigen::ColMajor> tmp;  //TODO 这里也有问题吧，不应该是一行三列吗，误差1维空间点3维，见tsde.c中124行
        tmp = Eigen::Matrix3d::Zero();
        tmp(0,0) = fx*lx;
        tmp(0,1) = fy*ly;
        tmp(0,2) = -(fx*lx*x+fy*ly*y)*invz;

        Matrix<double, 3, 3> R;
        R = T.rotation().toRotationMatrix();  //没毛病

//        _jacobianOplusXi = -1. * invz * tmp * R;
        _jacobianOplusXi = 1. * invz * tmp * R;  //这个公式应该没有问题
    }

    bool read(std::istream& is)
    {
        for(int i=0; i<3; i++)
        {
            is >> _measurement[i];
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                is >> information()(i, j);
                if(i!=j)
                    information()(j,i) = information()(i,j);
            }
        }
        return true;
    }

    bool write(std::ostream& os) const
    {
        for(int i=0; i<3; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                os << " " << information()(i,j);
            }
        }
        return os.good();
    }

    Vector2d project2d(const Vector3d& v)
    {
        Vector2d res;
        res(0) = v(0)/v(2);
        res(1) = v(1)/v(2);
        return res;
    }

    Vector2d cam_project(const Vector3d& trans_xyz)
    {
        Vector2d proj = project2d(trans_xyz);
        Vector2d res;
        res[0] = proj[0]*fx + cx;
        res[1] = proj[1]*fy + cy;
        return res;
    }

    Vector3d Xw;    //MapLine的一个端点在世界坐标系的位置
    double fx, fy, cx, cy;  //相机内参数
};


}



#endif //ORB_SLAM2_TYPES_LINE_EXPMAP_H
