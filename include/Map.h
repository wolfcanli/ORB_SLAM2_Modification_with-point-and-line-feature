/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include "MapLine.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapLine;

class Map
{
public:
    Map();

    // 插入新的关键帧和地图点到Map中
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    // 插入地图线到map中
    void AddMapLine(MapLine* pML);

    // 删除指定的关键帧和地图点
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    // 删除指定的地图线
    void EraseMapLine(MapLine* pML);

    // 设置参考地图点
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    // 设置参考地图线
    void SetReferenceMapLine(const std::vector<MapLine*> &vpMLs);

    // 这个在回环检测中用到了，暂时不知道什么意思
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    // 获取Map中全部的关键帧和地图点，以及参考地图点
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    // 回去全部地图线和参考地图线
    std::vector<MapLine*> GetAllMapLines();
    std::vector<MapLine*> GetReferenceMapLines();

    // 获取Map中地图点和关键帧的数量
    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();
    // 获取Map中地图线数量
    long unsigned int MapLinesInMap();

    // 获取最大关键帧的id
    long unsigned int GetMaxKFid();

    // 清除全部内容，包括清除全部地图点、关键帧，以及参考地图点等一系列变量
    void clear();

    // 保存了最初始的关键帧
    vector<KeyFrame*> mvpKeyFrameOrigins;

    // 当更新地图时的互斥量.回环检测中和局部BA后更新全局地图的时候会用到这个
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    // 为了避免地图点id冲突设计的互斥量
    std::mutex mMutexPointCreation;
    // 避免地图线id冲突
    std::mutex mMutexLineCreation;

protected:
    // 所有地图点
    std::set<MapPoint*> mspMapPoints;
    // 所有关键帧
    std::set<KeyFrame*> mspKeyFrames;
    // 所有地图线
    std::set<MapLine*> mspMapLines;

    // 参考地图点
    std::vector<MapPoint*> mvpReferenceMapPoints;
    // 参考地图线
    std::vector<MapLine*> mvpReferenceMapLines;

    // 当前地图中最大关键帧id
    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    // 回环和全局BA中用到，暂时不知道什么意思
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
