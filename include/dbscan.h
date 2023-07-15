//
// Created by vell on 2022/12/20.
//

#ifndef DYNAMIC_MAP_DBSCAN_H
#define DYNAMIC_MAP_DBSCAN_H

#include <iostream>
#include<string>
#include<vector>

#include<pcl/io/pcd_io.h>
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/visualization/cloud_viewer.h>
#include <pcl/common/geometry.h>

template<typename PointT, typename CloudPtr>
inline bool
dbscan(const CloudPtr &cloud_in, std::vector<std::vector<int>> &clusters_index, const double r, const int size) {
  if (!cloud_in->size())
    return false;
  pcl::KdTreeFLANN<PointT> tree;
  tree.setInputCloud(cloud_in);
  std::vector<bool> cloud_processed(cloud_in->size(), false);

  for (int i = 0; i < cloud_in->points.size(); ++i) {
    if (cloud_processed[i]) {
      continue;
    }

    std::vector<int> seed_queue;
    std::vector<int> center_vec;
    //检查近邻数是否大于给定的size（判断是否是核心对象）
    std::vector<int> indices_cloud;
    std::vector<float> dists_cloud;
    if (tree.radiusSearch(cloud_in->points[i], r, indices_cloud, dists_cloud) >= size) {
      seed_queue.push_back(i);
      cloud_processed[i] = true;
    } else {
      continue;
    }

    int seed_index = 0;
    while (seed_index < seed_queue.size()) {
      std::vector<int> indices;
      std::vector<float> dists;
      if (tree.radiusSearch(cloud_in->points[seed_queue[seed_index]], r, indices, dists) < size)//函数返回值为近邻数量
      {
        //cloud_processed[i] = true;//不满足<size可能是边界点，也可能是簇的一部分，不能标记为已处理
        ++seed_index;
        continue;
      }
      for (int j = 0; j < indices.size(); ++j) {
        if (cloud_processed[indices[j]]) {
          continue;
        }
        seed_queue.push_back(indices[j]);
        cloud_processed[indices[j]] = true;
      }
      if (center_vec.empty()){
        center_vec.push_back(seed_queue[seed_index]);
      } else {
        auto &lastP = cloud_in->points[center_vec.back()];
        auto &curP = cloud_in->points[seed_queue[seed_index]];
        double dist = pcl::geometry::squaredDistance(lastP, curP);
        if (dist > r) {
          center_vec.push_back(seed_queue[seed_index]);
        } else {
          if (curP.z > lastP.z) {
            center_vec[center_vec.size() - 1] = seed_queue[seed_index];
          }
        }
        ++seed_index;
      }
    }
    clusters_index.push_back(center_vec);

  }
  return !clusters_index.empty();
}


#endif //DYNAMIC_MAP_DBSCAN_H
