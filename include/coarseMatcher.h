#ifndef COARSE_MATCHER_H_
#define COARSE_MATCHER_H_

// eigen
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <Eigen/Eigen>

// cpp
#include <cstddef>
#include <optional>
#include <pcl/impl/point_types.hpp>

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

// pcl
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>

namespace surface_reconstruction {
class CoarseMatcher {
public:
    CoarseMatcher(
        const cv::Mat cameraMatrix, const cv::Mat distCoeffs, const cv::Size2i imgSize,
        const float scale, float subSampleSize = 0.03f, float curvatureThreshold = 0.15f,
        int nIterations = 8000, int sampleSize = 3, float distanceThreshold = 0.03f,
        float minInlierArtio = 0.4f, float searchRadius = 0.05f, float similarityThreshold = 0.7f);

    void preprocessDepthFrame(cv::Mat& origin_depth, cv::Mat& processed_depth);

    void updatePrevDepth(cv::Mat& depth, pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud) {
        prev_depth_ = depth;
        prev_pointclouds_ = pointcloud;
    }

    void computePointClouds(const cv::Mat& depth, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr extractHighCurvatureFeatures(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input);

    std::optional<Eigen::Matrix4f> computeTransformMatrix(
        pcl::PointCloud<pcl::PointXYZ>::Ptr current, pcl::PointCloud<pcl::PointXYZ>::Ptr prev);

    static Eigen::Matrix4f icpIteration(
        std::vector<Eigen::Vector3f>& current, std::vector<Eigen::Vector3f>& prev);

    bool hasInitialed() { return prev_pointclouds_ != nullptr; }

    cv::Mat project(Eigen::Matrix4f transform, cv::Mat& source);

    pcl::PointCloud<pcl::PointXYZ>::Ptr get_prev_pointcloud() { return prev_pointclouds_; }

private:
    // 相机内参
    Eigen::Matrix3f k_;

    // 映射表
    cv::Mat mapX_, mapY_;

    // 原始深度值转m单位尺度
    float scale_;

    // 新的图像尺寸
    cv::Size2i imgSize_;

    // 上一帧去畸变的深度图
    cv::Mat prev_depth_;

    // 上一帧对应的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pointclouds_{nullptr};

    // 体素下采样尺寸
    float subSampleSize_;

    // 点云曲率阈值
    float curvatureThreshold_;

    // ransac迭代次数
    int nIterations_;

    // ransac采样样本数
    int sampleSize_;

    // 视为内点的距离阈值
    float distanceThreshold_;

    // ransac内点最小比例
    float minInlierArtio_;

    // kd树搜索半径，剪枝加速搜索
    float searchRadius_;

    // fpfh特征描述子相似度阈值，满足才视为配对点
    float similarityThreshold_;

    // 是否需要icp迭代
    bool useICP_{true};
};
}  // namespace surface_reconstruction

#endif
