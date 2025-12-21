#include <coarseMatcher.h>
#include <opencv2/core/hal/interface.h>
#include <pcl/types.h>
#include <cmath>
#include <mutex>
#include <pcl/impl/point_types.hpp>

namespace surface_reconstruction {

CoarseMatcher::CoarseMatcher(
    const cv::Mat cameraMatrix, const cv::Mat distCoeffs, const cv::Size2i imgSize,
    const float scale, float subSampleSize, float curvatureThreshold, int nIterations,
    int sampleSize, float distanceThreshold, float minInlierArtio, float searchRadius,
    float similarityThreshold)
    : scale_(scale),
      subSampleSize_(subSampleSize),
      curvatureThreshold_(curvatureThreshold),
      nIterations_(nIterations),
      sampleSize_(sampleSize),
      distanceThreshold_(distanceThreshold),
      minInlierArtio_(minInlierArtio),
      searchRadius_(searchRadius),
      similarityThreshold_(similarityThreshold) {
    cv::Mat new_k = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 0, imgSize);
    imgSize_ = imgSize;
    k_ = Eigen::Matrix3f::Identity();
    k_(0, 0) = new_k.at<double>(0, 0);
    k_(1, 1) = new_k.at<double>(1, 1);
    k_(0, 2) = new_k.at<double>(0, 2);
    k_(1, 2) = new_k.at<double>(1, 2);
    cv::initUndistortRectifyMap(
        cameraMatrix, distCoeffs, cv::Mat::eye(3, 3, CV_64F), new_k, imgSize, CV_32FC1, mapX_,
        mapY_);
}

void CoarseMatcher::preprocessDepthFrame(cv::Mat& origin_depth, cv::Mat& processed_depth) {
    cv::Mat tmp;
    cv::remap(origin_depth, tmp, mapX_, mapY_, cv::INTER_NEAREST);

    const int rows = imgSize_.height;
    const int cols = imgSize_.width;

    processed_depth.create(imgSize_, CV_32FC1);
    float* depth = (float*)processed_depth.data;
    uint16_t* ori_depth = (uint16_t*)tmp.data;

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            uint16_t depth_measure = ori_depth[offset + x];
            depth[offset + x] = depth_measure > 0 ? (float)depth_measure / scale_ : 0.0f;
        }
    }
}

void CoarseMatcher::computePointClouds(
    const cv::Mat& depth, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud) {
    if (point_cloud == nullptr) point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    point_cloud->clear();

    int rows = imgSize_.height;
    int cols = imgSize_.width;
    point_cloud->resize(rows * cols);

    Eigen::Matrix3f inv_k = k_.inverse();
    float* depth_data = (float*)depth.data;
#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            float depth_measure = depth_data[offset + x];
            if (depth_measure > 1e-4) {
                Eigen::Vector3f point = inv_k * Eigen::Vector3f(x, y, 1.0f) * depth_measure;
                point_cloud->data()[offset + x] = pcl::PointXYZ(point(0), point(1), point(2));
            } else
                point_cloud->points[offset + x].x = std::numeric_limits<float>::infinity();
        }
    }

    point_cloud->points.erase(
        std::remove_if(
            point_cloud->points.begin(), point_cloud->points.end(),
            [](const pcl::PointXYZ& p) { return !std::isfinite(p.x); }),
        point_cloud->points.end());

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(point_cloud);
    sor.setLeafSize(subSampleSize_, subSampleSize_, subSampleSize_);
    sor.filter(*cloud_filtered);

    point_cloud = cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CoarseMatcher::extractHighCurvatureFeatures(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(input);
    ne.setSearchMethod(tree);
    ne.setKSearch(30);
    ne.compute(*normals);

    pcl::PointIndices::Ptr char_indices(new pcl::PointIndices);
    for (size_t i = 0; i < normals->points.size(); ++i) {
        float curvature = normals->points[i].curvature;
        if (curvature > curvatureThreshold_ && std::isfinite(curvature)) {
            char_indices->indices.push_back(i);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input);
    extract.setIndices(char_indices);
    extract.setNegative(false);
    extract.filter(*keypoints);

    return keypoints;
}

std::optional<Eigen::Matrix4f> CoarseMatcher::computeTransformMatrix(
    pcl::PointCloud<pcl::PointXYZ>::Ptr current, pcl::PointCloud<pcl::PointXYZ>::Ptr prev) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(30);

    pcl::PointCloud<pcl::Normal>::Ptr current_norm(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr prev_norm(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(current);
    ne.setSearchSurface(current);
    ne.compute(*current_norm);

    ne.setInputCloud(prev);
    ne.setSearchSurface(prev);
    ne.compute(*prev_norm);

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(searchRadius_);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr current_feat(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr prev_feat(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.setInputCloud(current);
    fpfh.setInputNormals(current_norm);
    fpfh.setSearchSurface(current);
    fpfh.compute(*current_feat);

    fpfh.setInputCloud(prev);
    fpfh.setInputNormals(prev_norm);
    fpfh.setSearchSurface(prev);
    fpfh.compute(*prev_feat);

    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> align;
    align.setInputSource(current);
    align.setSourceFeatures(current_feat);
    align.setInputTarget(prev);
    align.setTargetFeatures(prev_feat);

    align.setMaximumIterations(nIterations_);
    align.setNumberOfSamples(sampleSize_);
    align.setCorrespondenceRandomness(5);
    align.setSimilarityThreshold(0.8f);
    align.setMaxCorrespondenceDistance(distanceThreshold_);
    align.setInlierFraction(minInlierArtio_);

    pcl::PointCloud<pcl::PointXYZ> final_cloud;
    align.align(final_cloud);

    if (align.hasConverged()) {
        if (useICP_) {
            std::vector<Eigen::Vector3f> current_points, prev_points;
            pcl::Indices inliers = align.getInliers();
            pcl::search::KdTree<pcl::PointXYZ> tree;
            tree.setInputCloud(prev);

            for (int idx : inliers) {
                pcl::PointXYZ pt_src_transformed = final_cloud.points[idx];
                std::vector<int> nn_indices(1);
                std::vector<float> nn_dists(1);

                if (tree.nearestKSearch(pt_src_transformed, 1, nn_indices, nn_dists) > 0) {
                    if (nn_dists[0] < distanceThreshold_ * distanceThreshold_) {
                        current_points.push_back(current->points[idx].getVector3fMap());
                        prev_points.push_back(prev->points[nn_indices[0]].getVector3fMap());
                    }
                }
            }
            return icpIteration(current_points, prev_points);
        } else
            return align.getFinalTransformation();
    }

    return std::nullopt;
}

Eigen::Matrix3f skew(const Eigen::Vector3f v) {
    Eigen::Matrix3f m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

Eigen::Matrix4f CoarseMatcher::icpIteration(
    std::vector<Eigen::Vector3f>& current, std::vector<Eigen::Vector3f>& prev) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f old_transform = transform;
    float f = std::numeric_limits<float>::infinity();

    constexpr int iteration = 10;

    const int num = current.size();

    float lamdba{1.0f};

    for (int count{0}; count < iteration; ++count) {
        float local_f{0.0f};

        Eigen::Matrix<float, 6, 6> local_hessian = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> local_nabla = Eigen::Vector<float, 6>::Zero();

        for (int id{0}; id < num; ++id) {
            Eigen::Matrix<float, 3, 6> A = Eigen::Matrix<float, 3, 6>::Zero();
            Eigen::Vector3f transform_point =
                (transform * (Eigen::Vector4f() << current[id], 1.0f).finished()).head(3);
            Eigen::Vector3f diff = transform_point - prev[id];
            A.block<3, 3>(0, 0) = -skew(transform_point);
            A.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();
            local_hessian += A.transpose() * A;
            local_nabla += -A.transpose() * diff;
            local_f += diff.squaredNorm();
        }

        if (local_f < f) {
            old_transform = transform;
            f = local_f;
            lamdba /= 10.0f;
        } else {
            lamdba *= 10.0f;
            transform = old_transform;
            continue;
        }

        // 将变化量左乘transform
        local_hessian += Eigen::Matrix<float, 6, 6>::Identity() * lamdba;
        Eigen::Matrix<float, 1, 6> delta = local_hessian.ldlt().solve(local_nabla);
        Eigen::Vector3f dw = delta.head<3>();
        Eigen::Vector3f dv = delta.tail<3>();
        Eigen::Matrix3f dR = (Eigen::AngleAxisf(dw.norm(), dw.normalized())).toRotationMatrix();
        transform.block(0, 0, 3, 3) = dR * transform.block(0, 0, 3, 3);
        transform.block(0, 3, 3, 1) = dR * transform.block(0, 3, 3, 1) + dv;
        Eigen::Quaternionf q_final(transform.block<3, 3>(0, 0));
        q_final.normalize();
        transform.block<3, 3>(0, 0) = q_final.toRotationMatrix();
    }

    return transform;
}

cv::Mat CoarseMatcher::project(Eigen::Matrix4f transform, cv::Mat& source) {
    int rows = source.rows;
    int cols = source.cols;

    cv::Mat project_mat = cv::Mat(rows, cols, CV_32FC1, 0.0f);

    Eigen::Vector3f t = transform.block(0, 3, 3, 1);
    Eigen::Matrix3f r = transform.block(0, 0, 3, 3);

    Eigen::Matrix3f inv_k = k_.inverse();

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float depth_measure = source.at<float>(y, x);
            Eigen::Vector3f transform_point =
                k_ * (r * depth_measure * inv_k * Eigen::Vector3f(x, y, 1.0f) + t);
            int coord_x = std::round(transform_point(0) / transform_point(2));
            int coord_y = std::round(transform_point(1) / transform_point(2));
            if (transform(2) > 0 && coord_x > 0 && coord_x < cols && coord_y > 0 && coord_y < rows)
                project_mat.at<float>(coord_y, coord_x) = transform(2);
        }
    }

    return project_mat;
}
}  // namespace surface_reconstruction
