#ifndef VIEWER_H_
#define VIEWER_H_

#include <Types.h>

// cpp
#include <cstddef>

// opencv
#include <cstdint>
#include <opencv2/opencv.hpp>

// tbb
#include <tbb/blocked_range2d.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>

namespace ScanReconstruction {

class Viewer {
public:
    explicit Viewer(int height, int width_, Eigen::Matrix3f k, float scale_factor)
        : height_(height), width_(width_), scale_factor_(scale_factor) {
        size_t total_size = (size_t)(height_ * width_);
        prev_normals_.resize(total_size, Eigen::Vector3f::Zero());
        prev_points_.resize(total_size, Eigen::Vector3f::Zero());
        current_points_.resize(total_size, Eigen::Vector3f::Zero());
        ray_map_.resize(total_size, Eigen::Vector3f::Zero());

        const Eigen::Matrix3f inv_k = k.inverse();

        for (int y = 0; y < height_; ++y)
            for (int x = 0; x < width_; ++x)
                ray_map_[(size_t)(y * width_ + x)] =
                    inv_k * Eigen::Vector3f(float(x), float(y), 1.0f);
    }

    void set_current_points(cv::Mat& origin_depth) {
        tbb::parallel_for(
            tbb::blocked_range2d<int>(0, height_, 0, width_),
            [&](const tbb::blocked_range2d<int>& r) {
                for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                    Eigen::Vector3f* points_ptr = current_points_.data() + (y * width_);
                    Eigen::Vector3f* ray_map_ptr = ray_map_.data() + (y * width_);
                    uint16_t* depth_img_ptr =
                        reinterpret_cast<uint16_t*>(origin_depth.data) + y * width_;
                    for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                        float depth_measure = (float)depth_img_ptr[x] / scale_factor_;
                        Eigen::Vector3f& point = points_ptr[x];
                        if (depth_measure > 1e-4)
                            point = ray_map_ptr[x] * depth_measure;
                        else
                            point(0) = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            });
    }

    void transform(const Eigen::Matrix4f global_pose) {
        const Eigen::Matrix3f rotation = global_pose.block(0, 0, 3, 3);
        const Eigen::Vector3f translation = global_pose.block(0, 3, 3, 1);

        tbb::parallel_for(
            tbb::blocked_range2d<int>(0, height_, 0, width_),
            [&](const tbb::blocked_range2d<int>& r) {
                for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                    Eigen::Vector3f* points_ptr = current_points_.data() + y * width_;
                    for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                        auto& point = points_ptr[x];
                        if (!std::isnan(point(0))) point = rotation * point + translation;
                    }
                }
            });
    }

    void swap_current_and_previous_points() {
        prev_points_.swap(current_points_);
        computePrevNormals();
    }

    const Points& get_current_points() const { return current_points_; }

    const Points& get_prev_points() const { return prev_points_; }

    const Normals& get_prev_normals() const { return prev_normals_; }

    TrackingResult get_tracking_result() const { return tracking_result_; }

    void set_tracking_result(TrackingResult result) { tracking_result_ = result; }

private:
    void computePrevNormals() {
        Eigen::Vector3f* points_ptr = prev_points_.data();
        tbb::parallel_for(
            tbb::blocked_range2d<int>(3, height_ - 3, 3, width_ - 3),
            [&](const tbb::blocked_range2d<int>& r) {
                for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                    Eigen::Vector3f* normals_ptr = prev_normals_.data() + (y * width_);
                    for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                        Eigen::Vector3f points[4];
                        Eigen::Vector3f& normal = normals_ptr[x];
                        Eigen::Vector3f diff_x, diff_y;

                        points[0] = points_ptr[x + 2 + y * width_];
                        points[1] = points_ptr[x + (y + 2) * width_];
                        points[2] = points_ptr[x - 2 + y * width_];
                        points[3] = points_ptr[x + (y - 2) * width_];

                        bool doPlus{false};

                        if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                            std::isnan(points[2](0)) || std::isnan(points[3](0)))
                            doPlus = true;

                        if (doPlus) {
                            points[0] = points_ptr[x + 1 + y * width_];
                            points[1] = points_ptr[x + (y + 1) * width_];
                            points[2] = points_ptr[x - 1 + y * width_];
                            points[3] = points_ptr[x + (y - 1) * width_];

                            if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                                std::isnan(points[2](0)) || std::isnan(points[3](0))) {
                                normal(0) = std::numeric_limits<float>::quiet_NaN();
                                continue;
                            }
                        }
                        diff_x = points[0] - points[2];
                        diff_y = points[1] - points[3];

                        normal = diff_y.cross(diff_x);

                        float norm = normal.norm();

                        if (norm < 1e-5) {
                            normal(0) = std::numeric_limits<float>::quiet_NaN();
                            continue;
                        }
                        normal /= norm;
                    }
                }
            });
    }

    Normals prev_normals_;
    Points prev_points_, current_points_;

    int height_, width_;

    float scale_factor_;

    std::vector<Eigen::Vector3f> ray_map_;

    TrackingResult tracking_result_{TrackingResult::GOOD};
};
}  // namespace ScanReconstruction
#endif  // VIEWER_H_
