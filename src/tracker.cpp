#include <tracker.h>

// tbb
#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

// eigen
#include <Eigen/Core>

// cpp
#include <utils.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

Tracker::Tracker(
    cv::Mat k, cv::Mat d, cv::Size2i imgSize, int max_num_threads, int max_num_iterations,
    float initial_lamdba, float lamdba_scale, float depth_scale, float space_threshold)
    : max_num_iterations_(max_num_iterations),
      initial_lamdba_(initial_lamdba),
      lamdba_scale_(lamdba_scale),
      depth_scale_(depth_scale),
      space_threshold_(space_threshold) {
    cv::Mat new_k = cv::getOptimalNewCameraMatrix(k, d, imgSize, 0, imgSize);
    height_ = imgSize.height;
    width_ = imgSize.width;
    k_ = Eigen::Matrix3f::Identity();
    k_(0, 0) = (float)k.at<double>(0, 0);
    k_(1, 1) = (float)k.at<double>(1, 1);
    k_(0, 2) = (float)k.at<double>(0, 2);
    k_(1, 2) = (float)k.at<double>(1, 2);
    cv::initUndistortRectifyMap(
        k, d, cv::Mat::eye(3, 3, CV_64F), new_k, imgSize, CV_32FC1, mapX_, mapY_);

    rayMap_.resize((size_t)(width_ * height_), Eigen::Vector3f::Zero());

    global_pose_ = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f inv_k = k_.inverse();
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x)
            rayMap_[(size_t)(y * width_ + x)] = inv_k * Eigen::Vector3f(float(x), float(y), 1.0f);

    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads));
}

Eigen::Matrix4f Tracker::track(View& view) {
    float f = std::numeric_limits<float>::infinity();
    Eigen::Matrix4f pose_good = global_pose_;
    Eigen::Matrix4f pose_old = pose_good;
    float lamdba{initial_lamdba_};

    for (int i{0}; i < max_num_iterations_; ++i) {
        auto [local_hessian, local_nabla, local_f] =
            buildLinearSystem(view, pose_good, global_pose_, k_, width_, height_);

        if (local_f < f) {
            lamdba /= lamdba_scale_;
            pose_old = pose_good;
            f = local_f;
        } else {
            pose_good = pose_old;
            lamdba *= lamdba_scale_;
            continue;
        }

        local_hessian += Eigen::Matrix<float, 6, 6>::Identity() * lamdba;
        Eigen::Vector<float, 6> delta = local_hessian.ldlt().solve(local_nabla);
        applyDelta(pose_good, delta);
    }

    return pose_good;
}

inline Eigen::Matrix3f skew(const Eigen::Vector3f v) {
    Eigen::Matrix3f m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

void Tracker::applyDelta(Eigen::Matrix4f& pose, Eigen::Vector<float, 6> delta) {
    Eigen::Vector3f dw = delta.head<3>();
    Eigen::Vector3f dv = delta.tail<3>();
    Eigen::Matrix3f dR = (Eigen::AngleAxisf(dw.norm(), dw.normalized())).toRotationMatrix();
    pose.block(0, 0, 3, 3) = dR * pose.block(0, 0, 3, 3);
    pose.block(0, 3, 3, 1) = dR * pose.block(0, 3, 3, 1) + dv;
    Eigen::Quaternionf q_final(pose.block<3, 3>(0, 0));
    q_final.normalize();
    pose.block<3, 3>(0, 0) = q_final.toRotationMatrix();
}

std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Vector<float, 6>, float> Tracker::buildLinearSystem(
    const View& view, const Eigen::Matrix4f& current_pose, const Eigen::Matrix4f& prev_pose,
    Eigen::Matrix3f k, int width, int height) {
    const Eigen::Matrix3f current_r = current_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f current_t = current_pose.block(0, 3, 3, 1);
    const Eigen::Matrix3f prev_r_inv = prev_pose.block(0, 0, 3, 3).inverse();
    const Eigen::Vector3f prev_t = prev_pose.block(0, 3, 3, 1);

    const Eigen::Vector3f* prev_depth_ptr = view.prev_depth.data();
    const Eigen::Vector3f* prev_normal_ptr = view.prev_normal.data();
    const Eigen::Vector3f* current_depth_ptr = view.depth.data();

    // 计算hessian，nabla,f
    auto compute_jacobian_and_residual = [&](int x, int y) -> LinearSystem {
        // 获取上一帧所对应的法向量以及点
        const Eigen::Vector3f& current_point = current_depth_ptr[y * width + x];
        if (std::isnan(current_point[0])) return LinearSystem();
        Eigen::Vector3f point = current_r * current_point + current_t;
        Eigen::Vector3f project_point = k * prev_r_inv * (point - prev_t);
        project_point /= project_point(2);
        int coord_x = static_cast<int>(std::floor(project_point(0)));
        int coord_y = static_cast<int>(std::floor(project_point(1)));
        if (coord_x < 0 || coord_x > width - 1 || coord_y < 0 || coord_y > height - 1)
            return LinearSystem();
        const Eigen::Vector3f& prev_point = prev_depth_ptr[coord_y * width + coord_x];
        if (std::isnan(prev_point(0))) return LinearSystem();
        const Eigen::Vector3f& prev_normal = prev_normal_ptr[coord_y * width + coord_x];
        if (std::isnan(prev_normal(0))) return LinearSystem();

        LinearSystem linearSystem;
        // 计算误差
        float diff = prev_normal.transpose() * (point - prev_point);

        Eigen::Matrix<float, 3, 6> A = Eigen::Matrix<float, 3, 6>::Zero();
        A.block<3, 3>(0, 0) = -skew(point);
        A.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();

        Eigen::Matrix<float, 1, 6> J_r = prev_normal.transpose() * A;
        linearSystem.hessian = J_r.transpose() * J_r * rho_deriv2(diff, space_threshold_);
        linearSystem.nabla = -J_r * rho_deriv(diff, space_threshold_);
        linearSystem.f = rho(diff, space_threshold_);
        return linearSystem;
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem& b) {
        a.hessian += b.hessian;
        a.nabla += b.nabla;
        a.f += b.f;
        return a;
    };

    LinearSystem total = tbb::parallel_reduce(
        tbb::blocked_range2d<int>(0, height, 0, width), LinearSystem(),
        [&](const tbb::blocked_range2d<int>& r, LinearSystem local_sum) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                for (int x = r.cols().begin(); x < r.cols().end(); ++x)
                    local_sum = sum_linear_systems(local_sum, compute_jacobian_and_residual(x, y));

            return local_sum;
        },
        sum_linear_systems);

    return {total.hessian, total.nabla, total.f};
}

void Tracker::undistortion(cv::Mat& origin_depth, View& view) {
    cv::Mat tmp;
    cv::remap(origin_depth, tmp, mapX_, mapY_, cv::INTER_NEAREST);
    Eigen::Vector3f* depth_ptr = view.depth.data();

    uint16_t* depth_img_ptr = reinterpret_cast<uint16_t*>(tmp.data);
    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, height_, 0, width_), [&](const tbb::blocked_range2d<int>& r) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    size_t offset = static_cast<size_t>(y * width_ + x);
                    float depth_measure = (float)depth_img_ptr[offset] / depth_scale_;
                    Eigen::Vector3f& point = depth_ptr[offset];
                    if (depth_measure > 1e-4)
                        point = rayMap_[offset] * depth_measure;
                    else
                        point(0) = std::numeric_limits<float>::quiet_NaN();
                }
        });
}

void Tracker::computeNormalMap(View& view) {
    const int width = view.width;
    const int height = view.height;

    Eigen::Vector3f* normal_ptr = view.prev_normal.data();
    const Eigen::Vector3f* depth_ptr = view.prev_depth.data();
    tbb::parallel_for(
        tbb::blocked_range2d<int>(3, height - 3, 3, width - 3),
        [&](const tbb::blocked_range2d<int>& r) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    Eigen::Vector3f points[4];
                    Eigen::Vector3f& normal = normal_ptr[x + y * width];
                    Eigen::Vector3f diff_x, diff_y;

                    points[0] = depth_ptr[x + 2 + y * width];
                    points[1] = depth_ptr[x + (y + 2) * width];
                    points[2] = depth_ptr[x - 2 + y * width];
                    points[3] = depth_ptr[x + (y - 2) * width];

                    bool doPlus{false};

                    if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                        std::isnan(points[2](0)) || std::isnan(points[3](0)))
                        doPlus = true;

                    if (doPlus) {
                        points[0] = depth_ptr[x + 1 + y * width];
                        points[1] = depth_ptr[x + (y + 1) * width];
                        points[2] = depth_ptr[x - 1 + y * width];
                        points[3] = depth_ptr[x + (y - 1) * width];

                        if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                            std::isnan(points[2](0)) || std::isnan(points[3](0)))
                            continue;
                    }
                    diff_x = points[0] - points[2];
                    diff_y = points[1] - points[3];

                    normal = diff_y.cross(diff_x);

                    float norm = normal.norm();

                    if (norm < 1e-5) continue;
                    normal(0) /= norm;
                    normal(1) /= norm;
                    normal(2) /= norm;
                }
        });
}

void Tracker::transform(View& view, const Eigen::Matrix4f global_pose) {
    const int width = view.width;
    const int height = view.height;
    const Eigen::Matrix3f rotation = global_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f translation = global_pose.block(0, 3, 3, 1);
    Eigen::Vector3f* depth_ptr = view.prev_depth.data();

    tbb::parallel_for(
        tbb::blocked_range2d<int>(3, height - 3, 3, width - 3),
        [&](const tbb::blocked_range2d<int>& r) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    auto& point = depth_ptr[y * width + x];
                    if (!std::isnan(point(0))) point = rotation * point + translation;
                }
        });
}
