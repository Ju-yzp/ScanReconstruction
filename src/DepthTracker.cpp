// scan_recomstruction
#include <DepthTracker.h>
#include <PixelUtils.h>
#include <Utils.h>

// cpp
#include <cstddef>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

namespace ScanReconstruction {
DepthTracker::DepthTracker(std::shared_ptr<GlobalSettings> global_settings)
    : levels_(global_settings->pyramid_levels),
      last_pose_(Eigen::Matrix4f::Identity()),
      last_delta_(Eigen::Matrix4f::Identity()),
      initial_lamdba_(global_settings->initial_lamdba),
      lamdba_scale_(global_settings->lamdba_scale) {
    size_t levels = global_settings->pyramid_levels;
    current_points_pyramids_.resize(levels);
    prev_points_pyramids_.resize(levels);
    prev_normals_pyramids_.resize(levels);
    prev_normals_pyramids_.resize(levels);
    space_threhold_pyramids_.resize(levels);
    camera_params_pyramids_.resize(levels);
    iterations_pyramids_.resize(levels);

    camera_params_pyramids_[0] =
        CameraParams{global_settings->k, global_settings->width, global_settings->height};
    space_threhold_pyramids_[0] = global_settings->space_threshold_min;
    iterations_pyramids_[0] = global_settings->min_num_iterations;

    int width = global_settings->width;
    int height = global_settings->height;
    size_t total_size = size_t(width * height);

    current_points_pyramids_[0].resize(total_size, Eigen::Vector3f::Zero());
    prev_points_pyramids_[0].resize(total_size, Eigen::Vector3f::Zero());
    prev_normals_pyramids_[0].resize(total_size, Eigen::Vector3f::Zero());

    float space_threhold_step =
        (global_settings->space_threshold_max - global_settings->space_threshold_min) /
        static_cast<float>(levels - 1);
    int iterations_step =
        (global_settings->max_num_iterations - global_settings->min_num_iterations) /
        static_cast<int>(levels - 1);
    for (size_t id = 1; id < levels; ++id) {
        width /= 2;
        height /= 2;
        total_size = size_t(width * height);
        current_points_pyramids_[id].resize(total_size, Eigen::Vector3f::Zero());
        prev_points_pyramids_[id].resize(total_size, Eigen::Vector3f::Zero());
        prev_normals_pyramids_[id].resize(total_size, Eigen::Vector3f::Zero());
        space_threhold_pyramids_[id] = space_threhold_pyramids_[id - 1] + space_threhold_step;
        iterations_pyramids_[id] = iterations_pyramids_[id - 1] + iterations_step;
        camera_params_pyramids_[id] =
            CameraParams{camera_params_pyramids_[id - 1].k / 2.0f, width, height};
        camera_params_pyramids_[id].k(2, 2) = 1.0f;
    }
}

void DepthTracker::prepare(std::shared_ptr<Viewer> viewer) {
    current_points_pyramids_[0] = viewer->get_current_points();
    prev_points_pyramids_[0] = viewer->get_prev_points();
    prev_normals_pyramids_[0] = viewer->get_prev_normals();

    for (size_t i = 1; i < levels_; ++i) {
        filterSubsampleWithHoles(
            current_points_pyramids_[i - 1], current_points_pyramids_[i],
            camera_params_pyramids_[i - 1].height, camera_params_pyramids_[i - 1].width, false);
        filterSubsampleWithHoles(
            prev_points_pyramids_[i - 1], prev_points_pyramids_[i],
            camera_params_pyramids_[i - 1].height, camera_params_pyramids_[i - 1].width, false);
        filterSubsampleWithHoles(
            prev_normals_pyramids_[i - 1], prev_normals_pyramids_[i],
            camera_params_pyramids_[i - 1].height, camera_params_pyramids_[i - 1].width, true);
    }
}

void DepthTracker::reset() {
    last_pose_ = Eigen::Matrix4f::Identity();
    last_delta_ = Eigen::Matrix4f::Identity();
}

void DepthTracker::applyDelta(Eigen::Matrix4f& pose, const Eigen::Vector<float, 6>& delta) {
    Eigen::Vector3f dw = delta.head<3>();
    Eigen::Vector3f dv = delta.tail<3>();
    Eigen::Matrix3f dR = (Eigen::AngleAxisf(dw.norm(), dw.normalized())).toRotationMatrix();
    pose.block(0, 0, 3, 3) = dR * pose.block(0, 0, 3, 3);
    pose.block(0, 3, 3, 1) = dR * pose.block(0, 3, 3, 1) + dv;
    Eigen::Quaternionf q_final(pose.block<3, 3>(0, 0));
    q_final.normalize();
    pose.block<3, 3>(0, 0) = q_final.toRotationMatrix();
}

DepthTracker::LinearSystem DepthTracker::buuildLinearSystem(
    const Points& current_points, const Points& prev_points, const Normals& prev_normals,
    const Eigen::Matrix4f& current_pose, const Eigen::Matrix4f& prev_pose,
    const DepthTracker::CameraParams camera_params, const float space_threshold) {
    const Eigen::Matrix3f current_r = current_pose.block(0, 0, 3, 3);
    const Eigen::Vector3f current_t = current_pose.block(0, 3, 3, 1);
    const Eigen::Matrix3f prev_r_inv = prev_pose.block(0, 0, 3, 3).inverse();
    const Eigen::Vector3f prev_t = prev_pose.block(0, 3, 3, 1);

    const int width = camera_params.width;
    const int height = camera_params.height;
    const Eigen::Matrix3f k = camera_params.k;

    const Eigen::Vector3f* prev_points_ptr = prev_points.data();
    const Eigen::Vector3f* prev_normals_ptr = prev_normals.data();
    const Eigen::Vector3f* current_points_ptr = current_points.data();

    // 计算hessian，nabla,f
    auto compute_jacobian_and_residual = [&](int x, int y) -> LinearSystem {
        // 获取上一帧所对应的法向量以及点
        const Eigen::Vector3f& current_point = current_points_ptr[y * width + x];
        if (std::isnan(current_point[0])) return LinearSystem();
        Eigen::Vector3f point = current_r * current_point + current_t;
        Eigen::Vector3f project_point = k * prev_r_inv * (point - prev_t);
        project_point /= project_point(2);
        int coord_x = static_cast<int>(std::floor(project_point(0)));
        int coord_y = static_cast<int>(std::floor(project_point(1)));
        if (coord_x < 0 || coord_x > width - 1 || coord_y < 0 || coord_y > height - 1)
            return LinearSystem();
        const Eigen::Vector3f& prev_point = prev_points_ptr[coord_y * width + coord_x];
        if (std::isnan(prev_point(0))) return LinearSystem();
        const Eigen::Vector3f& prev_normal = prev_normals_ptr[coord_y * width + coord_x];
        if (std::isnan(prev_normal(0))) return LinearSystem();

        LinearSystem linearSystem;
        // 计算误差
        float diff = prev_normal.transpose() * (point - prev_point);

        Eigen::Matrix<float, 3, 6> A = Eigen::Matrix<float, 3, 6>::Zero();
        A.block<3, 3>(0, 0) = -skew(point);
        A.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();

        Eigen::Matrix<float, 1, 6> J_r = prev_normal.transpose() * A;
        linearSystem.hessian = J_r.transpose() * J_r * rho_deriv2(diff, space_threshold);
        linearSystem.nabla = -J_r * rho_deriv(diff, space_threshold);
        linearSystem.f = rho(diff, space_threshold);
        linearSystem.valid = 1;
        return linearSystem;
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem& b) {
        a.hessian += b.hessian;
        a.nabla += b.nabla;
        a.f += b.f;
        a.valid += b.valid;
        return a;
    };

    LinearSystem total = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range2d<int>(0, height, 0, width), LinearSystem(),
        [&](const oneapi::tbb::blocked_range2d<int>& r, LinearSystem local_sum) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y)
                for (int x = r.cols().begin(); x < r.cols().end(); ++x)
                    local_sum = sum_linear_systems(local_sum, compute_jacobian_and_residual(x, y));

            return local_sum;
        },
        sum_linear_systems);

    return total;
}

void DepthTracker::track(std::shared_ptr<Viewer> viewer, Eigen::Matrix4f& initial_pose) {
    prepare(viewer);

    // Eigen::Matrix<float, 6, 6> hessian_good = Eigen::Matrix<float, 6, 6>::Zero();
    // Eigen::Vector<float, 6> nabla_good = Eigen::Vector<float, 6>::Zero();
    Eigen::Matrix4f pose_good = initial_pose;
    // int valid_points_good = 0;

    for (int current_level = (int)(levels_ - 1); current_level >= 0; --current_level) {
        Eigen::Matrix4f old_pose = initial_pose;
        float f = std::numeric_limits<float>::max();
        float lamdba = initial_lamdba_;
        for (int iter = 0; iter < iterations_pyramids_[(size_t)current_level]; ++iter) {
            LinearSystem linearSystem = buuildLinearSystem(
                current_points_pyramids_[(size_t)current_level],
                prev_points_pyramids_[(size_t)current_level],
                prev_normals_pyramids_[(size_t)current_level], pose_good, last_pose_,
                camera_params_pyramids_[(size_t)current_level],
                space_threhold_pyramids_[(size_t)current_level]);

            if (linearSystem.valid < 1) {
                viewer->set_tracking_result(TrackingResult::LOST);
                return;
            }

            linearSystem.f /= static_cast<float>(linearSystem.valid);
            if (linearSystem.f < f) {
                f = linearSystem.f;
                // hessian_good = linearSystem.hessian;
                // nabla_good = linearSystem.nabla;
                old_pose = pose_good;
                // valid_points_good = linearSystem.valid;
                lamdba /= lamdba_scale_;
            } else {
                pose_good = old_pose;
                lamdba *= lamdba_scale_;
                continue;
            }

            // 处理几何环境退化，表现为hessian矩阵分解，特征值接近0
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> saes(linearSystem.hessian);

            // std::cout << "hessian matrix:\n" << linearSystem.hessian << std::endl;
            Eigen::Vector<float, 6> eigenvalues = saes.eigenvalues();
            for (int i = 0; i < 6; ++i)
                if (eigenvalues(i) < 1e-7) {
                    viewer->set_tracking_result(TrackingResult::POOR);
                    return;
                }

            linearSystem.hessian += lamdba * Eigen::Matrix<float, 6, 6>::Identity();
            Eigen::Vector<float, 6> delta = linearSystem.hessian.ldlt().solve(linearSystem.nabla);
            applyDelta(pose_good, delta);
        }
    }

    // updateTrackingingQuality(viewer, valid_points_good);

    if (viewer->get_tracking_result() == TrackingResult::GOOD) {
        last_delta_ = last_pose_.inverse() * pose_good;
        last_pose_ = pose_good;
        initial_pose = pose_good;
    }
}

// void DepthTracker::updateTrackingingQuality(std::shared_ptr<Viewer> viewer, int nVaildPoints) {}
}  // namespace ScanReconstruction
