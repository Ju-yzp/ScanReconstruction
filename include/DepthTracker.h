#ifndef DEPTH_TRACKER_H_
#define DEPTH_TRACKER_H_

#include <GlobalSettings.h>
#include <Viewer.h>

// sophus
#include <memory>
#include <sophus/se3.hpp>

namespace ScanReconstruction {
class DepthTracker {
public:
    DepthTracker(std::shared_ptr<GlobalSettings> global_settings);

    void reset();

    void track(std::shared_ptr<Viewer> viewer, Eigen::Matrix4f& initial_pose);

    // void updateTrackingingQuality(std::shared_ptr<Viewer> viewer, int nVaildPoints);

    void applyDelta(Eigen::Matrix4f& pose, const Eigen::Vector<float, 6>& delta);

    Eigen::Matrix4f get_initial_guess_pose() { return last_pose_ * last_delta_; }

private:
    struct LinearSystem {
        Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> nabla = Eigen::Vector<float, 6>::Zero();
        float f = 0.0f;
        int valid = 0;
    };

    struct CameraParams {
        Eigen::Matrix3f k;
        int width, height;
    };

    LinearSystem buuildLinearSystem(
        const Points& current_points, const Points& prev_points, const Normals& prev_normals,
        const Eigen::Matrix4f& current_pose, const Eigen::Matrix4f& prev_pose,
        const CameraParams camera_params, const float space_threshold);

    void prepare(std::shared_ptr<Viewer> viewer);

    size_t levels_;

    Eigen::Matrix4f last_pose_;
    Eigen::Matrix4f last_delta_;

    std::vector<Points> current_points_pyramids_;
    std::vector<Points> prev_points_pyramids_;
    std::vector<Normals> prev_normals_pyramids_;
    std::vector<float> space_threhold_pyramids_;
    std::vector<CameraParams> camera_params_pyramids_;
    std::vector<int> iterations_pyramids_;

    int min_num_valid_points_;

    float initial_lamdba_;
    float lamdba_scale_;
};
}  // namespace ScanReconstruction
#endif  // DEPTH_TRACKER_H_
