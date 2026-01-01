#ifndef TRACKER_H_
#define TRACKER_H_

#include <tbb/global_control.h>

// eigen
#include <Eigen/Core>
#include <Eigen/Eigen>

// cpp
#include <limits>
#include <tuple>

// opencv
#include <opencv2/opencv.hpp>

struct View {
    View(int height_, int width_) : width(width_), height(height_) {
        Eigen::Vector3f initiali_value(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
        depth.resize((size_t)(height_ * width_), initiali_value);
        prev_normal.resize((size_t)(height_ * width_), initiali_value);
        prev_depth.resize((size_t)(height_ * width_), initiali_value);
    }
    std::vector<Eigen::Vector3f> depth;
    std::vector<Eigen::Vector3f> prev_depth;
    std::vector<Eigen::Vector3f> prev_normal;

    void swapDepth() { depth.swap(prev_depth); }

    int width;
    int height;
};

enum class TrackingResult { TRACKING_GOOD = 0, TRACKING_FAILD = 1, TRACKING_POOR = 2 };

class Tracker {
public:
    Tracker(
        cv::Mat k, cv::Mat d, cv::Size2i imgSize, int max_num_threads, int max_num_iterations,
        int min_nVaildPoints, float initial_lamdba, float lamdba_scale, float depth_scale,
        float space_threshold);

    void undistortion(cv::Mat& origin_depth, View& view);

    Eigen::Matrix4f track(View& view);

    std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Vector<float, 6>, float, int> buildLinearSystem(
        const View& view, const Eigen::Matrix4f& current_pose, const Eigen::Matrix4f& prev_pose,
        Eigen::Matrix3f k, int width, int height);

    static void applyDelta(Eigen::Matrix4f& pose, Eigen::Vector<float, 6> delta);

    static void computeNormalMap(View& view);

    static void transform(View& view, const Eigen::Matrix4f global_pose);

    Eigen::Matrix4f get_pose() { return global_pose_; }

    void set_global_pose(Eigen::Matrix4f new_pose) {
        last_delta_ = global_pose_.inverse() * new_pose;
        global_pose_ = new_pose;
    }

    // void evaluateTrackingQuality(View& view, TrackingResult& trackingResult);

private:
    struct LinearSystem {
        Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> nabla = Eigen::Vector<float, 6>::Zero();
        float f{0.0f};
        int VaildPoint{0};
    };

    cv::Mat mapX_, mapY_;

    Eigen::Matrix3f k_;

    int max_num_iterations_;

    int height_, width_;

    int min_nVaildPoints_;

    float initial_lamdba_;

    float lamdba_scale_;

    float depth_scale_;

    float space_threshold_;

    std::vector<Eigen::Vector3f> rayMap_;

    Eigen::Matrix4f global_pose_;

    Eigen::Matrix4f last_delta_;
};
#endif
