// surface_reconstruction
#include <pixelUtils.h>
#include <tracker.h>

// eigen
#include <Eigen/Core>

// cpp
#include <algorithm>
#include <limits>

// opencv
#include <memory>
#include <opencv2/opencv.hpp>

#include <sophus/se3.hpp>
namespace surface_reconstruction {

Tracker::Tracker(
    std::shared_ptr<Settings> settins, int nPyramidLevel, int maxNIteration, int minNIteration,
    float maxSpaceThreshold, float minSpaceThreshold)
    : settings_(settins), nPyramidLevel_(nPyramidLevel) {
    rgbPyramid_.resize(nPyramidLevel);
    depthPyramid_.resize(nPyramidLevel);
    rgbIntrinsicsPyramid_.reserve(nPyramidLevel);
    depthIntrinsicsPyramid_.reserve(nPyramidLevel);
    pointcloudPyramid_.resize(nPyramidLevel);
    normalPyramid_.resize(nPyramidLevel);
    nIterationPyramid_.resize(nPyramidLevel);
    spaceThresholds_.reserve(nPyramidLevel);

    float iterStep = (float)(maxNIteration - minNIteration) / (float)(nPyramidLevel - 1);
    float spaceStep = (maxSpaceThreshold - minSpaceThreshold) / (float)(nPyramidLevel - 1);
    for (int i{0}; i < nPyramidLevel; ++i) {
        nIterationPyramid_[i] = maxNIteration - iterStep * i;
        spaceThresholds_[i] = maxSpaceThreshold - spaceStep * i;
    }
}

void Tracker::track(std::shared_ptr<View> view, std::shared_ptr<TrackingState> trackingState) {
    prepare(view, trackingState);

    Eigen::Matrix<float, 6, 6> hessian_good = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Vector<float, 6> nabla_good = Eigen::Vector<float, 6>::Zero();
    int nVaildPoints_good{0};

    for (int level = nPyramidLevel_ - 1; level >= 0; --level) {
        // 相机到世界坐标系
        Eigen::Matrix4f approxPose = trackingState->get_current_camera_in_localmap().inverse();
        Eigen::Matrix4f old_pose = trackingState->get_current_camera_in_localmap().inverse();
        float old_f = std::numeric_limits<float>::infinity();
        float lamdba{1.0f};

        for (int i = 0; i < nIterationPyramid_[level]; ++i) {
            float local_f = {0.0f};
            Eigen::Matrix<float, 6, 6> local_hessian = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Vector<float, 6> local_nabla = Eigen::Vector<float, 6>::Zero();
            int local_nVaildPoints{0};

            computeHessianAndGradient(
                level, local_hessian, local_nabla, local_f, local_nVaildPoints, view, approxPose,
                trackingState);

            if (local_nVaildPoints > settings_->minNVaildPoints) {
                local_hessian /= (float)local_nVaildPoints;
                local_nabla /= (float)local_nVaildPoints;
                local_f /= (float)local_nVaildPoints;
            } else
                local_f = std::numeric_limits<float>::infinity();

            if (local_nVaildPoints < 1 || local_f >= old_f) {
                trackingState->set_current_camera_in_localmap(old_pose.inverse());
                approxPose = trackingState->get_current_camera_in_localmap().inverse();
                lamdba *= settings_->lamdbaScale;
                continue;
            } else {
                old_f = local_f;
                old_pose = trackingState->get_current_camera_in_localmap().inverse();
                hessian_good = local_hessian;
                nabla_good = local_nabla;
                nVaildPoints_good = local_nVaildPoints;
                lamdba /= settings_->lamdbaScale;
            }

            local_hessian = hessian_good + Eigen::Matrix<float, 6, 6>::Identity() * lamdba;
            local_nabla = nabla_good;

            Eigen::Matrix<float, 1, 6> delta = Eigen::Matrix<float, 1, 6>::Identity();
            computeDelta(local_hessian, local_nabla, delta);
            applyDelta(approxPose, delta);
            trackingState->set_current_camera_in_localmap(approxPose.inverse());
        }
    }
}

void Tracker::prepare(std::shared_ptr<View> view, std::shared_ptr<TrackingState> trackingState) {
    // 获取图像金字塔以及相机内参金字塔的第一层
    rgbPyramid_[0] = view->rgb;
    depthPyramid_[0] = view->depth;
    normalPyramid_[0] = trackingState->get_normals();
    pointcloudPyramid_[0] = trackingState->get_pointclouds();

    if (rgbIntrinsicsPyramid_.empty()) {
        rgbIntrinsicsPyramid_.emplace_back(view->calibrationParams.rgb);
        depthIntrinsicsPyramid_.emplace_back(view->calibrationParams.depth);
        for (int i = 1; i < nPyramidLevel_; ++i) {
            rgbIntrinsicsPyramid_.emplace_back(rgbIntrinsicsPyramid_[i - 1].subIntrisic());
            depthIntrinsicsPyramid_.emplace_back(depthIntrinsicsPyramid_[i - 1].subIntrisic());
        }
    }

    for (int i = 1; i < nPyramidLevel_; ++i) {
        // 彩色图像金字塔下采样
        // cv::pyrDown(rgbPyramid_[i - 1], rgbPyramid_[i]);
        // 深度和法向量图像金字塔下采样
        Eigen::Vector2i mapSize{depthPyramid_[i - 1].rows, depthPyramid_[i - 1].cols};
        filterSubsampleWithHoles(depthPyramid_[i - 1], depthPyramid_[i], mapSize);
        filterSubsampleWithHoles(pointcloudPyramid_[i - 1], pointcloudPyramid_[i], mapSize);
        filterSubsampleWithHoles(normalPyramid_[i - 1], normalPyramid_[i], mapSize, true);
    }
}

inline float rho(float r, float huber_r) {
    float tmp = std::abs(r) - huber_r;
    tmp = std::max(tmp, 0.0f);
    return r * r - tmp * tmp;
}

#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

#ifndef CLAMP
#define CLAMP(x, a, b) MAX((a), MIN((b), (x)))
#endif

inline float rho_deriv(float r, float huber_b) { return 2.0f * CLAMP(r, -huber_b, huber_b); }

inline float rho_deriv2(float r, float huber_b) { return fabs(r) < huber_b ? 2.0f : 0.0f; }

Eigen::Matrix3f skew(const Eigen::Vector3f v) {
    Eigen::Matrix3f m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
void Tracker::computeHessianAndGradient(
    int id, Eigen::Matrix<float, 6, 6>& hessian, Eigen::Vector<float, 6>& nabla, float& f,
    int& nVaildPoints, std::shared_ptr<View> view, Eigen::Matrix4f approxPose,
    std::shared_ptr<TrackingState> trackingState) {
    int rows = depthPyramid_[id].rows;
    int cols = depthPyramid_[id].cols;
    float* depth = (float*)depthPyramid_[id].data;
    const auto normalMap = normalPyramid_[id];
    const auto pointsMap = pointcloudPyramid_[id];

    Eigen::Matrix4f generatePose = trackingState->get_generate_camera_in_localmap();
    Eigen::Matrix3f k_inv = depthIntrinsicsPyramid_[id].k_inv;
    Eigen::Matrix3f k = k_inv.inverse();

    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    Eigen::Matrix<float, 1, 6> A;
    const float spaceThreshold = spaceThresholds_[id];

    for (int y = 1; y < rows - 1; ++y) {
        int offset = y * cols;
        for (int x = 1; x < cols - 1; ++x) {
            Eigen::Vector<float, 6> local_hessian;
            Eigen::Vector<float, 6> local_nabla;
            float local_f{0.0f};

            float b;
            float currentDepth = depth[x + offset];
            if (currentDepth < 1e-4) continue;
            Eigen::Vector3f currentPointcloud(x, y, 1.0f);
            currentPointcloud = currentDepth * k_inv * currentPointcloud;

            Eigen::Vector4f point_ =
                approxPose * (Eigen::Vector4f() << currentPointcloud, 1.0f).finished();
            Eigen::Vector3f point_in_last_view =
                k * (generatePose * approxPose *
                     (Eigen::Vector4f() << currentPointcloud, 1.0f).finished())
                        .head(3);

            if (point_in_last_view(2) < 1e-4) continue;

            point_in_last_view /= point_in_last_view(2);

            if (point_in_last_view(0) < 1 || point_in_last_view(0) > cols - 1 ||
                point_in_last_view(1) < 1 || point_in_last_view(1) > rows - 1)
                continue;
            Eigen::Vector4f point =
                interpolateBilinear_withHoles(pointsMap, point_in_last_view.head(2), cols);
            if (point(3) < 0.0f) continue;
            Eigen::Vector4f normal =
                interpolateBilinear_withHoles(normalMap, point_in_last_view.head(2), cols);
            if (normal(3) < 0.0f) continue;
            b = normal.head(3).dot(point_.head(3) - point.head(3));

            float depthWeight =
                MAX(0.0f,
                    1.0f - (currentDepth - viewFrustum_min) / (viewFrustum_max - viewFrustum_min));

            A.block<1, 3>(0, 0) = -normal.head(3).transpose() * skew(point_.head(3));
            A.block<1, 3>(0, 3) = normal.head(3).transpose();
            local_f = rho(b, spaceThreshold) * pow(depthWeight, 2);
            f += local_f;

            // hessian += rho_deriv2(b, spaceThreshold) * A.transpose() * A * pow(depthWeight, 2);
            // nabla += -A.transpose()* local_f * pow(depthWeight, 2) * rho_deriv(b,spaceThreshold);
            hessian += rho_deriv2(b, spaceThreshold) * A.transpose() * A * pow(depthWeight, 2);
            nabla +=
                -A.transpose() * depthWeight * rho_deriv(b, spaceThreshold) * pow(depthWeight, 2);
            nVaildPoints++;
        }
    }
}

void Tracker::computeDelta(
    Eigen::Matrix<float, 6, 6> hessian, Eigen::Vector<float, 6> nabla,
    Eigen::Matrix<float, 1, 6>& delta) {
    delta = hessian.ldlt().solve(nabla);
}

void Tracker::applyDelta(Eigen::Matrix4f& pose, Eigen::Matrix<float, 1, 6> delta) {
    Eigen::Vector3f dw = delta.head<3>();
    Eigen::Vector3f dv = delta.tail<3>();
    Eigen::Matrix3f dR = (Eigen::AngleAxisf(dw.norm(), dw.normalized())).toRotationMatrix();
    pose.block(0, 0, 3, 3) = dR * pose.block(0, 0, 3, 3);
    pose.block(0, 3, 3, 1) = dR * pose.block(0, 3, 3, 1) + dv;
}
void Tracker::updateQualityOfTracking() {}
}  // namespace surface_reconstruction
