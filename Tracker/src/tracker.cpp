#include <Tracker/pixelUtils.h>
#include <Tracker/tracker.h>

// cpp
#include <cassert>
#include <cmath>
#include <limits>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <sophus/se3.hpp>

constexpr int MIN_VALID_POINTS = 100;

#ifndef CLAMP
#define CLAMP(x, a, b) std::max((a), std::min((b), (x)))
#endif

inline float rho(float r, float huber_b) {
    float tmp = fabs(r) - huber_b;
    tmp = MAX(tmp, 0.0f);
    return r * r - tmp * tmp;
}

inline float rho_deriv(float r, float huber_b) { return 2.0f * CLAMP(r, -huber_b, huber_b); }

inline float rho_deriv2(float r, float huber_b) { return fabs(r) < huber_b ? 2.0f : 0.0f; }

Tracker::Tracker(
    int nLevel, int max_iteration, float spaceThreshCoarse, float spaceThreshFine,
    const CalibrationParams calibrationParams) {
    depthPyramid_.resize(nLevel);
    normalPyramid_.resize(nLevel);
    pointCloudyramid_.resize(nLevel);
    iterationLevels_.resize(nLevel);
    calibrationParamsPyramid_.resize(nLevel);
    spaceThreshold_.resize(nLevel);

    calibrationParamsPyramid_[0] = calibrationParams;

    // 初始化每一层的迭代次数，以及对应的相机内参
    float step = (float)max_iteration / (float)nLevel;
    iterationLevels_[0] = max_iteration;

    for (int i{1}; i < nLevel; ++i) {
        iterationLevels_[i] = (nLevel - i) * step;
        calibrationParamsPyramid_[i] = calibrationParamsPyramid_[i - 1].subCameraParams();
    }

    if (spaceThreshCoarse >= 0.0f && spaceThreshFine >= 0.0f) {
        float step = (float)(spaceThreshCoarse - spaceThreshFine) / (float)(nLevel - 1);
        float val = spaceThreshCoarse;
        for (int levelId = nLevel - 1; levelId >= 0; levelId--) {
            this->spaceThreshold_[levelId] = val;
            val -= step;
        }
    }
}

void Tracker::preparePyramid(View* view) {
    // 对原始深度图像做一次双边滤波，平滑一下，降低噪声干扰
    int d = 7;
    double sigmaColor = 0.05;
    double sigmaSpace = 5.0;

    cv::Mat processed_depth;
    cv::bilateralFilter(view->depth, processed_depth, d, sigmaColor, sigmaSpace);

    scenePose_ = view->scenePose.matrix();

    // 设置每个金字塔的第一层
    depthPyramid_[0] = processed_depth;
    normalPyramid_[0] = view->normal;
    pointCloudyramid_[0] = view->points;

    // 进行下采样
    for (int i{1}; i < depthPyramid_.size(); ++i) {
        filterSubsampleWithHoles(&depthPyramid_[i - 1], &depthPyramid_[i]);
        filterSubsampleWithHoles(&normalPyramid_[i - 1], &normalPyramid_[i]);
        filterSubsampleWithHoles(&pointCloudyramid_[i - 1], &pointCloudyramid_[i]);
    }
}

void Tracker::track(View* view) {
    preparePyramid(view);

    int nVaildPoints_good{0};
    Eigen::Matrix<float, 6, 6> hessian_good = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Vector<float, 6> nabla_good = Eigen::Vector<float, 6>::Zero();

    // 从粗到细进行迭代
    for (int levelId = depthPyramid_.size() - 1; levelId >= 0; --levelId) {
        Eigen::Matrix4f approxPose = view->pose_d.matrix();
        Sophus::SE3f lastKownPose = view->pose_d;

        float f_old = std::numeric_limits<float>::infinity();
        float lambda = 1.0f;

        for (int i{0}; i < iterationLevels_[levelId]; ++i) {
            Eigen::Matrix<float, 6, 6> hessian = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Vector<float, 6> nabla = Eigen::Vector<float, 6>::Zero();
            int nVaildPoints{0};
            float f{0.0f};

            nVaildPoints = computeGandH(levelId, approxPose, &f, &hessian, &nabla);
            if (nVaildPoints > MIN_VALID_POINTS) {
                hessian /= (float)nVaildPoints;
                nabla /= (float)nVaildPoints;
                f /= (float)nVaildPoints;
            } else
                f = std::numeric_limits<float>::infinity();

            if (nVaildPoints < 1 || (f > f_old)) {
                lambda *= 10.0f;
                view->pose_d = lastKownPose;
                approxPose = view->pose_d.matrix();
            } else {
                f_old = f;
                lastKownPose = view->pose_d;
                hessian_good = hessian;
                nabla_good = nabla;
                nVaildPoints_good = nVaildPoints;
                lambda /= 10.0f;
                std::cout << f << std::endl;
            }

            Eigen::Matrix<float, 6, 6> A =
                (hessian_good + lambda * Eigen::Matrix<float, 6, 6>::Identity());
            Eigen::Matrix<float, 6, 6> A_inv =
                A.llt().solve(Eigen::Matrix<float, 6, 6>::Identity());
            Eigen::Vector<float, 6> delta = A_inv * nabla_good;

            Sophus::SE3f T_inc = Sophus::SE3f::exp(delta);
            approxPose = T_inc.matrix() * approxPose;

            view->pose_d = Sophus::SE3f(approxPose);
            view->pose_d.normalize();
            approxPose = view->pose_d.matrix();
        }
    }
}

int Tracker::computeGandH(
    int id, Eigen::Matrix4f approxPose, float* f, Eigen::Matrix<float, 6, 6>* hessian,
    Eigen::Vector<float, 6>* nabla) {
    int nVaildPoints{0};

    cv::Mat& depthMap = depthPyramid_[id];
    cv::Mat& pointMap = pointCloudyramid_[id];
    cv::Mat& normalMap = normalPyramid_[id];

    for (int y{1}; y < depthMap.rows - 1; ++y) {
        CalibrationParams calibrationParams = calibrationParamsPyramid_[id];

        auto approxInvPose = approxPose.inverse();
        for (int x{1}; x < depthMap.cols - 1; ++x) {
            Eigen::Matrix<float, 6, 6> local_hession = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Vector<float, 6> local_nabla = Eigen::Vector<float, 6>::Zero();
            Eigen::Vector<float, 6> A = Eigen::Vector<float, 6>::Zero();
            float local_f{0.0f};

            // 遇到无效值跳过
            float depth = depthMap.at<float>(y, x);
            if (depth <= 1e-6) continue;

            // 使用近似位姿,转换至全局坐标系下(低速运动,两帧的位姿变化不大)
            auto p = calibrationParams.computePointCloud(depth, cv::Vec2i{x, y});
            Eigen::Vector4f tmp3Dpoint = approxInvPose * Eigen::Vector4f{p(0), p(1), p(2), p(3)};

            // 将坐标再次转换至渲染的相机坐标系下
            tmp3Dpoint = scenePose_ * tmp3Dpoint;

            if (tmp3Dpoint(2) <= 1e-5) continue;

            // 重投影至图像上
            Eigen::Vector2f imagePoint = calibrationParams.reproject(tmp3Dpoint);

            // 获取对应的点云
            Eigen::Vector4f cur3Dpoint = interpolateBilinear_withHoles(pointMap, imagePoint);

            if (cur3Dpoint(2) < 1e-6) continue;

            Eigen::Vector4f diff = cur3Dpoint - tmp3Dpoint;

            float depthWeight = std::pow(
                std::max(
                    0.0f, 1.0f - (depth - calibrationParams.viewFrustum_min) /
                                     (calibrationParams.viewFrustum_max -
                                      calibrationParams.viewFrustum_min)),
                2);

            // 获取对应的法向量
            Eigen::Vector4f cur3Dnormal = interpolateBilinear_withHoles(normalMap, imagePoint);

            float b = (diff.block(0, 0, 3, 1).transpose() * cur3Dnormal.block(0, 0, 3, 1)).sum();

            A(0) = tmp3Dpoint(2) * cur3Dnormal(1) - tmp3Dpoint(1) * cur3Dnormal(2);
            A(1) = -tmp3Dpoint(2) * cur3Dnormal(0) + tmp3Dpoint(0) * cur3Dnormal(2);
            A(2) = tmp3Dpoint(1) * cur3Dnormal(0) - tmp3Dpoint(0) * cur3Dnormal(1);
            A(3) = cur3Dnormal(0);
            A(4) = cur3Dnormal(1);
            A(5) = cur3Dnormal(2);

            local_f = rho(b, spaceThreshold_[id]);
            for (int r{0}; r < 6; ++r) {
                local_nabla[r] = rho_deriv(b, spaceThreshold_[id]) * depthWeight * A[r];
                for (int k{0}; k <= r; ++k) {
                    local_hession(r, k) =
                        rho_deriv2(b, spaceThreshold_[id]) * depthWeight * A[r] * A(k);
                    if (r != k) {
                        local_hession(k, r) = local_hession(r, k);
                    }
                }
            }

            nVaildPoints++;
            *hessian += local_hession;
            *nabla += local_nabla;
            *f += local_f;
        }
    }

    return nVaildPoints;
}
