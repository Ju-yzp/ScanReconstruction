#include "../../VisualisationEngine/include/VisualisationEngine/VisualisationEngine.h"

#include <Eigen/Core>
#include <cmath>

void VisualisationEngine::processFrame(
    std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view) {
    // TODO:深度图使用CV_32F格式进行储存
    float* depth = reinterpret_cast<float*>(view->depth.data);

    int rows = view->depth.rows;
    int cols = view->depth.cols;

    Eigen::Vector3f point_in_camera;
    Eigen::Matrix3f inv_depth = view->calibrationParams.depth.k_inv;
    Eigen::Matrix4f pose_inv;

    float norm{0.0f};  // 点云在相机坐标系下的模长
    float depth_measure;
    float mu;                // 截断距离，在这个距离外的都会被设置为-1/1
    float oneOverVoxelSize;  // 一米内存在的体素个数
    int nstep;
    Eigen::Vector3f direction = Eigen::Vector3f::Zero();

    // 视锥体范围
    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            depth_measure = depth[y * cols + x];

            // 无效深度值
            if (depth_measure < 1e-4 || (depth_measure - mu) < viewFrustum_min ||
                (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max)
                continue;

            // 获取在depth下的点云数据
            point_in_camera(2) = 1.0f;
            point_in_camera(0) = x;
            point_in_camera(1) = y;
            point_in_camera = inv_depth * point_in_camera * depth_measure;

            norm = point_in_camera.norm();

            // 获取从该点云延伸的截断线段的起始点和终点
            Eigen::Vector3f point_s =
                (pose_inv *
                 (Eigen::Vector4f() << point_in_camera * (1.0f - mu / norm), 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;
            Eigen::Vector3f point_e =
                (pose_inv *
                 (Eigen::Vector4f() << point_in_camera * (1.0f + mu / norm), 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            direction = point_e - point_s;
            nstep = (int)ceil(2.0f * direction.norm());
            direction /= (float)(nstep - 1);

            for (int i{0}; i < nstep; ++i) {
                int hashId = VoxelBlockHash::getHashIndex(
                    Eigen::Vector3i(point_s(0), point_s(1), point_s(2)));

                point_s += direction;
            }
        }
    }
}

void VisualisationEngine::raycast(
    std::shared_ptr<VoxelBlockHash> vbh, RenderState* renderState,
    RGBDCalibrationParams* cameraParams, const Eigen::Matrix4f inv_m) {
    int rows = renderState->raycastResult.rows;
    int cols = renderState->raycastResult.cols;

    float mu;                // tsdf截断距离
    float oneOverVoxelSize;  // 一米所能容纳的体素个数
    float stepScale = mu * oneOverVoxelSize;

    Eigen::Vector3f imagePoint;

    // 视锥体范围
    float viewFrustum_max = cameraParams->viewFrustum_max;
    float viewFrustum_min = cameraParams->viewFrustum_min;

    // 重投影
    Eigen::Matrix3f k_inv_depth = cameraParams->depth.k_inv;

    // 光线投射的方向
    Eigen::Vector3f rayDirection = Eigen::Vector3f::Zero();

    // 每次前进或者后退的步长，如果sdf小于0,那么根据tsdf的计算公式我们可以知道明确的
    float stepLen;
#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // 图像齐次坐标
            imagePoint(2) = 1.0f;
            imagePoint(0) = x;
            imagePoint(1) = y;

            Eigen::Vector3f point_s =
                (inv_m *
                 (Eigen::Vector4f() << k_inv_depth * imagePoint * viewFrustum_min, 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            Eigen::Vector3f point_e =
                (inv_m *
                 (Eigen::Vector4f() << k_inv_depth * imagePoint * viewFrustum_max, 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            rayDirection = (point_e - point_s).normalized();

            u_char sdf_v;

            while (true) {  // 沿着光线出发，在这个范围内寻找等值面

                if (sdf_v < 0.0f) {  // 此时需要回退，同时步长乘以
                }
            }
        }
    }
}
