#ifndef TRACKER_H_
#define TRACKER_H_

// tracker
#include <Tracker/cameraParams.h>

// opencv
#include <opencv2/core/mat.hpp>

// sophus
#include <sophus/se3.hpp>

struct View {
    cv::Mat depth;
    cv::Mat normal;
    cv::Mat points;
    Sophus::SE3f pose_d;
    Sophus::SE3f scenePose;
};

class Tracker {
public:
    enum class SolveType { GOOG = 1, POOR = 0 };

    Tracker(
        int nLevel, int max_iteration, float spaceThreshCoarse, float spaceThreshFine,
        const CalibrationParams calibrationParams);

    void track(View* view);

private:
    void preparePyramid(View* view);

    int computeGandH(
        int id, Eigen::Matrix4f approxPose, float* f, Eigen::Matrix<float, 6, 6>* hessian,
        Eigen::Vector<float, 6>* nabla);

    // 点云图像金字塔
    std::vector<cv::Mat> pointCloudyramid_;

    // 法向量图像金字塔
    std::vector<cv::Mat> normalPyramid_;

    // 深度图像金字塔
    std::vector<cv::Mat> depthPyramid_;

    // 相机参数金字塔
    std::vector<CalibrationParams> calibrationParamsPyramid_;

    // 迭代数次
    std::vector<int> iterationLevels_;

    // 空间阈值
    std::vector<float> spaceThreshold_;

    // 场景位姿
    Eigen::Matrix4f scenePose_;
};

#endif
