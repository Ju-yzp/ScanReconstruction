#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include <utils.h>
#include <Eigen/Core>
#include <array>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

using Histogram = std::array<int, 11>;

using Keyframe = std::pair<std::vector<Histogram>, cv::Mat>;

class KeyframeManager {
public:
    KeyframeManager(
        int width, int height, Eigen::MatrixXi area, Eigen::VectorXf gaussian_kernel,
        float entropy_threshold);

    bool hasEnoughImformation(
        std::vector<Eigen::Vector3f>& normals, std::vector<Histogram>& histograms);

    void updateHistogram(Eigen::Vector3f& normal, Histogram& histogram);

    void computeHistogram(
        std::vector<Eigen::Vector3f>& normals, Histogram& histogram, int start_row, int start_col,
        int end_row, int end_col);

    float computeEntropy(Histogram& histogram);

    bool isSimilarity(
        std::vector<Histogram>& lHistograms, std::vector<Histogram>& rHistograms,
        float& similarity);

private:
    // 记录图像的高和宽，检查边界情况
    int height_, width_;

    // 每个patch的区域,其中中心patch的面积最大，高斯权重也是最大的
    Eigen::MatrixXi area_;

    // 高斯
    Eigen::VectorXf gaussian_kernel_;

    // 只有超过这个阈值才能认为有机会成为关键帧
    float entropy_threshold_;

    // 相似度阈值
    float similarty_threshold_{0.80f};
};
#endif
