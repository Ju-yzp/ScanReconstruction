#include <keyframe.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>

#include <keyframe.h>
#include <Eigen/Dense>
#include <stdexcept>

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

KeyframeManager::KeyframeManager(
    int width, int height, Eigen::MatrixXi area, Eigen::VectorXf gaussian_kernel,
    float entropy_threshold)
    : height_(height),
      width_(width),
      area_(area),
      gaussian_kernel_(gaussian_kernel),
      entropy_threshold_(entropy_threshold) {}

bool KeyframeManager::hasEnoughImformation(
    std::vector<Eigen::Vector3f>& normals, std::vector<Histogram>& histograms) {
    size_t num = (size_t)area_.rows();
    histograms.clear();
    histograms.resize(num);

    Eigen::VectorXf entropy_list = Eigen::VectorXf::Zero((int)num);

    tbb::parallel_for(size_t(0), num, [&](size_t i) {
        Eigen::Vector4i bound = area_.row((int)i);
        computeHistogram(normals, histograms[i], bound[0], bound[1], bound[2], bound[3]);
        entropy_list[(int)i] = computeEntropy(histograms[i]);
    });

    float total_entropy = entropy_list.transpose() * gaussian_kernel_;
    return total_entropy > entropy_threshold_;
}

void KeyframeManager::computeHistogram(
    std::vector<Eigen::Vector3f>& normals, Histogram& histogram, int start_row, int start_col,
    int end_row, int end_col) {
    if (start_col < 0 || start_row < 0 || end_col >= width_ || end_row >= height_) {
        std::cerr << "Boundary Error: "
                  << "R:" << start_row << "-" << end_row << ", "
                  << "C:" << start_col << "-" << end_col << std::endl;
        throw std::runtime_error("Patch coordinates out of image bounds!");
    }

    for (int y = start_row; y <= end_row; ++y) {
        int offset = y * width_;
        for (int x = start_col; x <= end_col; ++x) {
            Eigen::Vector3f& normal = normals[size_t(offset + x)];
            if (std::isnan(normal(0))) continue;
            updateHistogram(normal, histogram);
        }
    }
}

void KeyframeManager::updateHistogram(Eigen::Vector3f& normal, Histogram& histogram) {
    float cos_theta = normal.z();

    if (std::abs(cos_theta) > 0.866f)
        if (cos_theta > 0)
            histogram[0]++;
        else
            histogram[1]++;
    else {
        float phi = std::atan2(normal.y(), normal.x());
        float normalized_phi = (phi + M_PIf) / (2.0f * M_PIf);
        int bin_idx = 2 + static_cast<int>(normalized_phi * 8.999f);

        histogram[(size_t(bin_idx))]++;
    }
}

float KeyframeManager::computeEntropy(Histogram& histogram) {
    int total_num{0};
    for (size_t i = 0; i < 11; ++i) total_num += histogram[i];

    float entropy{0.0f};
    for (size_t i = 0; i < 11; ++i) {
        int num = histogram[i];
        if (num == 0) continue;
        float pi = (float)num / (float)total_num;
        entropy += -pi * std::log2(pi);
    }

    return entropy;
}

bool KeyframeManager::isSimilarity(
    std::vector<Histogram>& lHistograms, std::vector<Histogram>& rHistograms, float& similarity) {
    if (lHistograms.size() != rHistograms.size()) return false;

    float score = 0.0f;
    float weight_sum = 0.0f;

    for (int i = 0; i < (int)lHistograms.size(); ++i) {
        const auto& histL = lHistograms[(size_t)i];
        const auto& histR = rHistograms[(size_t)i];

        float intersection = 0.0f;
        float sumL = 0.0f;
        for (size_t j = 0; j < 11; ++j) {
            intersection += std::min(static_cast<float>(histL[j]), static_cast<float>(histR[j]));
            sumL += static_cast<float>(histL[j]);
        }

        float patch_score = (sumL > 0) ? (intersection / sumL) : 0.0f;

        score += gaussian_kernel_[i] * patch_score;
        weight_sum += gaussian_kernel_[i];
    }

    similarity = score / weight_sum;

    return similarity > similarty_threshold_;
}
