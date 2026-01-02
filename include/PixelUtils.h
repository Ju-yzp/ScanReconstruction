#ifndef PIXEL_UTILS_H_
#define PIXEL_UTILS_H_

#include <Types.h>

// tbb
#include <tbb/blocked_range2d.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <limits>

namespace ScanReconstruction {
inline void filterSubsampleWithHoles(
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& input,
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& output, int height,
    int width, bool isNormal = false) {
    int output_rows = height / 2;
    int output_cols = width / 2;

    if (output.size() != size_t(output_cols * output_rows))
        throw std::runtime_error("Output size is not correct");

    Eigen::Vector3f* input_ptr = input.data();

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, output_rows, 0, output_cols),
        [&](const tbb::blocked_range2d<int>& r) {
            for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                Eigen::Vector3f* output_ptr = output.data() + (y * output_cols);
                for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                    Eigen::Vector3f pixel_in[4], pixel_out = Eigen::Vector3f::Zero();
                    pixel_in[0] = input_ptr[y * 2 * width + x * 2];
                    pixel_in[1] = input_ptr[(y * 2 + 1) * width + x * 2 + 1];
                    pixel_in[2] = input_ptr[(y * 2 + 1) * width + x * 2];
                    pixel_in[3] = input_ptr[(y * 2) * width + x * 2 + 1];

                    int nVaildPoints{0};
                    for (int k{0}; k < 4; ++k)
                        if (!std::isnan(pixel_in[k](0))) {
                            pixel_out += pixel_in[k];
                            ++nVaildPoints;
                        }

                    if (nVaildPoints == 0) {
                        pixel_out(0) = std::numeric_limits<float>::quiet_NaN();

                        output_ptr[x] = pixel_out;
                        continue;
                    }

                    pixel_out /= (float)nVaildPoints;
                    if (isNormal) {
                        float norm = pixel_out.norm();
                        pixel_out /= norm;
                    }
                    output_ptr[x] = pixel_out;
                }
            }
        });
}

inline Eigen::Vector3f interpolateBilinear_withHoles(
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& input,
    Eigen::Vector2f coorinate, int cols) {
    Eigen::Vector2i imgPoint((int)floor(coorinate(0)), (int)floor(coorinate(1)));
    Eigen::Vector3f* input_ptr = input.data();
    auto a = input_ptr[imgPoint(0) + imgPoint(1) * cols];
    auto b = input_ptr[imgPoint(0) + 1 + imgPoint(1) * cols];
    auto c = input_ptr[imgPoint(0) + (imgPoint(1) + 1) * cols];
    auto d = input_ptr[imgPoint(0) + 1 + (imgPoint(1) + 1) * cols];
    Eigen::Vector3f result;
    Eigen::Vector2f delta{coorinate(0) - (float)imgPoint(0), coorinate(1) - (float)imgPoint(1)};

    if (std::isnan(a(0)) || std::isnan(b(0)) || std::isnan(c(0)) || std::isnan(d(0))) {
        result(0) = std::numeric_limits<float>::quiet_NaN();
        return result;
    }

    result(0) = a(0) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(0) * delta(0) * (1.0f - delta(1)) +
                c(0) * (1.0f - delta(0)) * delta(1) + d(0) * delta(0) * delta(1);

    result(1) = a(1) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(1) * delta(0) * (1.0f - delta(1)) +
                c(1) * (1.0f - delta(0)) * delta(1) + d(1) * delta(0) * delta(1);

    result(2) = a(2) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(2) * delta(0) * (1.0f - delta(1)) +
                c(2) * (1.0f - delta(0)) * delta(1) + d(2) * delta(0) * delta(1);

    result(3) = a(3) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(3) * delta(0) * (1.0f - delta(1)) +
                c(3) * (1.0f - delta(0)) * delta(1) + d(3) * delta(0) * delta(1);
    return result;
}
}  // namespace ScanReconstruction

#endif  // PIXEL_UTILS_HPP
