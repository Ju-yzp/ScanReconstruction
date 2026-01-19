#ifndef SCAN_RECONSTRUCTION_PIXEL_UTILS_H_
#define SCAN_RECONSTRUCTION_PIXEL_UTILS_H_

#include <Types.h>

// tbb
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>

// cpp
#include <limits>

namespace ScanReconstruction {
inline void filterSubsampleWithHoles(
    const Image& input, Image& output, int height, int width, bool isNormal = false) {
    int output_rows = height / 2;
    int output_cols = width / 2;

    if (output.size() != size_t(output_cols * output_rows))
        throw std::runtime_error("Output size is not correct");

    const Eigen::Vector3f* input_ptr = input.data();

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range2d<int>(0, output_rows, 0, output_cols),
        [&](const oneapi::tbb::blocked_range2d<int>& r) {
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
    const Image& input, Eigen::Vector2f point, const int cols) {
    Eigen::Vector2i img_coord = point.array().floor().cast<int>();
    const Eigen::Vector3f* input_ptr = input.data() + img_coord(1) * cols + img_coord(0);
    Eigen::Vector2f delta = img_coord.cast<float>() - point;

    Eigen::Vector3f points[4], result;
    points[0] = *input_ptr;
    points[1] = *(input_ptr + 1);
    points[2] = *(input_ptr + cols);
    points[3] = *(input_ptr + 1 + cols);

    if (std::isnan(points[0](0)) || std::isnan(points[1](0)) || std::isnan(points[2](0)) ||
        std::isnan(points[3](0))) {
        result(0) = std::numeric_limits<float>::quiet_NaN();
        return result;
    }
    float d0 = delta(0);
    float d1 = delta(1);

    result = points[0] * (1.0f - d0) * (1.0f - d1) + points[1] * d0 * (1.0f - d1) +
             points[2] * (1.0f - d0) * d1 + points[3] * d0 * d1;
    return result;
}
}  // namespace ScanReconstruction

#endif
