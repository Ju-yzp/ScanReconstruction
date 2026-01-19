#ifndef SCAN_RECONSTRUCTION_RENDERED_VIEW_H_
#define SCAN_RECONSTRUCTION_RENDERED_VIEW_H_

#include <Types.h>
#include <cassert>
#include <cstddef>

// tbb
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

namespace ScanReconstruction {
struct RenderedView {
    RenderedView(int height_, int width_) : height(height_), width(width_) {
        assert(height <= 0 || height <= 6 || width <= 0 || width <= 6);
        size_t img_size = size_t(height * width);
        points.resize(img_size);
        normals.resize(img_size);
        rendered_pose = Eigen::Matrix4f::Identity();
    }

    void precomputeNormals() {
        Eigen::Vector3f* points_ptr = points.data();
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<int>(3, height - 3, 3, width - 3),
            [&](const oneapi::tbb::blocked_range2d<int>& r) {
                for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
                    Eigen::Vector3f* normals_ptr = normals.data() + (y * width);
                    for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
                        Eigen::Vector3f points[4];
                        Eigen::Vector3f& normal = normals_ptr[x];
                        Eigen::Vector3f diff_x, diff_y;

                        points[0] = points_ptr[x + 2 + y * width];
                        points[1] = points_ptr[x + (y + 2) * width];
                        points[2] = points_ptr[x - 2 + y * width];
                        points[3] = points_ptr[x + (y - 2) * width];

                        bool doPlus{false};

                        if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                            std::isnan(points[2](0)) || std::isnan(points[3](0)))
                            doPlus = true;

                        if (doPlus) {
                            points[0] = points_ptr[x + 1 + y * width];
                            points[1] = points_ptr[x + (y + 1) * width];
                            points[2] = points_ptr[x - 1 + y * width];
                            points[3] = points_ptr[x + (y - 1) * width];

                            if (std::isnan(points[0](0)) || std::isnan(points[1](0)) ||
                                std::isnan(points[2](0)) || std::isnan(points[3](0))) {
                                normal(0) = std::numeric_limits<float>::quiet_NaN();
                                continue;
                            }
                        }
                        diff_x = points[0] - points[2];
                        diff_y = points[1] - points[3];

                        normal = diff_y.cross(diff_x);

                        float norm = normal.norm();

                        if (norm < 1e-5) {
                            normal(0) = std::numeric_limits<float>::quiet_NaN();
                            continue;
                        }
                        normal /= norm;
                    }
                }
            });
    }

    Image points;
    Image normals;
    int height, width;
    Eigen::Matrix4f rendered_pose;
};
}  // namespace ScanReconstruction

#endif
