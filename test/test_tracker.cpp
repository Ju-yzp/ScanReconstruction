#include <Tracker/cameraParams.h>
#include <Tracker/pixelUtils.h>
#include <Tracker/tracker.h>
#include <opencv2/core/persistence.hpp>
#include <sophus/se3.hpp>

cv::Mat getDepth(std::string file_path) {
    cv::Mat origin = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    cv::Mat convert;
    convertShortToFloat(&origin, &convert, 5000.0f);
    return convert;
}

cv::Mat computeNormalMapFromGradient_CV4(const cv::Mat& depthMap) {
    if (depthMap.empty() || depthMap.type() != CV_32F) {
        std::cerr << "Error: Input depth map must be CV_32F type." << std::endl;
        return cv::Mat();
    }

    int rows = depthMap.rows;
    int cols = depthMap.cols;

    cv::Mat normalMap(rows, cols, CV_32FC4, cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));

    for (int y = 1; y < rows - 1; ++y) {
        const float* depth_prev = depthMap.ptr<float>(y - 1);
        const float* depth_curr = depthMap.ptr<float>(y);
        const float* depth_next = depthMap.ptr<float>(y + 1);

        cv::Vec4f* normal_ptr = normalMap.ptr<cv::Vec4f>(y);

        for (int x = 1; x < cols - 1; ++x) {
            float Z_x_minus_1 = depth_curr[x - 1];
            float Z_x_plus_1 = depth_curr[x + 1];
            float Z_y_minus_1 = depth_prev[x];
            float Z_y_plus_1 = depth_next[x];

            if (Z_x_minus_1 <= 1e-6 || Z_x_plus_1 <= 1e-6 || Z_y_minus_1 <= 1e-6 ||
                Z_y_plus_1 <= 1e-6) {
                continue;
            }

            float dzdx = (Z_x_plus_1 - Z_x_minus_1) / 2.0f;
            float dzdy = (Z_y_plus_1 - Z_y_minus_1) / 2.0f;

            cv::Vec4f normal_vec(-dzdx, -dzdy, 1.0f, 0.0f);

            float norm_sq = normal_vec[0] * normal_vec[0] + normal_vec[1] * normal_vec[1] +
                            normal_vec[2] * normal_vec[2];
            float norm = std::sqrt(norm_sq);

            if (norm > 1e-6) {
                normal_vec[0] /= norm;
                normal_vec[1] /= norm;
                normal_vec[2] /= norm;
            } else {
                continue;
            }

            normal_ptr[x] = normal_vec;
        }
    }

    return normalMap;
}

void print_se3f_components_via_so3(const Sophus::SE3f& se3_pose) {
    // 1. 先获取 Sophus::SO3f 对象
    Sophus::SO3f rotation_so3 = se3_pose.so3();

    // 2. 将 SO3f 转换为 Eigen::Quaternionf
    Eigen::Quaternionf rotation_q = rotation_so3.unit_quaternion();

    // 3. 获取平移分量
    Eigen::Vector3f translation_t = se3_pose.translation();

    std::cout << "--- Sophus::SE3f 位姿分解 (通过 SO3) ---" << std::endl;

    std::cout << "Quaternion (w, x, y, z): " << rotation_q.w() << ", " << rotation_q.x() << ", "
              << rotation_q.y() << ", " << rotation_q.z() << std::endl;

    std::cout << "Quaternion (x, y, z, w): " << rotation_q.x() << ", " << rotation_q.y() << ", "
              << rotation_q.z() << ", " << rotation_q.w() << std::endl;

    std::cout << "Translation (x, y, z): " << translation_t.x() << ", " << translation_t.y() << ", "
              << translation_t.z() << std::endl;

    std::cout << "----------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    View view;

    std::string file1_path = "/home/adrewn/surface_restruction/data/1305031102.160407.png";
    std::string file2_path = "/home/adrewn/surface_restruction/data/1305031102.194330.png";

    CalibrationParams cp;
    cp.fx = 525.0;
    cp.fy = 525.0;
    cp.cx = 319.5;
    cp.cy = 239.5;

    cp.viewFrustum_min = 0.1;
    cp.viewFrustum_max = 3.0f;

    view.depth = getDepth(file1_path);
    cv::Mat convert = getDepth(file2_path);
    cv::Mat points = cv::Mat(convert.rows, convert.cols, CV_32FC4);
    for (int y{0}; y < points.rows; ++y)
        for (int x{0}; x < points.cols; ++x) {
            cv::Vec4f p = cp.computePointCloud(convert.at<float>(y, x), cv::Vec2i(x, y));
            points.at<cv::Vec4f>(y, x) = p;
        }

    view.points = points;
    Eigen::Vector3f translation(1.3352, 0.6261, 1.6519);
    Eigen::Quaternionf rotation_q(-0.3231, 0.6564, 0.6139, -0.2963);
    view.scenePose = Sophus::SE3f(rotation_q, translation);
    view.pose_d = view.scenePose;
    view.normal = computeNormalMapFromGradient_CV4(view.depth);

    Tracker tracker(4, 12, 0.01f, 0.002f, cp);
    tracker.track(&view);
    print_se3f_components_via_so3(view.pose_d);
}
