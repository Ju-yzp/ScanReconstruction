#include <Allocator.h>
#include <GlobalSettings.h>
#include <Viewer.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/version.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

using namespace ScanReconstruction;

cv::Mat get_colored_depth(const cv::Mat& depth_m, float max_range) {
    cv::Mat gray_8u, color_map;
    depth_m.convertTo(gray_8u, CV_8U, 255.0 / max_range);
    cv::applyColorMap(gray_8u, color_map, cv::COLORMAP_JET);
    color_map.setTo(cv::Scalar(0, 0, 0), gray_8u <= 0);
    return color_map;
}

int main() {
    std::string base_path = "/home/adrewn/Downloads/rgbd_dataset_freiburg1_desk/";
    std::string assoc_file = base_path + "final_associations.txt";

    std::shared_ptr<GlobalSettings> global_settings = std::make_shared<GlobalSettings>();
    Eigen::Matrix3f k = Eigen::Matrix3f::Identity();
    k(0, 0) = 517.3f;
    k(1, 1) = 516.5f;
    k(0, 2) = 318.6f;
    k(1, 2) = 255.3f;

    global_settings->k = k;
    global_settings->height = 480;
    global_settings->width = 640;
    global_settings->viewFrustum_max = 4.0f;
    global_settings->viewFrustum_min = 0.1f;
    global_settings->voxel_size = 0.01f;
    global_settings->mu = 0.03f;
    global_settings->max_weight = 100.0f;
    global_settings->set_max_num_threads(8);

    std::shared_ptr<Viewer> viewer = std::make_shared<Viewer>(480, 640, k, 5000.0f);
    std::shared_ptr<ReconstructionPipeline> rp =
        std::make_shared<ReconstructionPipeline>(global_settings, 8);

    std::cout << "--- TBB Version Info ---" << std::endl;
    std::cout << "Major Version: " << TBB_VERSION_MAJOR << std::endl;
    std::cout << "Minor Version: " << TBB_VERSION_MINOR << std::endl;
    std::cout << "Interface Version: " << TBB_INTERFACE_VERSION << std::endl;

    std::ifstream fin(assoc_file);
    if (!fin.is_open()) return -1;

    std::string line;
    int frame_count = 0;
    int width = global_settings->width;
    int height = global_settings->height;
    float max_d = global_settings->viewFrustum_max;

    Points ray_points;
    ray_points.resize(height * width);

    Timer total("total");

    while (std::getline(fin, line) && frame_count < 570) {
        std::stringstream ss(line);
        double t_rgb, t_depth, t_pose;
        std::string rgb_path, depth_path;
        float tx, ty, tz, qx, qy, qz, qw;

        if (!(ss >> t_rgb >> rgb_path >> t_depth >> depth_path >> t_pose >> tx >> ty >> tz >> qx >>
              qy >> qz >> qw))
            continue;

        cv::Mat depth_img = cv::imread(base_path + depth_path, cv::IMREAD_UNCHANGED);
        if (depth_img.empty()) continue;

        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
        camera_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        camera_pose.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);

        viewer->set_current_points(depth_img);
        rp->fusion(viewer->get_current_points(), camera_pose);
        rp->raycast(ray_points, camera_pose);

        cv::Mat raw_depth_m;
        depth_img.convertTo(raw_depth_m, CV_32F, 1.0 / 5000.0);

        cv::Mat ray_depth_m = cv::Mat::zeros(height, width, CV_32FC1);
        Eigen::Matrix4f inv_pose = camera_pose.inverse();

        for (int i = 0; i < (int)ray_points.size(); ++i) {
            if (std::isnan(ray_points[i](0))) continue;
            Eigen::Vector3f pt_cam =
                inv_pose.block<3, 3>(0, 0) * ray_points[i] + inv_pose.block<3, 1>(0, 3);
            ray_depth_m.at<float>(i / width, i % width) = pt_cam.z();
        }

        cv::Mat raw_color = get_colored_depth(raw_depth_m, max_d);
        cv::Mat ray_color = get_colored_depth(ray_depth_m, max_d);

        cv::Mat combined;
        cv::hconcat(raw_color, ray_color, combined);

        cv::putText(
            combined, "RAW", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(255, 255, 255), 2);
        cv::putText(
            combined, "RAYCAST", cv::Point(width + 20, 40), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(255, 255, 255), 2);

        cv::imshow("TSDF Reconstruction", combined);
        if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
