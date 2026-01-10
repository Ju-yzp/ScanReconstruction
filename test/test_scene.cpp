#include <GlobalSettings.h>
#include <Integrator.h>
#include <Raycaster.h>
#include <Scene.h>
#include <Viewer.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace ScanReconstruction;

int main() {
    std::string base_path = "/home/adrewn/Downloads/rgbd_dataset_freiburg1_desk/";
    std::string assoc_file = base_path + "final_associations.txt";  // 確保這是你對齊後的文件名

    std::shared_ptr<GlobalSettings> global_settings = std::make_shared<GlobalSettings>();
    Eigen::Matrix3f k = Eigen::Matrix3f::Identity();
    k(0, 0) = 517.3f;
    k(1, 1) = 516.5f;
    k(0, 2) = 318.6f;
    k(1, 2) = 255.3f;

    global_settings->k = k;
    global_settings->height = 480;
    global_settings->width = 640;
    global_settings->reverse_entries_num = 3000000;
    global_settings->viewFrustum_max = 4.0f;
    global_settings->viewFrustum_min = 0.1f;
    global_settings->voxel_size = 0.005f;
    global_settings->mu = 0.02f;
    global_settings->max_weight = 100.0f;
    global_settings->set_max_num_threads(8);

    std::shared_ptr<Viewer> viewer = std::make_shared<Viewer>(480, 640, k, 5000.0f);
    std::shared_ptr<Raycaster> raycaster = std::make_shared<Raycaster>(global_settings);
    std::shared_ptr<Scene> scene = std::make_shared<Scene>(global_settings);
    std::shared_ptr<Integrator> integrator = std::make_shared<Integrator>(global_settings);

    std::ifstream fin(assoc_file);
    if (!fin.is_open()) {
        std::cerr << "Cannot open association file: " << assoc_file << std::endl;
        return -1;
    }

    std::string line;
    int frame_count = 0;
    Points ray_points;
    ray_points.resize(global_settings->height * global_settings->width);
    while (std::getline(fin, line) && frame_count < 300) {
        std::stringstream ss(line);
        double t_rgb, t_depth, t_pose;
        std::string rgb_path, depth_path;
        float tx, ty, tz, qx, qy, qz, qw;

        // 解析格式: t_rgb rgb_path t_depth depth_path t_pose tx ty tz qx qy qz qw
        if (!(ss >> t_rgb >> rgb_path >> t_depth >> depth_path >> t_pose >> tx >> ty >> tz >> qx >>
              qy >> qz >> qw))
            continue;

        // 加載深度圖
        cv::Mat depth_img = cv::imread(base_path + depth_path, cv::IMREAD_UNCHANGED);
        if (depth_img.empty()) continue;

        // 構造當前位姿
        Eigen::Quaternionf q(qw, qx, qy, qz);
        q.normalize();
        Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
        camera_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        camera_pose.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);

        // 1. 更新點雲數據
        viewer->set_current_points(depth_img);

        // 2. 分配體素塊並融合
        raycaster->allocateVoxelblocks(viewer->get_current_points(), camera_pose, scene);
        integrator->integrateDepthIntoScene(
            viewer->get_current_points(), camera_pose, raycaster->get_updated_hashEntries(), scene);

        // 3. 執行渲染 (Raycast) 用於可視化

        raycaster->raycast(ray_points, camera_pose, scene);

        // 4. 生成可視化圖像
        // cv::Mat ray_image = cv::Mat(global_settings->height, global_settings->width, CV_32FC1);
        // for (int i = 0; i < (int)ray_points.size(); ++i) {
        //     int y = i / global_settings->width;
        //     int x = i % global_settings->width;
        //     ray_image.at<float>(y, x) = std::isnan(ray_points[i](0)) ? 0.0f : (ray_points[i](2)
        //     );
        // }

        // // 處理顯示用的原始深度圖 (歸一化)
        // cv::Mat depth_show;
        // depth_img.convertTo(depth_show, CV_32F, 1.0 / (5000.0 *
        // global_settings->viewFrustum_max));

        // // 拼接並顯示
        // cv::Mat combined;
        // cv::hconcat(depth_show, ray_image, combined);
        // cv::imshow("Incremental Reconstruction (Left: Depth, Right: Raycast)", combined);

        // // 控制循環速度，按 ESC 退出
        // if (cv::waitKey(1) == 27) break;

        // std::cout << "Processed frame: " << ++frame_count << " / 200" << std::endl;
    }

    std::cout << "Test Finished. Total frames: " << frame_count << std::endl;
    cv::waitKey(0);
    return 0;
}
