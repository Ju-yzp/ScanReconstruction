#include <GlobalSettings.h>
#include <Integrator.h>
#include <Raycaster.h>
#include <Scene.h>
#include <Viewer.h>

#include <cmath>
#include <memory>

#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace ScanReconstruction;
int main() {
    // for (int i = 0; i < 20; ++i) {
    std::shared_ptr<GlobalSettings> global_settings = std::make_shared<GlobalSettings>();
    Eigen::Matrix3f k = Eigen::Matrix3f::Identity();
    k(0, 0) = 517.3f;
    k(1, 1) = 516.5f;
    k(0, 2) = 318.6f;
    k(1, 2) = 255.3f;

    float tx = 1.1067f;
    float ty = -0.0225f;
    float tz = 1.5976f;
    float qx = 0.8286f;
    float qy = 0.4013f;
    float qz = -0.1914f;
    float qw = -0.3401f;

    Eigen::Quaternionf q(qw, qx, qy, qz);
    q.normalize();

    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    camera_pose.block<3, 3>(0, 0) = q.toRotationMatrix();
    camera_pose.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);

    global_settings->k = k;
    global_settings->height = 480;
    global_settings->width = 640;
    global_settings->reverse_entries_num = 1500000;
    global_settings->viewFrustum_max = 4.0f;
    global_settings->viewFrustum_min = 0.1f;
    global_settings->voxel_size = 0.005f;
    global_settings->mu = 0.02f;
    global_settings->max_weight = 100.0f;
    global_settings->set_max_num_threads(8);

    std::shared_ptr<Viewer> viewer = std::make_shared<Viewer>(480, 640, k, 5000.0f);
    cv::Mat depth_img = cv::imread(
        "/home/adrewn/ScanReconstruction/dataset/1305031455.939606.png", cv::IMREAD_UNCHANGED);

    viewer->set_current_points(depth_img);
    std::shared_ptr<Raycaster> raycaster = std::make_shared<Raycaster>(global_settings);
    std::shared_ptr<Scene> scene = std::make_shared<Scene>(global_settings);
    std::shared_ptr<Integrator> integrator = std::make_shared<Integrator>(global_settings);
    raycaster->allocateVoxelblocks(viewer->get_current_points(), camera_pose, scene);
    integrator->integrateDepthIntoScene(
        viewer->get_current_points(), camera_pose, raycaster->get_updated_hashEntries(), scene);
    Points ray_points;
    ray_points.resize(global_settings->height * global_settings->width);
    raycaster->raycast(ray_points, camera_pose, scene);
    cv::Mat ray_image = cv::Mat(global_settings->height, global_settings->width, CV_32FC1);
    for (int y = 0; y < global_settings->height; ++y) {
        for (int x = 0; x < global_settings->width; ++x) {
            Eigen::Vector3f& point = ray_points[y * global_settings->width + x];
            if (std::isnan(point(0)))
                ray_image.at<float>(y, x) = 0.0f;
            else
                ray_image.at<float>(y, x) = point(2);
        }
    }
    //}
    // cv::namedWindow("ray result", cv::WINDOW_AUTOSIZE);
    // cv::Mat depth_32f;
    // depth_img.convertTo(depth_32f, CV_32F, 1.0 / 5000.0);

    // cv::imshow("Depth CV_32F", depth_32f);
    // cv::imshow("ray result", ray_image);
    // cv::waitKey(0);
    // return 0;
}
