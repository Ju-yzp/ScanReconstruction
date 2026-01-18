#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <Allocator.h>
#include <DepthTracker.h>
#include <GlobalSettings.h>
#include <Types.h>
#include <Viewer.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>

using namespace ScanReconstruction;

struct KeyframeInfo {
    std::vector<float> features;
};

class AsyncCosPlaceManager {
private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest curr_request;
    const size_t INPUT_SIZE = 224;
    float threshold = 0.80f;

public:
    std::vector<KeyframeInfo> database;

    AsyncCosPlaceManager(const std::string& model_path, float th) : threshold(th) {
        try {
            core.set_property(
                "CPU", {{"PERFORMANCE_HINT", "LATENCY"}, {"INFERENCE_NUM_THREADS", "4"}});
        } catch (const std::exception& e) {
            std::cout << "[OpenVINO] Warning: " << e.what() << std::endl;
        }

        auto model = core.read_model(model_path);
        ov::preprocess::PrePostProcessor ppp(model);

        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .mean({123.675f, 116.28f, 103.53f})
            .scale({58.395f, 57.12f, 57.375f});

        ppp.input().model().set_layout("NCHW");
        compiled_model = core.compile_model(ppp.build(), "CPU");
        curr_request = compiled_model.create_infer_request();
    }

    std::vector<float> extract_features(const cv::Mat& frame) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size((int)INPUT_SIZE, (int)INPUT_SIZE));
        ov::Tensor input_tensor(ov::element::u8, {1, INPUT_SIZE, INPUT_SIZE, 3}, resized.data);
        curr_request.set_input_tensor(input_tensor);
        curr_request.infer();

        auto output = curr_request.get_output_tensor();
        float* data = output.data<float>();
        std::vector<float> vec(data, data + 512);

        float norm = 1e-9f;
        for (float f : vec) norm += f * f;
        norm = std::sqrt(norm);
        for (float& f : vec) f /= norm;
        return vec;
    }

    bool isNewScene(const std::vector<float>& feat, float& out_max_sim) {
        out_max_sim = 0.0f;
        if (database.empty()) return true;

        for (const auto& kf : database) {
            float sim = 0;
            for (size_t j = 0; j < 512; ++j) sim += feat[j] * kf.features[j];
            if (sim > out_max_sim) out_max_sim = sim;
        }
        return (out_max_sim < threshold);
    }

    void addKeyframe(const std::vector<float>& feat) { database.push_back({feat}); }
};

cv::Mat get_colored_depth(const cv::Mat& depth_m, float max_range) {
    cv::Mat gray_8u, color_map;
    depth_m.convertTo(gray_8u, CV_8U, 255.0 / max_range);
    cv::applyColorMap(gray_8u, color_map, cv::COLORMAP_JET);
    color_map.setTo(cv::Scalar(0, 0, 0), gray_8u <= 0);
    return color_map;
}

class ImageConverterNode : public rclcpp::Node {
public:
    ImageConverterNode(const std::string& model_path) : Node("tsdf_cosplace_node") {
        this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        cosplace_manager_ = std::make_unique<AsyncCosPlaceManager>(model_path, 0.99f);

        global_settings_ = std::make_shared<GlobalSettings>();

        Eigen::Matrix3f k_mat = Eigen::Matrix3f::Identity();
        k_mat(0, 0) = 517.3f;
        k_mat(1, 1) = 516.5f;
        k_mat(0, 2) = 318.6f;
        k_mat(1, 2) = 255.3f;

        global_settings_->k = k_mat;
        global_settings_->height = 480;
        global_settings_->width = 640;
        global_settings_->set_max_num_threads(8);
        global_settings_->pyramid_levels = 3;
        global_settings_->space_threshold_min = 0.02f;
        global_settings_->space_threshold_max = 0.10f;
        global_settings_->lamdba_scale = 5.0f;
        global_settings_->initial_lamdba = 1.0f;
        global_settings_->min_num_iterations = 3;
        global_settings_->max_num_iterations = 6;
        global_settings_->viewFrustum_max = 4.0f;
        global_settings_->viewFrustum_min = 0.1f;
        global_settings_->voxel_size = 0.01f;
        global_settings_->mu = 0.05f;
        global_settings_->max_weight = 40.0f;

        viewer_ = std::make_shared<Viewer>(
            global_settings_->height, global_settings_->width, global_settings_->k, 5000.0f);
        depth_tracker_ = std::make_shared<DepthTracker>(global_settings_);
        rp_ = std::make_shared<ReconstructionPipeline>(global_settings_);

        auto static_tf_pub = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        geometry_msgs::msg::TransformStamped map_tf;
        map_tf.header.stamp = this->get_clock()->now();
        map_tf.header.frame_id = "world";
        map_tf.child_frame_id = "map";
        map_tf.transform.rotation.w = 1.0;
        static_tf_pub->sendTransform(map_tf);

        cv::namedWindow("TSDF System", cv::WINDOW_NORMAL);
    }

    void process_frame(const cv::Mat& rgb_image, cv::Mat& depth_image) {
        if (depth_image.empty() || rgb_image.empty()) return;

        if (viewer_->get_tracking_result() == TrackingResult::LOST) {
            RCLCPP_WARN(this->get_logger(), "Tracking lost!");
            return;
        }

        viewer_->set_current_points(depth_image);
        if (!is_initialized_) {
            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

            rp_->fusion(viewer_->get_current_points(), pose);
            rp_->raycast(viewer_->get_prev_points(), pose);
            viewer_->computePrevNormals();

            auto features = cosplace_manager_->extract_features(rgb_image);
            cosplace_manager_->addKeyframe(features);

            is_initialized_ = true;
            visualize(depth_image, pose, true, 0.0f);
            return;
        }

        Eigen::Matrix4f current_pose = depth_tracker_->get_initial_guess_pose();
        depth_tracker_->track(viewer_, current_pose);

        if (viewer_->get_tracking_result() != TrackingResult::GOOD &&
            viewer_->get_tracking_result() != TrackingResult::POOR) {
            return;
        }

        publish_tf_transform(current_pose, "map", "camera", this->get_clock()->now());

        auto features = cosplace_manager_->extract_features(rgb_image);
        float max_sim = 0.0f;
        bool is_new_scene = cosplace_manager_->isNewScene(features, max_sim);

        if (is_new_scene) {
            std::cout << "[FUSION] Sim: " << std::fixed << std::setprecision(4) << max_sim
                      << " (New Keyframe)" << std::endl;
            rp_->fusion(viewer_->get_current_points(), current_pose);
            cosplace_manager_->addKeyframe(features);
        } else {
            rp_->fusion(viewer_->get_current_points(), current_pose, false);
        }

        rp_->raycast(viewer_->get_prev_points(), current_pose);
        viewer_->computePrevNormals();

        visualize(depth_image, current_pose, is_new_scene, max_sim);
    }

    void exportSceneToSTL(std::string filename) { rp_->exportToSTL(filename); }

private:
    void visualize(
        const cv::Mat& raw_depth, const Eigen::Matrix4f& pose, bool is_keyframe, float sim) {
        cv::Mat raw_depth_m;
        raw_depth.convertTo(raw_depth_m, CV_32F, 1.0 / 5000.0);
        cv::Mat raw_color = get_colored_depth(raw_depth_m, 4.0f);

        cv::Mat ray_depth_m = cv::Mat::zeros(480, 640, CV_32FC1);
        Eigen::Matrix4f inv_pose = pose.inverse();
        Points& ray_points = viewer_->get_prev_points();

        for (int i = 0; i < (int)ray_points.size(); ++i) {
            if (std::isnan(ray_points[i](0))) continue;
            Eigen::Vector3f pt_cam =
                inv_pose.block<3, 3>(0, 0) * ray_points[i] + inv_pose.block<3, 1>(0, 3);
            if (pt_cam.z() > 0) ray_depth_m.at<float>(i / 640, i % 640) = pt_cam.z();
        }
        cv::Mat ray_color = get_colored_depth(ray_depth_m, 4.0f);

        cv::Mat combined;
        cv::hconcat(raw_color, ray_color, combined);

        std::string status = is_keyframe ? "FUSION" : "TRACKING";
        cv::Scalar color = is_keyframe ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(combined, "Live: " + status, cv::Point(20, 30), 1, 1.5, color, 2);
        cv::putText(
            combined, "Raycast (Model)", cv::Point(640 + 20, 30), 1, 1.5, cv::Scalar(255, 255, 0),
            2);
        cv::putText(
            combined, "Sim: " + std::to_string(sim).substr(0, 5), cv::Point(20, 60), 1, 1.2,
            cv::Scalar(255, 255, 255), 1);

        cv::imshow("TSDF System", combined);
        cv::waitKey(1);
    }

    void publish_tf_transform(
        const Eigen::Matrix4f& T, const std::string& parent, const std::string& child,
        rclcpp::Time stamp) {
        auto t_stamped = geometry_msgs::msg::TransformStamped();
        t_stamped.header.stamp = stamp;
        t_stamped.header.frame_id = parent;
        t_stamped.child_frame_id = child;
        t_stamped.transform.translation.x = T(0, 3);
        t_stamped.transform.translation.y = T(1, 3);
        t_stamped.transform.translation.z = T(2, 3);
        Eigen::Quaternionf q(T.block<3, 3>(0, 0));
        t_stamped.transform.rotation.x = q.x();
        t_stamped.transform.rotation.y = q.y();
        t_stamped.transform.rotation.z = q.z();
        t_stamped.transform.rotation.w = q.w();
        tf_broadcaster_->sendTransform(t_stamped);
    }

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<GlobalSettings> global_settings_;
    std::shared_ptr<DepthTracker> depth_tracker_;
    std::shared_ptr<ReconstructionPipeline> rp_;
    std::unique_ptr<AsyncCosPlaceManager> cosplace_manager_;
    bool is_initialized_{false};
};

struct AssociationEntry {
    double timestamp;
    std::string rgb_file;
    std::string depth_file;
};

std::vector<AssociationEntry> loadAssociations(
    const std::string& filename, const std::string& base_path) {
    std::vector<AssociationEntry> data;
    std::ifstream file(filename);
    if (!file.is_open()) return data;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string ts_rgb_str, rgb_rel, ts_depth_str, depth_rel;
        ss >> ts_rgb_str >> rgb_rel >> ts_depth_str >> depth_rel;

        AssociationEntry entry;
        try {
            entry.timestamp = std::stod(ts_rgb_str);
            entry.rgb_file = base_path + "/" + rgb_rel;
            entry.depth_file = base_path + "/" + depth_rel;
            data.push_back(entry);
        } catch (...) {
            continue;
        }
    }
    return data;
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    std::string model_xml = "/home/adrewn/Downloads/cosplace_openvino/model.xml";
    std::string dataset_base =
        "/media/adrewn/8f2e5859-196a-419d-b6f7-1e0c2c2578be/Downloads/rgbd_dataset_freiburg1_desk";
    std::string assoc_file = dataset_base + "/final_associations.txt";

    auto node = std::make_shared<ImageConverterNode>(model_xml);
    std::vector<AssociationEntry> entries = loadAssociations(assoc_file, dataset_base);
    std::cout << "Loaded " << entries.size() << " frames." << std::endl;

    for (const auto& entry : entries) {
        if (!rclcpp::ok()) break;
        cv::Mat rgb = cv::imread(entry.rgb_file);
        cv::Mat depth = cv::imread(entry.depth_file, cv::IMREAD_UNCHANGED);

        if (rgb.empty() || depth.empty()) continue;

        node->process_frame(rgb, depth);
        rclcpp::spin_some(node);
    }

    // node->exportSceneToSTL("desk_fusion_result.stl");
    rclcpp::shutdown();
    return 0;
}
