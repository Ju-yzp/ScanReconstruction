#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <DepthTracker.h>
#include <GlobalSettings.h>
#include <Viewer.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include "Types.h"

using namespace ScanReconstruction;

class ImageConverterNode : public rclcpp::Node {
public:
    ImageConverterNode() : Node("vo_node") {
        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", qos_profile,
            std::bind(&ImageConverterNode::depth_callback, this, std::placeholders::_1));

        this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        cv::Mat K =
            (cv::Mat_<double>(3, 3) << -751.5050659179688, 0.0, 635.4237670898438, 0.0,
             750.8400268554688, 363.7984313964844, 0.0, 0.0, 1.0);
        cv::Mat D = (cv::Mat_<double>(8, 1) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        global_settings_ = std::make_shared<GlobalSettings>();
        global_settings_->set_max_num_threads(8);
        global_settings_->width = 1280;
        global_settings_->height = 720;
        global_settings_->k << -751.5050659179688, 0.0, 635.4237670898438, 0.0, 750.8400268554688,
            363.7984313964844, 0.0, 0.0, 1.0;
        global_settings_->space_threshold_min = 0.03f;
        global_settings_->space_threshold_max = 0.1f;
        global_settings_->min_num_iterations = 3;
        global_settings_->max_num_iterations = 6;
        global_settings_->pyramid_levels = 3;
        global_settings_->initial_lamdba = 1.0f;
        global_settings_->lamdba_scale = 5.0f;
        global_settings_->viewFrustum_max = 4.0f;
        global_settings_->viewFrustum_min = 0.15f;
        global_settings_->voxel_size = 0.01f;
        global_settings_->voxel_block_size = 8;
        global_settings_->reverse_num = 100000;
        global_settings_->bin_num = 13;
        global_settings_->similarity_threshold = 0.5f;

        viewer_ = std::make_shared<Viewer>(
            global_settings_->height, global_settings_->width, global_settings_->k, 1000.0f);

        depth_tracker_ = std::make_shared<DepthTracker>(global_settings_);

        static auto static_tf_pub = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        geometry_msgs::msg::TransformStamped map_tf;
        map_tf.header.stamp = this->get_clock()->now();
        map_tf.header.frame_id = "world";
        map_tf.child_frame_id = "map";

        map_tf.transform.translation.x = 0.0;
        map_tf.transform.translation.y = 0.0;
        map_tf.transform.translation.z = 0.0;
        map_tf.transform.rotation.w = 1.0;

        static_tf_pub->sendTransform(map_tf);
    }

private:
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        auto start_total = this->get_clock()->now();
        cv_bridge::CvImagePtr cv_depth_ptr;
        try {
            cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (Depth): %s", e.what());
            return;
        }

        cv::Mat depth_image = cv_depth_ptr->image;

        if (!depth_image.empty()) {
            if (viewer_->get_tracking_result() == TrackingResult::LOST) {
                RCLCPP_WARN(this->get_logger(), "Tracking lost!");
                return;
            }

            viewer_->set_current_points(depth_image);

            if (is_initialized_ == false) {
                viewer_->swap_current_and_previous_points();
                is_initialized_ = true;
                return;
            }
            Eigen::Matrix4f initial_pose = depth_tracker_->get_initial_guess_pose();
            depth_tracker_->track(viewer_, initial_pose);
            if (viewer_->get_tracking_result() == TrackingResult::GOOD) {
                std::cout << "current pose :\n" << initial_pose << std::endl;
                publish_tf_transform(initial_pose, "map", "camera", this->get_clock()->now());
                viewer_->transform(initial_pose);
                viewer_->swap_current_and_previous_points();
            }
        }
    }

    void publish_tf_transform(
        const Eigen::Matrix4f& T_world_optical, const std::string& parent_frame_id,
        const std::string child_frame_id, const rclcpp::Time stamp) {
        Eigen::Matrix4f T_map_to_base = T_world_optical;
        auto t_stamped = geometry_msgs::msg::TransformStamped();
        t_stamped.header.stamp = stamp;
        t_stamped.header.frame_id = parent_frame_id;
        t_stamped.child_frame_id = child_frame_id;

        t_stamped.transform.translation.x = T_map_to_base(0, 3);
        t_stamped.transform.translation.y = T_map_to_base(1, 3);
        t_stamped.transform.translation.z = T_map_to_base(2, 3);

        Eigen::Quaternionf q(T_map_to_base.block<3, 3>(0, 0));
        t_stamped.transform.rotation.x = q.x();
        t_stamped.transform.rotation.y = q.y();
        t_stamped.transform.rotation.z = q.z();
        t_stamped.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(t_stamped);
    }
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::shared_ptr<Viewer> viewer_;
    std::shared_ptr<GlobalSettings> global_settings_;

    std::shared_ptr<DepthTracker> depth_tracker_;

    bool is_initialized_{false};
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverterNode>());
    while (rclcpp::ok()) {
    }
    rclcpp::shutdown();
    return 0;
}
