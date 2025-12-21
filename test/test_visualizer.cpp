#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tracker.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

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

        tracker_ = std::make_unique<Tracker>(
            K, D, cv::Size2i(1280, 720), 8, 10, 1.0f, 5.0f, 1000.0f, 0.03f);

        view_ = std::make_unique<View>(720, 1280);
        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos_profile);
        RCLCPP_INFO(
            this->get_logger(),
            "Image converter node initialized, subscribing to depth images only");

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
        cv_bridge::CvImagePtr cv_depth_ptr;
        try {
            cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (Depth): %s", e.what());
            return;
        }

        cv::Mat depth_image = cv_depth_ptr->image;

        if (!depth_image.empty()) {
            tracker_->undistortion(depth_image, *view_);
            if (!isInitialied) {
                view_->swapDepth();
                Tracker::transform(*view_, tracker_->get_pose());
                Tracker::computeNormalMap(*view_);
                isInitialied = true;
            } else {
                Eigen::Matrix4f pose = tracker_->track(*view_);
                view_->swapDepth();
                Tracker::transform(*view_, pose);
                Tracker::computeNormalMap(*view_);
                tracker_->set_global_pose(pose);
                std::cout << "Final Pose" << std::endl;
                std::cout << pose << std::endl;
                publish_tf_transform(pose, "map", "camera", this->get_clock()->now());
            }
        }
    }

    void publish_tf_transform(
        const Eigen::Matrix4f& T_world_optical, const std::string& parent_frame_id,
        const std::string child_frame_id, const rclcpp::Time stamp) {
        static Eigen::Matrix4f camera_to_base;
        camera_to_base << 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f;

        // Eigen::Matrix4f T_map_to_base = camera_to_base * T_world_optical *
        // camera_to_base.inverse();

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

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;

    std::unique_ptr<Tracker> tracker_;

    std::unique_ptr<View> view_;

    bool isInitialied{false};
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverterNode>());
    while (rclcpp::ok()) {
    }
    rclcpp::shutdown();
    return 0;
}
