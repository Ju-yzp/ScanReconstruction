#include <KeyframeSystem.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tracker.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstddef>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

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

        tracker_ = std::make_unique<Tracker>(
            K, D, cv::Size2i(1280, 720), 8, 10, 50, 1.0f, 5.0f, 1000.0f, 0.03f);

        view_ = std::make_unique<View>(720, 1280);

        keyframe_system_ = std::make_unique<KeyframeSystem>(1280, 720, 13, 0.7f, 2.0f, 3.0f, 0.0f);

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

        cv::namedWindow("Keyframes", cv::WINDOW_AUTOSIZE);
    }

private:
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(depth_msg, "16UC1");
        } catch (cv_bridge::Exception& e) {
            return;
        }

        cv::Mat depth = cv_ptr->image;
        if (depth.empty()) return;

        tracker_->undistortion(depth, *view_);
        view_->swapDepth();
        tracker_->computeNormalMap(*view_);

        Keyframe curr(keyframe_system_->get_bin_num());
        keyframe_system_->computeKeyFrame(view_->prev_normal, view_->prev_depth, curr);

        std::vector<size_t> ids;
        keyframe_system_->searchKNearsetNeighbor(ids, 1, curr);

        cv::Mat d32f;
        depth.convertTo(d32f, CV_32F, 0.001);

        const int target_h = 400;
        float scale = static_cast<float>(target_h) / d32f.rows;
        cv::Mat curr_resized;
        cv::resize(d32f / 5.0f, curr_resized, cv::Size(), scale, scale);

        cv::Mat left_display;

        if (ids.empty()) {
            if (keyframe_system_->insert(curr)) {
                depth_keyframes_.push_back(curr_resized.clone());
            }
            left_display = cv::Mat::zeros(curr_resized.size(), curr_resized.type());
        } else {
            if (ids[0] < depth_keyframes_.size()) {
                left_display = depth_keyframes_[ids[0]];
            } else {
                left_display = cv::Mat::zeros(curr_resized.size(), curr_resized.type());
            }
        }

        cv::Mat res;
        cv::hconcat(left_display, curr_resized, res);

        std::cout << "Total Keyframes: " << keyframe_system_->size() << std::endl;

        cv::imshow("Keyframes", res);
        cv::waitKey(3);
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

    std::unique_ptr<Tracker> tracker_;

    std::unique_ptr<View> view_;

    std::unique_ptr<KeyframeSystem> keyframe_system_;

    std::vector<cv::Mat> depth_keyframes_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverterNode>());
    while (rclcpp::ok()) {
    }
    rclcpp::shutdown();
    return 0;
}
