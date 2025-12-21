#ifndef LOCAL_MAP_H_
#define LOCAL_MAP_H_

#include <scene.h>
#include <memory>

#include <renderState.h>
#include <settings.h>
#include <trackingState.h>

namespace surface_reconstruction {
struct LocalMap {
    LocalMap(std::shared_ptr<Settings> settins) {
        scene_ = std::make_shared<Scene>(settins);
        tracking_state_ = std::make_shared<TrackingState>(
            settins->depth_imageSize.height, settins->depth_imageSize.width, 0.2, 0.3, 0.5);
    }
    // 子地图所拥有的场景数据
    std::shared_ptr<Scene> scene_;

    // 子地图在全局坐标系中的位姿估计
    Eigen::Matrix4f estimatedGlobalPose_;

    // 追踪状态
    std::shared_ptr<TrackingState> tracking_state_;

    // 渲染信息
    std::shared_ptr<RenderState> render_state_;
};
}  // namespace surface_reconstruction

#endif  // LOCAL_MAP_H_
