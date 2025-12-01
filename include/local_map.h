#ifndef LOCAL_MAP_H_
#define LOCAL_MAP_H_

#include <scene.h>
#include <memory>

namespace SufaceRestruction {
class LocalMap {
public:
    LocalMap(Eigen::Matrix4f estimatedGlobalPose);

private:
    // 子地图所拥有的场景数据
    std::shared_ptr<Scene> scene_;

    // 子地图在全局坐标系中的位姿估计
    Eigen::Matrix4f estimatedGlobalPose_;
};
}  // namespace SufaceRestruction

#endif  // LOCAL_MAP_H_
