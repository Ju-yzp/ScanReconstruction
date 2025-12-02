#ifndef VISUALISATION_ENGINE_H_
#define VISUALISATION_ENGINE_H_

#include <renderState.h>
#include <scene.h>
#include <trackingState.h>
#include <view.h>
#include <memory>

namespace surface_restruction {
void allocateVoxelFormDepth(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, bool updateVisibleList = true);

void intergrateIntoScene(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState);

void raycast(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, std::shared_ptr<RenderState> renderState);

void generatePointCloudsAndNormals(
    Eigen::Vector3f lightSource, float voxelSize, std::shared_ptr<RenderState> renderState,
    std::shared_ptr<TrackingState> trackingState);
}  // namespace surface_restruction

#endif  // VISUALISATION_ENGINE_H_+
