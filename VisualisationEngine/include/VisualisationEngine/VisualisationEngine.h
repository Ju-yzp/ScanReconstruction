#ifndef VISUALISATION_ENGINE_H_
#define VISUALISATION_ENGINE_H_

#include <memory>
#include "../VisualisationEngine/RenderState.h"
#include "../VisualisationEngine/TrackingState.h"
#include "../VisualisationEngine/View.h"
#include "../VisualisationEngine/VoxelBlockHash.h"

class VisualisationEngine {
public:
    // 分配体素并融合
    void processFrame(std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view);

    // 准备下一次跟踪所需要的相机位姿、法向量图、点云图
    void prepare(
        std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view,
        std::shared_ptr<TrackingState> trackingState);

    // 光线投射
    static void raycast(
        std::shared_ptr<VoxelBlockHash> vbh, RenderState* renderState,
        RGBDCalibrationParams* cameraParams, const Eigen::Matrix4f inv_m);

private:
};

#endif
