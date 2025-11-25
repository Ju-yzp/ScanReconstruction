#ifndef POSE_DATABASE_HPP_
#define POSE_DATABASE_HPP_

#include <sophus/se3.hpp>

#include <vector>

namespace FernRelocLib {
class PoseDatabase {
public:
    struct PoseInScene {
        PoseInScene(void) {}
        PoseInScene(const Sophus::SE3f& _pose, int _sceneIdx) : pose(_pose), sceneIdx(_sceneIdx) {}
        Sophus::SE3f pose;
        int sceneIdx;
    };

    PoseDatabase(void);

    ~PoseDatabase(void);

    void storePose(int id, const Sophus::SE3f& pose, int sceneId);

    int numPoses(void) const { return (int)mPoses.size(); }

    const PoseInScene& retrievePose(int id) const { return mPoses[id]; }

    PoseInScene retrieveWAPose(int k, int ids[], float weights[]) const;

    void saveToFile(const std::string& fileName);

    void loadFromFile(const std::string& fileName);

private:
    std::vector<PoseInScene> mPoses;
};
}  // namespace FernRelocLib

#endif
