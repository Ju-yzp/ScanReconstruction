#ifndef LOCAL_MAP_MANAGER_H_
#define LOCAL_MAP_MANAGER_H_

#include <GlobalSettings.h>
#include <LocalMap.h>
#include <Types.h>
#include <memory>

namespace ScanReconstruction {

struct ActiveDataDescriptor {
    MapType type;
    int localMapId = -1;
    int trackingAttempts = 0;
};

class LocalMapManager {
public:
    LocalMapManager(std::shared_ptr<GlobalSettings> global_settings);

    std::vector<std::shared_ptr<LocalMap>> get_inactive_maps();

    bool shouldCreateNewLocalMap();

    void createNewLocalMap(Eigen::Matrix4f pose, bool isPrimary = false);

    int get_primary_id();

    std::shared_ptr<LocalMap> get_primary_map();

private:
    std::shared_ptr<GlobalSettings> global_settings_;
    std::vector<std::shared_ptr<LocalMap>> maps_;
    std::vector<ActiveDataDescriptor> activeMaps;
    std::vector<std::shared_ptr<LocalMap>> inactive_maps_;
    std::mutex map_mutex_;
    float visiable_threshold_;
};
}  // namespace ScanReconstruction

#endif
