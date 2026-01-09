#include <LocalMapManager.h>
#include <cstddef>
#include <memory>
#include <mutex>

namespace ScanReconstruction {
LocalMapManager::LocalMapManager(std::shared_ptr<GlobalSettings> global_settings)
    : global_settings_(global_settings) {
    visiable_threshold_ = global_settings_->visiable_threshold;
}

std::vector<std::shared_ptr<LocalMap>> LocalMapManager::get_inactive_maps() {
    std::unique_lock<std::mutex> map_lock(map_mutex_);
    return inactive_maps_;
}

bool LocalMapManager::shouldCreateNewLocalMap() {
    int primaryId = -1;
    for (size_t i = 0; i < activeMaps.size(); ++i)
        if (activeMaps[i].type == MapType::PRIMARY_LOCAL_MAP) primaryId = (int)i;

    if (primaryId == -1) return false;
    std::shared_ptr<LocalMap> localMap = maps_[(size_t)primaryId];

    return localMap->get_visible_ratio() > visiable_threshold_;
}

void LocalMapManager::createNewLocalMap(Eigen::Matrix4f pose, bool isPrimary) {
    if (isPrimary) {
        for (auto iter = activeMaps.begin(); iter != activeMaps.end(); ++iter)
            if ((*iter).type == MapType::PRIMARY_LOCAL_MAP) {
                activeMaps.erase(iter);
                std::unique_lock<std::mutex> map_lock(map_mutex_);
                inactive_maps_.emplace_back(maps_[(size_t)(iter)->localMapId]);
                break;
            }

        std::shared_ptr<LocalMap> newLocalMap = std::make_shared<LocalMap>(global_settings_);
        newLocalMap->set_global_pose(pose);
        ActiveDataDescriptor activeDescriptor;
        activeDescriptor.localMapId = (int)maps_.size();
        activeDescriptor.type = MapType::PRIMARY_LOCAL_MAP;
        maps_.emplace_back(newLocalMap);
        activeMaps.emplace_back(activeDescriptor);
    }
}

int LocalMapManager::get_primary_id() {
    for (auto activeDescriptor : activeMaps) {
        if (activeDescriptor.type == MapType::PRIMARY_LOCAL_MAP) return activeDescriptor.localMapId;
    }

    return -1;
}

std::shared_ptr<LocalMap> LocalMapManager::get_primary_map() {
    int primaryId = get_primary_id();
    if (primaryId == -1)
        return nullptr;
    else
        return maps_[(size_t)primaryId];
}
}  // namespace ScanReconstruction
