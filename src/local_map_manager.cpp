#include <localMapManager.h>
#include <cassert>

namespace surface_restruction {
ActiveMapDescriptor LocalMapManager::createNewLocalMap(bool isPrimary) {
    ActiveMapDescriptor descriptor;
    descriptor.id = static_cast<int>(activeLocalMaps_.size());
    descriptor.state = isPrimary ? LocalMapState::PRIMARY_LOCAL_MAP : LocalMapState::NEW_LOCAL_MAP;
    activeLocalMaps_.emplace_back(descriptor);
    return descriptor;
}

std::optional<int> LocalMapManager::getPrimaryLocalMapIndex() const {
    for (const auto& map : activeLocalMaps_)
        if (map.state == LocalMapState::PRIMARY_LOCAL_MAP) return map.id;
    return std::nullopt;
}

void LocalMapManager::recordTrackingResult(
    int localMapId, std::shared_ptr<TrackingState> tracking_state) {
    assert(localMapId <= static_cast<int>(activeLocalMaps_.size()));

    ActiveMapDescriptor& currentLocalMap = activeLocalMaps_[localMapId];

    std::optional<int> primaryLocalMapId = getPrimaryLocalMapIndex();

    if (tracking_state->get_tracking_result() == TrackingState::TrackingResult::TRACKING_GOOD) {
        if (currentLocalMap.state == LocalMapState::RELOCALISATION)
            ;
        else if (
            (currentLocalMap.state == LocalMapState::NEW_LOCAL_MAP ||
             currentLocalMap.state == LocalMapState::LOOPCOUSED) &&
            primaryLocalMapId.has_value()) {
        }
    } else if (
        tracking_state->get_tracking_result() == TrackingState::TrackingResult::TRACKING_FAILED) {
        if (currentLocalMap.state == LocalMapState::PRIMARY_LOCAL_MAP)
            for (auto& localMap : activeLocalMaps_) {
                if (localMap.state == LocalMapState::NEW_LOCAL_MAP)
                    localMap.state = LocalMapState::NEW_LOST;
                else
                    localMap.state = LocalMapState::LOST;
            }
    }
}
}  // namespace surface_restruction
