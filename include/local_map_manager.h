#ifndef LOCAL_MAP_MANAGER_H_
#define LOCAL_MAP_MANAGER_H_

// cpp
#include <cstdint>
#include <vector>

namespace SufaceRestruction {

enum class LocalMapState : uint8_t {
    PRIMARY_LOCAL_MAP = 0,
    NEW_LOCAL_MAP = 1,
    NEW_LOST = 2,
    LOST = 3,
    LOOPCOUSED = 4,
    RELOCALISATION = 5
};

struct ActiveMapDescriptor {
    LocalMapState state{LocalMapState::NEW_LOCAL_MAP};
    int id{-1};
};

class LocalMapManager {
public:
    ActiveMapDescriptor createNewLocalMap(bool isPrimary = false);

private:
    std::vector<ActiveMapDescriptor> localMaps_;
};
}  // namespace SufaceRestruction

#endif  // LOCAL_MAP_MANAGER_H_
