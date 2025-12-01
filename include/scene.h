#ifndef SCENE_H_
#define SCENE_H_

// eigen
#include <Eigen/Eigen>

// cpp
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>

namespace SufaceRestruction {
// 一个体素块边长/一个体素边长
constexpr int SDF_BLOCK_SIZE = 8;

// 一个体素块所容纳的体素个数
constexpr int SDF_BLOCK_SIZE3 = 512;

// 哈希表掩码
constexpr uint32_t SDF_HASH_MASK = 0xfffff;

constexpr uint32_t SDF_BUCKET_NUM = 0x100000;

constexpr uint32_t SDF_EXCESS_LIST_SIZE = 0x20000;

// 哈希表实体
struct HashEntry {
    // 指向的体素块的坐标
    Eigen::Vector3i pos{Eigen::Vector3i::Zero()};
    // 发生冲突时，指向的冲突列表元素的全局偏移量
    int offset{-1};
    // 指向的体素块地址偏移量
    int ptr;
    // 该所对应的体素块是否被使用标志
    bool isUsed{false};
};

// 体素
struct Voxel {
    float sdf = 1.0f;
    float w_depth = 0;
};

struct SceneParams {
    // 截断距离
    float mu;

    // 最大权重
    float maxWeight;

    // 一个体素的边长，其实就是建图的分辨率
    float voxelSize;
};

// TODO：当前版本不支持动态内存
class Scene {
public:
    Scene(const SceneParams& sceneParams) : sceneParams_(sceneParams) {
        hashEntries_ = new HashEntry[SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE];
        voxel_ = new Voxel[(SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) * SDF_BLOCK_SIZE3];
#pragma omp parallel for
        for (int i = 0; i < (SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE); ++i) hashEntries_[i].ptr = i;
        freeExcessEntries_.resize(SDF_EXCESS_LIST_SIZE);
        for (int i = 0; i < SDF_EXCESS_LIST_SIZE; ++i)
            freeExcessEntries_[i] = SDF_BUCKET_NUM + i + 1;
    }

    ~Scene() {
        delete[] voxel_;
        delete[] hashEntries_;
    }

    // 哈希表索引函数
    static int getHashIndex(Eigen::Vector3i voxelBlockPos) {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }

    // 获取体素块地址
    Voxel* get_voxelBolck(int id) { return voxel_ + id * SDF_BLOCK_SIZE3; }

    // 清除上一帧的可见体素块索引并交换上一帧和当前帧的数据指针以及其他数据
    void swapVisibleList() {
        lastFrametVisibleVoxelBlockList_.clear();
        lastFrametVisibleVoxelBlockList_.swap(currentFrameVisibleVoxelBlockIdList_);
    }

    // 设置哈希表entry
    void set_entry(HashEntry hashEntry, int id) { hashEntries_[id] = hashEntry; }

    // 获取相对的哈希表entry
    HashEntry& get_entry(int id) { return hashEntries_[id]; }

    void set_entryOffset(int id, int offset) { hashEntries_[id].offset = offset; }

    // 获取空闲的冲突列表所对应的Entry指针
    std::optional<HashEntry*> get_freeExcessEntry() {
        if (freeExcessEntries_.empty()) return std::nullopt;
        int id = freeExcessEntries_.back();
        freeExcessEntries_.pop_back();
        return &hashEntries_[id];
    }

    SceneParams get_sceneParams() const { return sceneParams_; }

    const std::set<int>& get_constLastFrameVisibleVoxelBlockList() const {
        return lastFrametVisibleVoxelBlockList_;
    }

    std::set<int>& get_lastFrameVisibleVoxelBlockList() { return lastFrametVisibleVoxelBlockList_; }

    const std::set<int>& get_constCurrentFrameVisibleVoxelBlockList() const {
        return currentFrameVisibleVoxelBlockIdList_;
    }

    std::set<int>& get_currentFrameVisibleVoxelBlockList() {
        return currentFrameVisibleVoxelBlockIdList_;
    }

    void freeExcessEntry(int id) {
        if (id - SDF_BUCKET_NUM + 1 >= 0) {
            freeExcessEntries_.push_back(id);
            hashEntries_[id].isUsed = false;
            hashEntries_[id].offset = -1;
        }
    }

    void resetEntry(int id) {
        hashEntries_[id].isUsed = false;
        hashEntries_[id].offset = -1;
    }

private:
    Voxel* voxel_{nullptr};

    HashEntry* hashEntries_{nullptr};

    std::vector<int> freeExcessEntries_;

    SceneParams sceneParams_;

    std::set<int> lastFrametVisibleVoxelBlockList_;

    std::set<int> currentFrameVisibleVoxelBlockIdList_;
};
}  // namespace SufaceRestruction

#endif  // SCENE_H_
