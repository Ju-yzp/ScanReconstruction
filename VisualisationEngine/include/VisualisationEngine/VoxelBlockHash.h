#ifndef VOXELBLOCKHASH_H_
#define VOXELBLOCKHASH_H_

#include <Eigen/Eigen>
#include <climits>
#include <cstdint>
#include <optional>

// 一个体素块边长/一个体素边长
constexpr int SDF_BLOCK_SIZE = 8;

// 哈希表掩码
constexpr uint32_t SDF_HASH_MASK = 0xfffff;

constexpr uint32_t SDF_BUCKET_NUM = 0x100000;

constexpr uint32_t SDF_EXCESS_LIST_SIZE = 0x20000;

// 内存管理(固定内存池)
template <typename T>
class MemoryMannager {
public:
    MemoryMannager(int max_capacity) {
        ptr_ = new T[max_capacity];
        free_blocks_.reserve(max_capacity);
        for (int i{0}; i < max_capacity; ++i) free_blocks_[i] = i;
    }

    std::optional<T*> allocate() {
        if (free_blocks_.empty())
            return std::nullopt;
        else
            return ptr_[free_blocks_.pop_back()];
    }

    void free(T* ptr) { free_blocks_.push_back(ptr - ptr_); }

    T& operator()(int id) { return ptr_[id]; }

private:
    // 指向申请的堆内存
    T* ptr_;

    // 存储空闲指针的栈
    std::vector<int> free_blocks_;
};

// 哈希表实体
struct HashEntry {
    Eigen::Vector3i pos;
    int offset;
    int ptr;
};

// 体素
struct Voxel {
    short sdf;
    u_char w_depth;
};

// 体素哈希表
class VoxelBlockHash {
public:
    static const int noTotalEntries = SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE;

    inline static int getHashIndex(Eigen::Vector3i blockPos) {
        return (((uint)blockPos(0) * 73856093u) ^ ((uint)blockPos(1) * 19349669u) ^
                ((uint)blockPos(2) * 83492791u)) &
               (uint)SDF_HASH_MASK;
    }

    VoxelBlockHash() : hashEntries_(noTotalEntries) {}

private:
    MemoryMannager<HashEntry> hashEntries_;
};
#endif
