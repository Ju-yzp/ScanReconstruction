#ifndef SCAN_RECONSTRUCTION_OCTREE_H_
#define SCAN_RECONSTRUCTION_OCTREE_H_

#include <Constants.h>
#include <MemoryAllocator.h>

#include <Types.h>
#include <Utils.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
namespace ScanReconstruction {

struct Node {
    // 如果當前所在樹深度是k,那麼這個就是k-2層的子節點掩碼，代表是否存在物體表面
    std::atomic<uint64_t> mask = 0;
    // 指向子节点的指针,如果这一层是叶子，那么就是指向数据块的地址，在内存中，它们指针是连续的64个
    Node* node_ptr = nullptr;
    // 原子锁，保护不被其他线程干扰
    std::atomic_flag is_working = ATOMIC_FLAG_INIT;
};

class Octree {
public:
    Octree() { root_ = new Node; }
    void add_nodes(std::vector<uint64_t>& codes) {
        std::vector<void**> ptrs;
        std::sort(codes.begin(), codes.end());
        codes.erase(std::unique(codes.begin(), codes.end()), codes.end());

        for (auto code : codes) {
            Node* current_node = root_;
            for (int level = 0; level < MAX_OCTREE_LEVEL; ++level) {
                int offset = (code >> SHIFTS[level]) & BIT_6_MASK;
                current_node->mask.fetch_or(1ULL << offset, std::memory_order_relaxed);
                while (current_node->is_working.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#endif
                }

                if (!current_node->node_ptr) current_node->node_ptr = new Node[64]();
                current_node->is_working.clear(std::memory_order_release);
                current_node = &current_node->node_ptr[offset];
                if (level == MAX_OCTREE_LEVEL - 1) {
                    ptrs.emplace_back((void**)&current_node->node_ptr);
                    break;
                }
            }
        }
        vb_allocator_.allocate(ptrs);
    }

    bool has_voxel_block(uint64_t code) {
        Node* current_node = root_;
        for (int level = 0; level < MAX_OCTREE_LEVEL; ++level) {
            int offset = (code >> SHIFTS[level]) & BIT_6_MASK;

            uint64_t mask_val = current_node->mask.load(std::memory_order_relaxed);
            if (!(mask_val & (1ULL << offset))) {
                return false;
            }

            if (!current_node->node_ptr) return false;

            current_node = &current_node->node_ptr[offset];
        }
        return current_node->node_ptr != nullptr;
    }

    Voxel* has_leaf_node(uint64_t code) {
        Node* current_node = root_;
        for (int level = 0; level < MAX_OCTREE_LEVEL; ++level) {
            int offset = (code >> SHIFTS[level]) & BIT_6_MASK;

            if (!current_node->node_ptr) return nullptr;
            Node* next_node = &current_node->node_ptr[offset];

            if (level == MAX_OCTREE_LEVEL - 1) {
                return reinterpret_cast<Voxel*>(next_node->node_ptr);
            }

            current_node = next_node;
        }
        return nullptr;
    }

    void polygonizeBlock(
        const Voxel* block, const Eigen::Vector3i& blockOrigin, float voxel_size,
        std::vector<Triangle>& local_tris) {
        for (int z = 0; z < VOXEL_BLOCK_SIZE; ++z) {
            for (int y = 0; y < VOXEL_BLOCK_SIZE; ++y) {
                for (int x = 0; x < VOXEL_BLOCK_SIZE; ++x) {
                    Eigen::Vector3i pos(x, y, z);
                    float val[8];
                    Eigen::Vector3f pts[8];
                    static const Eigen::Vector3i corners[8] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                                               {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                                               {1, 1, 1}, {0, 1, 1}};
                    int cubeindex = 0;
                    for (int i = 0; i < 8; ++i) {
                        Eigen::Vector3i p = pos + corners[i];
                        val[i] = shortToFloat(block[p.x() + p.y() * 10 + p.z() * 100].sdf);
                        pts[i] = (blockOrigin + pos + corners[i]).cast<float>() * voxel_size;
                        if (val[i] < 0.0f) cubeindex |= (1 << i);
                    }
                    if (cubeindex == 0 || cubeindex == 255) continue;
                    if (edgeTable[cubeindex] == 0) continue;
                    auto interpolate = [&](int i1, int i2) -> Eigen::Vector3f {
                        float v1 = val[i1];
                        float v2 = val[i2];
                        float t = 0.5f;
                        if (std::abs(v2 - v1) > 1e-6f) t = -v1 / (v2 - v1);
                        return pts[i1] + t * (pts[i2] - pts[i1]);
                    };
                    Eigen::Vector3f vlist[12];
                    if (edgeTable[cubeindex] & 1) vlist[0] = interpolate(0, 1);
                    if (edgeTable[cubeindex] & 2) vlist[1] = interpolate(1, 2);
                    if (edgeTable[cubeindex] & 4) vlist[2] = interpolate(2, 3);
                    if (edgeTable[cubeindex] & 8) vlist[3] = interpolate(3, 0);
                    if (edgeTable[cubeindex] & 16) vlist[4] = interpolate(4, 5);
                    if (edgeTable[cubeindex] & 32) vlist[5] = interpolate(5, 6);
                    if (edgeTable[cubeindex] & 64) vlist[6] = interpolate(6, 7);
                    if (edgeTable[cubeindex] & 128) vlist[7] = interpolate(7, 4);
                    if (edgeTable[cubeindex] & 256) vlist[8] = interpolate(0, 4);
                    if (edgeTable[cubeindex] & 512) vlist[9] = interpolate(1, 5);
                    if (edgeTable[cubeindex] & 1024) vlist[10] = interpolate(2, 6);
                    if (edgeTable[cubeindex] & 2048) vlist[11] = interpolate(3, 7);
                    for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
                        Triangle tri;
                        tri.v[0] = vlist[triTable[cubeindex][i]];
                        tri.v[1] = vlist[triTable[cubeindex][i + 1]];
                        tri.v[2] = vlist[triTable[cubeindex][i + 2]];
                        tri.normal = (tri.v[1] - tri.v[0]).cross(tri.v[2] - tri.v[0]).normalized();
                        tri.attribute = 0;
                        local_tris.push_back(tri);
                    }
                }
            }
        }
    }

    void exportToSTL(const std::string& filename, float voxel_size) {
        struct LeafData {
            Voxel* ptr;
            Eigen::Vector3i origin;
        };
        std::vector<LeafData> leaf_list;
        std::function<void(Node*, int, uint64_t)> collect = [&](Node* node, int level,
                                                                uint64_t code) {
            if (!node || !node->node_ptr) return;
            uint64_t mask_val = node->mask.load(std::memory_order_acquire);
            for (int i = 0; i < 64; ++i) {
                if (mask_val & (1ULL << i)) {
                    uint64_t next_code = code | (static_cast<uint64_t>(i) << SHIFTS[level]);
                    Node* next_node = &node->node_ptr[i];
                    if (level == MAX_OCTREE_LEVEL - 1) {
                        if (next_node->node_ptr) {
                            leaf_list.push_back(
                                {reinterpret_cast<Voxel*>(next_node->node_ptr),
                                 decode(next_code) * VOXEL_BLOCK_SIZE});
                        }
                    } else {
                        collect(next_node, level + 1, next_code);
                    }
                }
            }
        };
        collect(root_, 0, 0);
        std::vector<Triangle> all_tris = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, leaf_list.size()), std::vector<Triangle>(),
            [&](const tbb::blocked_range<size_t>& r, std::vector<Triangle> init) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    polygonizeBlock(leaf_list[i].ptr, leaf_list[i].origin, voxel_size, init);
                }
                return init;
            },
            [](std::vector<Triangle> a, std::vector<Triangle> b) {
                a.insert(a.end(), b.begin(), b.end());
                return a;
            });
        std::ofstream os(filename, std::ios::binary);
        if (!os) return;
        char header[80] = {0};
        os.write(header, 80);
        uint32_t num_tris = static_cast<uint32_t>(all_tris.size());
        os.write(reinterpret_cast<const char*>(&num_tris), 4);
        for (const auto& tri : all_tris) {
            os.write(reinterpret_cast<const char*>(tri.normal.data()), 12);
            os.write(reinterpret_cast<const char*>(tri.v[0].data()), 12);
            os.write(reinterpret_cast<const char*>(tri.v[1].data()), 12);
            os.write(reinterpret_cast<const char*>(tri.v[2].data()), 12);
            os.write(reinterpret_cast<const char*>(&tri.attribute), 2);
        }
        os.close();
    }

private:
    Node* root_;
    VoxelBlockAllocator vb_allocator_;
};
}  // namespace ScanReconstruction

#endif
