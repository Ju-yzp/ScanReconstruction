#ifndef SCAN_RECONSTRUCTION_MEMORY_ALLOCATOR_H_
#define SCAN_RECONSTRUCTION_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <Types.h>

namespace ScanReconstruction {
class PageMemoryPool {
public:
    PageMemoryPool() { page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE)); }
    // page_num申请的页数量
    void* allocate(size_t page_num) {
        size_t total_bytes = page_num * page_size_;
        void* address = mmap(
            NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
            -1, 0);
        if (address == MAP_FAILED) throw std::runtime_error("Failed to allocate virtual memory.");

        madvise(address, total_bytes, MADV_WILLNEED | MADV_HUGEPAGE);
        return address;
    }

    size_t get_page_size() const { return page_size_; }

private:
    // 操作系统页面大小
    size_t page_size_;
};

static std::function<void(void*)> func = [](void* ptr) {
    Voxel* current_voxel_block = static_cast<Voxel*>(ptr);
    for (size_t i = 0; i < EXPANDED_VOXEL_BLOCK_SIZE3; ++i) {
        Voxel& current_voxel = current_voxel_block[i];
        current_voxel.sdf = std::numeric_limits<short>::max();
        current_voxel.depth_weight = std::numeric_limits<unsigned short>::min();
    }
};

class VoxelBlockAllocator {
public:
    VoxelBlockAllocator() { page_manager_ = new PageMemoryPool; }

    ~VoxelBlockAllocator() { delete page_manager_; }
    void allocate(std::vector<void**> ptrs) {
        if (ptrs.empty()) return;
        size_t allocate_page_num = ptrs.size();
        std::cout << allocate_page_num << std::endl;
        void* addr = page_manager_->allocate(allocate_page_num);

        for (size_t i = 0; i < ptrs.size(); ++i) {
            void* current_page_ptr =
                static_cast<char*>(addr) + (i * page_manager_->get_page_size());
            *ptrs[i] = current_page_ptr;

            func(current_page_ptr);
        }
    }

private:
    PageMemoryPool* page_manager_;
};

}  // namespace ScanReconstruction

#endif
