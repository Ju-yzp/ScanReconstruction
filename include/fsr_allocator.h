#ifndef SCAN_RECONSTRUCTION_ALLOCATOR_H_
#define SCAN_RECONSTRUCTION_ALLOCATOR_H_

#include <Types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstddef>
#include <mutex>
#include <stdexcept>

namespace ScanReconstruction {

class PageAllocator {
public:
    PageAllocator() { page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE)); }

    void* allocate(size_t allocate_page_num) {
        if (allocate_page_num == 0) return nullptr;
        size_t total_size = allocate_page_num * page_size_;
        void* address = mmap(
            NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
            -1, 0);
        if (address == MAP_FAILED) throw std::runtime_error("Failed to allocate virtual memory.");

        madvise(address, total_size, MADV_WILLNEED | MADV_HUGEPAGE);
        return address;
    }

    size_t get_page_size() const { return page_size_; }

private:
    // 操作系統頁大小
    size_t page_size_;
};

class VoxelBlockAllocator {
public:
    VoxelBlockAllocator(bool enable_page_algin) : enable_page_align_(enable_page_algin) {
        page_allocator_ = new PageAllocator;
        voxel_block_size_ = sizeof(VoxelBlock);
        enable_page_algin = false;
    }

    ~VoxelBlockAllocator() { delete page_allocator_; }

    void reverse(size_t reverse_num) {
        std::unique_lock<std::mutex> lock(m_);
        size_t total_size = voxel_block_size_ * reverse_num;
        size_t reverse_page_num = 0;
        // if (enable_page_align_) {
        //     if (voxel_block_size_ > page_allocator_->get_page_size()) {
        //         // reverse_page_num =
        //     }else{

        //     }
        // } else
        //     reverse_page_num = total_size / page_allocator_->get_page_size() + 1;

        address_ = page_allocator_->allocate(reverse_page_num);
        free_size_ = reverse_page_num * page_allocator_->get_page_size();
    }

    VoxelBlock* allocate() {
        std::unique_lock<std::mutex> lock(m_);
        if (free_size_ < voxel_block_size_) return nullptr;

        VoxelBlock* allocate_memory = reinterpret_cast<VoxelBlock*>(address_);
        if (enable_page_align_)
            address_ = static_cast<void*>(static_cast<char*>(address_) + per_voxel_block_pages_);
        else
            address_ = static_cast<void*>(static_cast<char*>(address_) + voxel_block_size_);
        return allocate_memory;
    }

    void swapMemoryToDisk() {}

    void swapDiskToMemory() {}

private:
    PageAllocator* page_allocator_;

    size_t voxel_block_size_;

    void* address_;

    size_t free_size_;

    bool enable_page_align_;

    std::mutex m_;

    size_t per_voxel_block_pages_;
};
}  // namespace ScanReconstruction

#endif
