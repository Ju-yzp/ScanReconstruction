#include <Allcator.h>

#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>

namespace ScanReconstruction {
Allcator::Allcator(size_t reversed_size, std::function<void(void*)> callback)
    : reversed_size_(reversed_size), initialize_callback_(callback) {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    address_ =
        mmap(NULL, reversed_size_ * page_size_, PROT_READ | PROT_WRITE, MAP_ANONYMOUS, -1, 0);
    if (address_ == MAP_FAILED)
        throw std::runtime_error("Failed to allcate virtual memory that user need ");
}

void Allcator::allocate(uint32_t offset) {
    initialize_callback_(static_cast<void*>((char*)address_ + offset * page_size_));
}
}  // namespace ScanReconstruction
