#include <hp_utils.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace ScanReconstruction;

int main() {
    unsigned short w_ = std::numeric_limits<unsigned short>::max();
    float c_w;
    atomUshortToFloat(&w_, &c_w);
    std::cout << c_w << std::endl;
    short sdf_ = -1000;
    float c_sdf;
    atomShortToFloat(&sdf_, &c_sdf);
    atomFloatToShort(&c_sdf, &sdf_);
    std::cout << c_sdf << std::endl;
    std::cout << sdf_ << std::endl;
    const int N = 512;

    std::vector<float> distances(N, 0.0f);
    std::vector<float> depth(N, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_gen(1.0f, 2.0f);
    std::uniform_real_distribution<float> noise_gen(-0.15f, 0.15f);

    for (int i = 0; i < N; ++i) {
        depth[i] = dist_gen(gen);
        distances[i] = depth[i] + noise_gen(gen);
    }

    VoxelBlock* block = new VoxelBlock();
    initVoxelBlock(block);
    for (int k = 0; k < 100; ++k) {
        // 使用指針或確保 block 在棧上完全分配

        // std::cout << "--- 512 Voxels Integration Test ---" << std::endl;

        // 2. 調用你的融合函數
        // 這裡 mu = 0.1, max_weight = 30.0
        integrateWithVoxelBlock(block, distances, depth, 0.1f);

        // 3. 安全打印 (嚴格限制在 10 以內)
        // std::cout << std::left << std::setw(8) << "Index" << std::setw(12) << "Eta" <<
        // std::setw(12)
        //           << "New_SDF" << std::setw(12) << "New_W" << std::endl;
    }

    for (int i = 0; i < 10; ++i) {
        float s, w;
        // 使用你的轉換函數
        atomShortToFloat(&(block->sdf[i]), &s);
        atomUshortToFloat(&(block->weight[i]), &w);

        float eta = depth[i] - distances[i];

        std::cout << std::left << std::setw(8) << i << std::setw(12) << std::fixed
                  << std::setprecision(4) << eta << std::setw(12) << s << std::setw(12) << w
                  << std::endl;
    }
    delete block;
    return 0;
}
