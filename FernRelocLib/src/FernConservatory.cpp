#include <FernRelocLib/FernConservatory.hpp>

#include <opencv2/core/hal/interface.h>
#include <fstream>

namespace FernRelocLib {

static float random_uniform01(void) { return (float)rand() / (float)RAND_MAX; }

FernConservatory::FernConservatory(
    int numFerns, Eigen::Vector2i imgSize, Eigen::Vector2f bounds, int decisionsPerFern) {
    mNumFerns = numFerns;
    mNumDecisions = decisionsPerFern;
    mEncoders = new FernTester[mNumFerns * decisionsPerFern];
    for (int f = 0; f < mNumFerns * decisionsPerFern; ++f) {
        mEncoders[f].location(0) = (int)floor(random_uniform01() * imgSize(0));
        mEncoders[f].location(1) = (int)floor(random_uniform01() * imgSize(1));
        mEncoders[f].threshold = random_uniform01() * (bounds(1) - bounds(0)) + bounds(0);
    }
}

FernConservatory::~FernConservatory(void) { delete[] mEncoders; }

void FernConservatory::computeCode(const cv::Mat* img, char* codeFragments) const {
    if (img->type() == CV_32F) {
        const float* data = (const float*)img->data;
        for (int i{0}; i < mNumFerns; ++i) {
            codeFragments[i] = 0;
            for (int d{0}; d < mNumDecisions; ++d) {
                const FernTester* tester = &(mEncoders[i * mNumDecisions + d]);
                int loc_id = tester->location(0) + tester->location(1) * img->rows;
                float val = data[loc_id];

                codeFragments[i] |= ((val < tester->threshold) ? 0 : 1) << d;
            }
        }
    }
    // else if(img->type() == ){

    // }
}

void FernConservatory::saveToFile(const std::string& fernsFileName) {
    std::ofstream ofs(fernsFileName.c_str());

    if (!ofs) throw std::runtime_error("Could not open " + fernsFileName + " for reading");
    ;

    for (int f = 0; f < mNumFerns * mNumDecisions; ++f)
        ofs << mEncoders[f].location(0) << ' ' << mEncoders[f].location(1) << ' '
            << mEncoders[f].threshold << '\n';
}

void FernConservatory::loadFromFile(const std::string& fernsFileName) {
    std::ifstream ifs(fernsFileName.c_str());
    if (!ifs) throw std::runtime_error("unable to load " + fernsFileName);

    for (int i = 0; i < mNumFerns; i++) {
        for (int j = 0; j < mNumDecisions; j++) {
            FernTester& fernTester = mEncoders[i * mNumDecisions + j];
            ifs >> fernTester.location(0) >> fernTester.location(1) >> fernTester.threshold;
        }
    }
}
}  // namespace FernRelocLib
