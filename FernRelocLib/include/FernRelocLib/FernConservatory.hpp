#ifndef FERNCONSERVATORY_HPP_
#define FERNCONSERVATORY_HPP_

#include <string>

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace FernRelocLib {
struct FernTester {
    Eigen::Vector2i location;
    float threshold;
};

class FernConservatory {
public:
    FernConservatory(
        int numFerns, Eigen::Vector2i imgSize, Eigen::Vector2f bounds, int decisionsPerFern = 1);
    ~FernConservatory(void);

    void computeCode(const cv::Mat* img, char* codeFragments) const;

    void saveToFile(const std::string& fernsFileName);
    void loadFromFile(const std::string& fernsFileName);

    int getNumFerns(void) const { return mNumFerns; }
    int getNumCodes(void) const { return 1 << mNumDecisions; }
    int getNumDecisions(void) const { return mNumDecisions; }

private:
    int mNumFerns;
    int mNumDecisions;
    FernTester* mEncoders;
};
}  // namespace FernRelocLib

#endif
