#ifndef RELOCALISER_HPP_
#define RELOCALISER_HPP_

// cpp
#include <fstream>

// FernRelocLib
#include <FernRelocLib/FernConservatory.hpp>
#include <FernRelocLib/PixelUtils.hpp>
#include <FernRelocLib/PoseDatabase.hpp>
#include <FernRelocLib/RelocDatabase.hpp>

// opencv
#include <opencv2/opencv.hpp>

// sophus
#include <sophus/se3.hpp>

namespace FernRelocLib {
class Relocaliser {
private:
    float keyframeHarvestingThreshold;
    FernConservatory* encoding;
    RelocDatabase* relocDatabase;
    PoseDatabase* poseDatabase;
    cv::Mat processedImage1, processedImage2;

public:
    Relocaliser(
        Eigen::Vector2i imgSize, int type, Eigen::Vector2f range, float harvestingThreshold,
        int numFerns, int numDecisionsPerFern) {
        static const int levels = 5;
        encoding =
            new FernConservatory(numFerns, imgSize / (1 << levels), range, numDecisionsPerFern);
        relocDatabase = new RelocDatabase(numFerns, encoding->getNumCodes());
        poseDatabase = new PoseDatabase();
        keyframeHarvestingThreshold = harvestingThreshold;

        processedImage1 = cv::Mat(cv::Size(imgSize(0), imgSize(1)), type);
        processedImage2 = cv::Mat(cv::Size(imgSize(0), imgSize(1)), type);
    }

    ~Relocaliser(void) {
        delete encoding;
        delete relocDatabase;
        delete poseDatabase;
    }

    bool processFrame(
        const cv::Mat* img, const Sophus::SE3f* pose, int sceneId, int k, int nearestNeighbours[],
        float* distances, bool harvestKeyframes) {
        // // downsample and preprocess image => processedImage1
        depthFilterSubsample(img, &processedImage1);
        depthFilterSubsample(&processedImage1, &processedImage2);
        depthFilterSubsample(&processedImage2, &processedImage1);
        depthFilterSubsample(&processedImage1, &processedImage2);

        cv::GaussianBlur(processedImage2, processedImage1, cv::Size(3, 3), 2.5f);

        // compute code
        int codeLength = encoding->getNumFerns();
        char* code = new char[codeLength];
        encoding->computeCode(&processedImage1, code);

        // prepare outputs
        int ret = -1;
        bool releaseDistances = (distances == NULL);
        if (distances == NULL) distances = new float[k];

        // find similar frames
        int similarFound = relocDatabase->findMostSimilar(code, nearestNeighbours, distances, k);

        std::cout << similarFound << std::endl;

        // add keyframe to database
        if (harvestKeyframes) {
            if (similarFound == 0) {
                ret = relocDatabase->addEntry(code);
                std::cout << "Not has similar image" << std::endl;
            } else if (distances[0] > keyframeHarvestingThreshold)
                ret = relocDatabase->addEntry(code);

            if (ret >= 0) poseDatabase->storePose(ret, *pose, sceneId);
        }

        // cleanup and return
        delete[] code;
        if (releaseDistances) delete[] distances;
        return ret >= 0;
    }

    const FernRelocLib::PoseDatabase::PoseInScene& RetrievePose(int id) {
        return poseDatabase->retrievePose(id);
    }

    void saveToDirectory(const std::string& outputDirectory) {
        std::string configFilePath = outputDirectory + "config.txt";
        std::ofstream ofs(configFilePath.c_str());

        // TODO MAKE WORK WITH TEMPLATE - type should change?
        if (!ofs) throw std::runtime_error("Could not open " + configFilePath + " for reading");
        ofs << "type=rgb,levels=4,numFerns=" << encoding->getNumFerns()
            << ",numDecisionsPerFern=" << encoding->getNumDecisions() / 3
            << ",harvestingThreshold=" << keyframeHarvestingThreshold;

        encoding->saveToFile(outputDirectory + "ferns.txt");
        relocDatabase->saveToFile(outputDirectory + "frames.txt");
        poseDatabase->saveToFile(outputDirectory + "poses.txt");
    }

    void loadFromDirectory(const std::string& inputDirectory) {
        std::string fernFilePath = inputDirectory + "ferns.txt";
        std::string frameCodeFilePath = inputDirectory + "frames.txt";
        std::string posesFilePath = inputDirectory + "poses.txt";

        if (!std::ifstream(fernFilePath.c_str()))
            throw std::runtime_error("unable to open " + fernFilePath);
        if (!std::ifstream(frameCodeFilePath.c_str()))
            throw std::runtime_error("unable to open " + frameCodeFilePath);
        if (!std::ifstream(posesFilePath.c_str()))
            throw std::runtime_error("unable to open " + posesFilePath);

        encoding->loadFromFile(fernFilePath);
        relocDatabase->loadFromFile(frameCodeFilePath);
        poseDatabase->loadFromFile(posesFilePath);
    }
};
}  // namespace FernRelocLib

#endif
