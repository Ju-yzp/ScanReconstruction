#include <Tracker/pixelUtils.h>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config_file.yaml>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];

    cv::Mat origin = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    cv::Mat convert;
    convertShortToFloat(&origin, &convert, 5000.0f);
    cv::Mat subsample;
    filterSubsampleWithHoles(&convert, &subsample);

    cv::namedWindow("Origin", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Subsample", cv::WINDOW_AUTOSIZE);

    cv::imshow("Origin", convert);
    cv::imshow("Subsample", subsample);
    cv::waitKey();
    cv::destroyWindow("Origin");
    cv::destroyWindow("Subsample");
    return 0;
}
