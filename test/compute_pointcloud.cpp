#include <Tracker/cameraParams.h>
#include <Tracker/pixelUtils.h>
#include <opencv2/core/persistence.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config_file.yaml>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];

    cv::Mat origin = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    cv::Mat convert;
    convertShortToFloat(&origin, &convert, 5000.0f);
    CalibrationParams cp;
    cp.fx = 525.0;
    cp.fy = 525.0;
    cp.cx = 319.5;
    cp.cy = 239.5;

    cv::Mat points = cv::Mat(origin.rows, origin.cols, CV_32FC4);
    for (int y{0}; y < points.rows; ++y)
        for (int x{0}; x < points.cols; ++x) {
            cv::Vec4f p = cp.computePointCloud(convert.at<float>(y, x), cv::Vec2i(x, y));
            std::cout << "x " << p(0) << std::endl;
            std::cout << "y " << p(1) << std::endl;
            std::cout << "z " << p(2) << std::endl;
            std::cout << std::endl;
            points.at<cv::Vec4f>(y, x) = p;
        }
}
