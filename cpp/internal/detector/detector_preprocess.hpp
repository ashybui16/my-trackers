#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace mot_inference::detail {
struct LetterboxInfo {
    double scale{1.0};
    int pad_x{0}, pad_y{0};
};
inline LetterboxInfo preprocess(const cv::Mat &frame, cv::Mat &canvas, float *chw) {
    if (frame.empty() || frame.dims != 2 || frame.type() != CV_8UC3)
        throw std::invalid_argument("frame must be a non-empty 2D CV_8UC3 BGR image");
    LetterboxInfo info;
    info.scale = std::min(static_cast<double>(canvas.cols) / frame.cols,
                          static_cast<double>(canvas.rows) / frame.rows);
    const int width = std::max(1, static_cast<int>(std::round(frame.cols * info.scale)));
    const int height = std::max(1, static_cast<int>(std::round(frame.rows * info.scale)));
    info.pad_x = (canvas.cols - width) / 2;
    info.pad_y = (canvas.rows - height) / 2;
    const cv::Scalar padding(114, 114, 114);
    if (info.pad_y)
        canvas.rowRange(0, info.pad_y).setTo(padding);
    if (info.pad_y + height < canvas.rows)
        canvas.rowRange(info.pad_y + height, canvas.rows).setTo(padding);
    if (info.pad_x)
        canvas(cv::Rect(0, info.pad_y, info.pad_x, height)).setTo(padding);
    if (info.pad_x + width < canvas.cols)
        canvas(cv::Rect(info.pad_x + width, info.pad_y, canvas.cols - info.pad_x - width, height))
            .setTo(padding);
    cv::Mat resized = canvas(cv::Rect(info.pad_x, info.pad_y, width, height));
    cv::resize(frame, resized, resized.size());
    const std::size_t plane = static_cast<std::size_t>(canvas.rows) * canvas.cols;
    for (int row = 0; row < canvas.rows; ++row) {
        const auto *pixels = canvas.ptr<cv::Vec3b>(row);
        const auto offset = static_cast<std::size_t>(row) * canvas.cols;
        float *red = chw + offset;
        float *green = chw + plane + offset;
        float *blue = chw + 2 * plane + offset;
        for (int col = 0; col < canvas.cols; ++col) {
            red[col] = static_cast<float>(pixels[col][2]) / 255.0F;
            green[col] = static_cast<float>(pixels[col][1]) / 255.0F;
            blue[col] = static_cast<float>(pixels[col][0]) / 255.0F;
        }
    }
    return info;
}
} // namespace mot_inference::detail
