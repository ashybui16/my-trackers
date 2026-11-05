#include "mot/visualization/annotator.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace mot {
namespace {

const std::array<cv::Scalar, 10> kClassColors{{
    cv::Scalar(80, 220, 255),  // yellow
    cv::Scalar(120, 255, 120), // green
    cv::Scalar(255, 210, 80),  // blue
    cv::Scalar(80, 160, 255),  // orange
    cv::Scalar(240, 130, 255), // pink
    cv::Scalar(255, 220, 120), // light blue
    cv::Scalar(120, 255, 220), // mint
    cv::Scalar(200, 150, 255), // violet
    cv::Scalar(80, 255, 255),  // bright yellow
    cv::Scalar(255, 180, 180)  // pale blue
}};

cv::Scalar classColor(int class_id) {
    const int color_count = static_cast<int>(kClassColors.size());
    const int color_index = (class_id % color_count + color_count) % color_count;
    return kClassColors[static_cast<std::size_t>(color_index)];
}

std::string className(const std::vector<std::string>& class_names, int class_id) {
    if (class_id >= 0 && static_cast<std::size_t>(class_id) < class_names.size() &&
        !class_names[static_cast<std::size_t>(class_id)].empty()) {
        return class_names[static_cast<std::size_t>(class_id)];
    }
    return "Class " + std::to_string(class_id);
}

void drawBoxAndLabel(cv::Mat& frame, const mot::BBox& box,
                     int class_id, const std::vector<std::string>& class_names,
                     const std::string& suffix) {
    const int width = frame.cols;
    const int height = frame.rows;
    const cv::Scalar color = classColor(class_id);
    if (!std::all_of(box.begin(), box.end(), [](double value) { return std::isfinite(value); }))
        return;
    const auto pixel = [](double value, int limit) {
        return static_cast<int>(std::lround(std::clamp(value, 0.0, static_cast<double>(limit - 1))));
    };
    const int x1 = pixel(box[0], width);
    const int y1 = pixel(box[1], height);
    const int x2 = pixel(box[2], width);
    const int y2 = pixel(box[3], height);
    if (x2 <= x1 || y2 <= y1)
        return;

    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_AA);
    const std::string label = className(class_names, class_id) + suffix;
    constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
    constexpr int text_thickness = 1;
    constexpr int padding_x = 5;
    constexpr int padding_y = 3;
    if (width <= 2 * padding_x || height <= 2 * padding_y)
        return;
    double font_scale = std::clamp(std::min(width, height) / 720.0 * 0.55, 0.42, 0.68);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font, font_scale, text_thickness, &baseline);
    if (text_size.width + 2 * padding_x > width) {
        font_scale *= static_cast<double>(width - 2 * padding_x) / text_size.width;
        text_size = cv::getTextSize(label, font, font_scale, text_thickness, &baseline);
    }
    const int label_width = text_size.width + 2 * padding_x;
    const int label_height = text_size.height + baseline + 2 * padding_y;
    if (label_width > width || label_height > height)
        return;
    const int label_x = std::clamp(x1, 0, width - label_width);
    const int label_y = std::clamp(y1 >= label_height ? y1 - label_height : y1,
                                   0, height - label_height);
    const cv::Rect label_rect(label_x, label_y, label_width, label_height);
    const double brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2];
    const cv::Scalar text_color = brightness >= 150.0
        ? cv::Scalar(16, 16, 16) : cv::Scalar(255, 255, 255);
    cv::rectangle(frame, label_rect, color, cv::FILLED);
    cv::putText(frame, label,
                cv::Point(label_x + padding_x, label_y + padding_y + text_size.height),
                font, font_scale, text_color, text_thickness, cv::LINE_AA);
}

}  // namespace

Annotator::Annotator(std::vector<std::string> class_names)
    : class_names_(std::move(class_names)) {}

void Annotator::draw(cv::Mat& bgr_frame,
                        const std::vector<mot::TrackOutput>& tracks) const {
    if (bgr_frame.empty())
        return;

    for (const auto& track : tracks) {
        drawBoxAndLabel(bgr_frame, track.bbox_xyxy, track.class_id, class_names_,
                        "  ID " + std::to_string(track.track_id));
    }
}

void Annotator::drawDetections(
    cv::Mat& bgr_frame, const std::vector<mot::Detection>& detections) const {
    if (bgr_frame.empty())
        return;

    for (const auto& detection : detections) {
        drawBoxAndLabel(bgr_frame, detection.bbox_xyxy, detection.class_id, class_names_, "");
    }
}

}  // namespace mot
