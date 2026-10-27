#include "detector/detector_postprocess.hpp"

#include "mot_tracker/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mot_inference::detail {
namespace {

constexpr std::size_t kMaxNmsCandidates = 30000;

using FloatBox = std::array<float, 4>;

struct RawCandidate {
    FloatBox box{};
    float score{0.0F};
    int class_id{0};
    int source_index{0};
};

struct LegacyCandidate {
    mot_tracker::BBox box{};
    double score{0.0};
    int class_id{0};
    int source_index{0};
};

float floatIou(const FloatBox& first, const FloatBox& second) {
    // Theo TorchNMS: diện tích box và phép chia đều dùng float32.
    const float left = std::max(first[0], second[0]);
    const float top = std::max(first[1], second[1]);
    const float right = std::min(first[2], second[2]);
    const float bottom = std::min(first[3], second[3]);
    const float intersection_width = std::max(0.0F, right - left);
    const float intersection_height = std::max(0.0F, bottom - top);
    const float intersection = intersection_width * intersection_height;
    const float first_area = (first[2] - first[0]) * (first[3] - first[1]);
    const float second_area = (second[2] - second[0]) * (second[3] - second[1]);
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<std::size_t> greedyNms(std::vector<RawCandidate>& candidates,
                                   float iou_threshold,
                                   int max_det) {
    // Điểm giảm dần; source index làm thứ tự bằng điểm xác định và ổn định.
    std::sort(candidates.begin(), candidates.end(), [](const RawCandidate& first, const RawCandidate& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }
        return first.source_index < second.source_index;
    });
    if (candidates.size() > kMaxNmsCandidates) {
        candidates.resize(kMaxNmsCandidates);
    }

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<std::size_t> keep;
    keep.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(max_det)));
    for (std::size_t index = 0; index < candidates.size() && keep.size() < static_cast<std::size_t>(max_det); ++index) {
        if (suppressed[index]) {
            continue;
        }
        keep.push_back(index);
        for (std::size_t other = index + 1; other < candidates.size(); ++other) {
            if (!suppressed[other] && candidates[index].class_id == candidates[other].class_id &&
                floatIou(candidates[index].box, candidates[other].box) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return keep;
}

std::vector<mot_tracker::Detection> processRaw(const float* output,
                                               int dimension_1,
                                               int dimension_2,
                                               bool feature_first,
                                               const OutputOptions& options) {
    const int candidate_count = feature_first ? dimension_2 : dimension_1;
    auto at = [output, dimension_2, feature_first](int candidate, int feature) -> float {
        const std::size_t offset = feature_first
                                       ? static_cast<std::size_t>(feature) * dimension_2 + candidate
                                       : static_cast<std::size_t>(candidate) * dimension_2 + feature;
        return output[offset];
    };

    std::vector<RawCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(candidate_count));
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        float score = at(candidate, 4);
        int class_id = 0;
        bool has_nan_class = std::isnan(score);
        for (int class_index = 1; class_index < options.class_count; ++class_index) {
            const float class_score = at(candidate, 4 + class_index);
            if (std::isnan(class_score)) {
                has_nan_class = true;
                break;
            }
            if (class_score > score) {
                score = class_score;
                class_id = class_index;
            }
        }
        if (has_nan_class || !std::isfinite(score) ||
            !(static_cast<double>(score) > options.conf_threshold)) {
            continue;
        }

        const float center_x = at(candidate, 0);
        const float center_y = at(candidate, 1);
        const float width = at(candidate, 2);
        const float height = at(candidate, 3);
        candidates.push_back({{center_x - width / 2.0F,
                               center_y - height / 2.0F,
                               center_x + width / 2.0F,
                               center_y + height / 2.0F},
                              score,
                              class_id,
                              candidate});
    }

    const std::vector<std::size_t> keep =
        greedyNms(candidates, static_cast<float>(options.iou_threshold), options.max_det);

    // Ultralytics giới hạn max_det trước scale/clip; box rỗng bị loại mà không bù thêm.
    const float scale = static_cast<float>(options.transform.scale);
    const float pad_x = static_cast<float>(options.transform.pad_x);
    const float pad_y = static_cast<float>(options.transform.pad_y);
    const float max_x = static_cast<float>(options.transform.original_width);
    const float max_y = static_cast<float>(options.transform.original_height);
    std::vector<mot_tracker::Detection> detections;
    detections.reserve(keep.size());
    for (std::size_t index : keep) {
        const RawCandidate& candidate = candidates[index];
        FloatBox box{
            (candidate.box[0] - pad_x) / scale,
            (candidate.box[1] - pad_y) / scale,
            (candidate.box[2] - pad_x) / scale,
            (candidate.box[3] - pad_y) / scale,
        };
        if (!std::all_of(box.begin(), box.end(), [](float value) { return std::isfinite(value); })) {
            continue;
        }
        for (int coordinate : {0, 2}) {
            box[coordinate] = std::clamp(box[coordinate], 0.0F, max_x);
        }
        for (int coordinate : {1, 3}) {
            box[coordinate] = std::clamp(box[coordinate], 0.0F, max_y);
        }
        if (box[2] <= box[0] || box[3] <= box[1]) {
            continue;
        }
        detections.emplace_back(mot_tracker::BBox{box[0], box[1], box[2], box[3]},
                                candidate.score,
                                candidate.class_id);
    }
    return detections;
}

std::vector<mot_tracker::Detection> processPostprocessed(const float* output,
                                                         int candidate_count,
                                                         const OutputOptions& options) {
    // Giữ nguyên hành vi cũ của output [1,N,6], ngoài phạm vi đồng bộ NMS raw.
    std::vector<LegacyCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(candidate_count));
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        const std::size_t offset = static_cast<std::size_t>(candidate) * 6;
        const float raw_class = output[offset + 5];
        if (!std::isfinite(raw_class)) {
            continue;
        }
        const double score = output[offset + 4];
        const int class_id = static_cast<int>(std::lround(raw_class));
        if (!std::isfinite(score) || score < options.conf_threshold || class_id < 0 ||
            class_id >= options.class_count) {
            continue;
        }
        const double scale = options.transform.scale;
        const mot_tracker::BBox box{
            (static_cast<double>(output[offset]) - options.transform.pad_x) / scale,
            (static_cast<double>(output[offset + 1]) - options.transform.pad_y) / scale,
            (static_cast<double>(output[offset + 2]) - options.transform.pad_x) / scale,
            (static_cast<double>(output[offset + 3]) - options.transform.pad_y) / scale,
        };
        try {
            mot_tracker::validateBBox(box);
        } catch (const std::invalid_argument&) {
            continue;
        }
        candidates.push_back({box, score, class_id, candidate});
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const LegacyCandidate& first,
                                                              const LegacyCandidate& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }
        return first.source_index < second.source_index;
    });
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<mot_tracker::Detection> result;
    result.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(options.max_det)));
    for (std::size_t index = 0;
         index < candidates.size() && static_cast<int>(result.size()) < options.max_det;
         ++index) {
        if (suppressed[index]) {
            continue;
        }
        result.emplace_back(candidates[index].box, candidates[index].score, candidates[index].class_id);
        for (std::size_t other = index + 1; other < candidates.size(); ++other) {
            if (!suppressed[other] && candidates[other].class_id == candidates[index].class_id &&
                mot_tracker::iou(candidates[index].box, candidates[other].box) > options.iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return result;
}

}  // namespace

std::vector<mot_tracker::Detection> processOutput(const float* output,
                                                  int dimension_1,
                                                  int dimension_2,
                                                  const OutputOptions& options) {
    if (output == nullptr || dimension_1 <= 0 || dimension_2 <= 0) {
        throw std::invalid_argument("detector output and dimensions must be valid");
    }
    if (options.class_count <= 0 || options.max_det <= 0 ||
        !std::isfinite(options.conf_threshold) || options.conf_threshold < 0.0 || options.conf_threshold > 1.0 ||
        !std::isfinite(options.iou_threshold) || options.iou_threshold < 0.0 || options.iou_threshold > 1.0 ||
        !std::isfinite(options.transform.scale) || options.transform.scale <= 0.0 ||
        options.transform.original_width <= 0 || options.transform.original_height <= 0) {
        throw std::invalid_argument("invalid detector output postprocess options");
    }

    const int feature_count = 4 + options.class_count;
    const bool raw_feature_first = dimension_1 == feature_count;
    const bool raw_candidate_first = dimension_2 == feature_count;
    if (raw_feature_first || raw_candidate_first) {
        return processRaw(output, dimension_1, dimension_2, raw_feature_first, options);
    }
    if (dimension_2 == 6) {
        return processPostprocessed(output, dimension_1, options);
    }
    throw std::runtime_error("unsupported detector output shape; expected [1," +
                             std::to_string(feature_count) + ",N], [1,N," +
                             std::to_string(feature_count) + "], or [1,N,6]");
}

}  // namespace mot_inference::detail
