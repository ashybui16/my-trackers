#include "tensorrt/postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mot::detail {
namespace {

constexpr std::size_t kMaxNmsCandidates = 30000;

using FloatBox = std::array<float, 4>;

float floatIou(const FloatBox& first, const FloatBox& second, float first_area, float second_area) {
    // Theo TorchNMS: diện tích box và phép chia đều dùng float32.
    const float left = std::max(first[0], second[0]);
    const float top = std::max(first[1], second[1]);
    const float right = std::min(first[2], second[2]);
    const float bottom = std::min(first[3], second[3]);
    const float intersection_width = std::max(0.0F, right - left);
    const float intersection_height = std::max(0.0F, bottom - top);
    const float intersection = intersection_width * intersection_height;
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

void greedyNms(OutputWorkspace& workspace,
                                   float iou_threshold,
                                   int max_det) {
    auto& candidates = workspace.candidates;
    // Điểm giảm dần; source index làm thứ tự bằng điểm xác định và ổn định.
    const auto compare = [](const RawCandidate& first, const RawCandidate& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }
        return first.source_index < second.source_index;
    };
    // Keep full sort: partial_sort regresses near the 30000-candidate cap
    // (measured on 30001/34000 heavily overlapping, tied-score candidates).
    std::sort(candidates.begin(), candidates.end(), compare);
    if (candidates.size() > kMaxNmsCandidates) {
        candidates.resize(kMaxNmsCandidates);
    }

    auto& suppressed = workspace.suppressed;
    auto& keep = workspace.keep;
    auto& areas = workspace.areas;
    suppressed.assign(candidates.size(), 0);
    keep.clear();
    areas.resize(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& box = candidates[i].box;
        areas[i] = (box[2] - box[0]) * (box[3] - box[1]);
    }
    keep.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(max_det)));
    for (std::size_t index = 0; index < candidates.size() && keep.size() < static_cast<std::size_t>(max_det); ++index) {
        if (suppressed[index]) {
            continue;
        }
        keep.push_back(index);
        if (keep.size() == static_cast<std::size_t>(max_det)) break;
        for (std::size_t other = index + 1; other < candidates.size(); ++other) {
            if (!suppressed[other] && candidates[index].class_id == candidates[other].class_id &&
                floatIou(candidates[index].box, candidates[other].box, areas[index], areas[other]) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
}

std::vector<mot::Detection> processRaw(const float* output,
                                               int candidate_count,
                                               const OutputOptions& options, OutputWorkspace& workspace) {
    auto at = [output, candidate_count](int candidate, int feature) -> float {
        return output[static_cast<std::size_t>(feature) * candidate_count + candidate];
    };

    auto& candidates = workspace.candidates;
    candidates.clear();
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

    greedyNms(workspace, static_cast<float>(options.iou_threshold), options.max_det);
    const auto& keep = workspace.keep;

    // Ultralytics giới hạn max_det trước scale/clip; box rỗng bị loại mà không bù thêm.
    const float scale = static_cast<float>(options.transform.scale);
    const float pad_x = static_cast<float>(options.transform.pad_x);
    const float pad_y = static_cast<float>(options.transform.pad_y);
    const float max_x = static_cast<float>(options.transform.original_width);
    const float max_y = static_cast<float>(options.transform.original_height);
    std::vector<mot::Detection> detections;
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
        detections.emplace_back(mot::BBox{box[0], box[1], box[2], box[3]},
                                candidate.score,
                                candidate.class_id);
    }
    return detections;
}

}  // namespace

std::vector<mot::Detection> processOutput(const float* output,
                                                  int candidate_count,
                                                  const OutputOptions& options, OutputWorkspace& workspace) {
    if (output == nullptr || candidate_count <= 0) {
        throw std::invalid_argument("detector output and dimensions must be valid");
    }
    if (options.class_count <= 0 || options.max_det <= 0 ||
        !std::isfinite(options.conf_threshold) || options.conf_threshold < 0.0 || options.conf_threshold > 1.0 ||
        !std::isfinite(options.iou_threshold) || options.iou_threshold < 0.0 || options.iou_threshold > 1.0 ||
        !std::isfinite(options.transform.scale) || options.transform.scale <= 0.0 ||
        options.transform.original_width <= 0 || options.transform.original_height <= 0) {
        throw std::invalid_argument("invalid detector output postprocess options");
    }

    return processRaw(output, candidate_count, options, workspace);
}

std::vector<mot::Detection> processOutput(const float* output, int candidate_count,
                                                  const OutputOptions& options) {
    OutputWorkspace workspace;
    return processOutput(output, candidate_count, options, workspace);
}

}  // namespace mot::detail
