#include "detection/yolo/postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mot::detail {
namespace {

constexpr std::size_t kMaxNmsCandidates = 30000;

using FloatBox = std::array<float, 4>;

float floatIou(const FloatBox& first, const FloatBox& second, float first_area, float second_area) {
    // Matches TorchNMS: box areas and the division both use float32.
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

constexpr std::size_t kNoNext = static_cast<std::size_t>(-1);

// Sorts candidates by score (source index breaks ties), caps them, caches their areas, and links each
// candidate to the next one of the same class (NMS is class-aware, so only those can suppress each other).
void sortCandidates(OutputWorkspace& workspace, int class_count) {
    auto& candidates = workspace.candidates;
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
    auto& areas = workspace.areas;
    areas.resize(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& box = candidates[i].box;
        areas[i] = (box[2] - box[0]) * (box[3] - box[1]);
    }
    auto& next_same_class = workspace.next_same_class;
    next_same_class.resize(candidates.size());
    workspace.last_of_class.assign(static_cast<std::size_t>(class_count), kNoNext);
    for (std::size_t i = candidates.size(); i-- > 0;) {
        auto& last = workspace.last_of_class[static_cast<std::size_t>(candidates[i].class_id)];
        next_same_class[i] = last;
        last = i;
    }
}

// Class-aware greedy NMS over sorted candidates; fills `keep` with candidate indices.
void greedyNms(OutputWorkspace& workspace,
               float iou_threshold,
               int max_det,
               std::vector<std::size_t>& keep) {
    const auto& candidates = workspace.candidates;
    const auto& areas = workspace.areas;
    const auto& next_same_class = workspace.next_same_class;
    auto& suppressed = workspace.suppressed;
    suppressed.assign(candidates.size(), 0);
    keep.clear();
    keep.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(max_det)));
    for (std::size_t index = 0; index < candidates.size() && keep.size() < static_cast<std::size_t>(max_det); ++index) {
        if (suppressed[index]) {
            continue;
        }
        keep.push_back(index);
        if (keep.size() == static_cast<std::size_t>(max_det)) break;
        for (std::size_t other = next_same_class[index]; other != kNoNext; other = next_same_class[other]) {
            if (!suppressed[other] &&
                floatIou(candidates[index].box, candidates[other].box, areas[index], areas[other]) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
}

// Runs the tight NMS and the loose (recovery) NMS in one pass, sharing IoU computations. Each pass has
// its own suppression flags and max_det stop, so the results equal two separate greedyNms calls.
void greedyNmsPair(OutputWorkspace& workspace,
                   float tight_threshold,
                   float loose_threshold,
                   int max_det,
                   std::vector<std::size_t>& tight_keep,
                   std::vector<std::size_t>& loose_keep) {
    const auto& candidates = workspace.candidates;
    const auto& areas = workspace.areas;
    const auto& next_same_class = workspace.next_same_class;
    auto& tight_suppressed = workspace.suppressed;
    auto& loose_suppressed = workspace.loose_suppressed;
    tight_suppressed.assign(candidates.size(), 0);
    loose_suppressed.assign(candidates.size(), 0);
    const auto limit = static_cast<std::size_t>(max_det);
    tight_keep.clear();
    loose_keep.clear();
    tight_keep.reserve(std::min(candidates.size(), limit));
    loose_keep.reserve(std::min(candidates.size(), limit));
    bool tight_done = false;
    bool loose_done = false;
    for (std::size_t index = 0; index < candidates.size() && !(tight_done && loose_done); ++index) {
        bool tight = !tight_done && !tight_suppressed[index];
        bool loose = !loose_done && !loose_suppressed[index];
        if (tight) {
            tight_keep.push_back(index);
            if (tight_keep.size() == limit) tight_done = true, tight = false;
        }
        if (loose) {
            loose_keep.push_back(index);
            if (loose_keep.size() == limit) loose_done = true, loose = false;
        }
        if (!tight && !loose) continue;
        for (std::size_t other = next_same_class[index]; other != kNoNext; other = next_same_class[other]) {
            const bool check_tight = tight && !tight_suppressed[other];
            const bool check_loose = loose && !loose_suppressed[other];
            if (!check_tight && !check_loose) continue;
            const float overlap = floatIou(candidates[index].box, candidates[other].box, areas[index], areas[other]);
            if (check_tight && overlap > tight_threshold) tight_suppressed[other] = true;
            if (check_loose && overlap > loose_threshold) loose_suppressed[other] = true;
        }
    }
}

std::vector<mot::Detection> processRaw(const float* output,
                                       int candidate_count,
                                       const OutputOptions& options, OutputWorkspace& workspace,
                                       std::vector<mot::Detection>* recovered) {
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

    sortCandidates(workspace, options.class_count);
    if (recovered == nullptr) {
        greedyNms(workspace, static_cast<float>(options.iou_threshold), options.max_det, workspace.keep);
    } else {
        greedyNmsPair(workspace, static_cast<float>(options.iou_threshold), static_cast<float>(kLooseNmsIou),
                      options.max_det, workspace.keep, workspace.loose_keep);
    }

    // Ultralytics caps max_det before scale/clip; empty boxes are dropped without backfilling.
    const float scale = static_cast<float>(options.transform.scale);
    const float pad_x = static_cast<float>(options.transform.pad_x);
    const float pad_y = static_cast<float>(options.transform.pad_y);
    const float max_x = static_cast<float>(options.transform.original_width);
    const float max_y = static_cast<float>(options.transform.original_height);
    // Maps kept candidates to image-space detections, skipping boxes that are empty after clipping.
    const auto collect = [&](const std::vector<std::size_t>& keep, std::vector<mot::Detection>& out,
                             std::vector<FloatBox>* boxes) {
        out.clear();
        out.reserve(keep.size());
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
            std::vector<float> class_scores(static_cast<std::size_t>(options.class_count));
            for (int class_index = 0; class_index < options.class_count; ++class_index) {
                class_scores[static_cast<std::size_t>(class_index)] =
                    std::clamp(at(candidate.source_index, 4 + class_index), 0.0F, 1.0F);
            }
            if (boxes) boxes->push_back(box);
            out.emplace_back(mot::BBox{box[0], box[1], box[2], box[3]},
                             candidate.score,
                             candidate.class_id,
                             std::move(class_scores));
        }
    };
    std::vector<mot::Detection> detections;
    if (recovered == nullptr) {
        collect(workspace.keep, detections, nullptr);
        return detections;
    }

    // Recovered detections (Ultralytics TrackTrack `compute_dets_del`): kept by a looser NMS but not
    // near-duplicates (IoU >= kLooseNmsDedupIou) of any tight detection.
    recovered->clear();
    std::vector<FloatBox> tight_boxes;
    std::vector<FloatBox> loose_boxes;
    collect(workspace.keep, detections, &tight_boxes);
    std::vector<mot::Detection> loose;
    collect(workspace.loose_keep, loose, &loose_boxes);
    if (tight_boxes.empty() || loose_boxes.empty()) {
        return detections;
    }
    const auto area = [](const FloatBox& box) { return (box[2] - box[0]) * (box[3] - box[1]); };
    for (std::size_t i = 0; i < loose.size(); ++i) {
        float max_iou = 0.0F;
        for (const auto& tight : tight_boxes) {
            const float inter = std::max(std::min(loose_boxes[i][2], tight[2]) - std::max(loose_boxes[i][0], tight[0]), 0.0F) *
                                std::max(std::min(loose_boxes[i][3], tight[3]) - std::max(loose_boxes[i][1], tight[1]), 0.0F);
            // torchvision box_iou-style with Ultralytics' eps.
            max_iou = std::max(max_iou, inter / (area(loose_boxes[i]) + area(tight) - inter + 1e-7F));
        }
        if (max_iou < static_cast<float>(kLooseNmsDedupIou)) {
            recovered->push_back(std::move(loose[i]));
        }
    }
    return detections;
}

}

std::vector<mot::Detection> processOutput(const float* output,
                                          int candidate_count,
                                          const OutputOptions& options, OutputWorkspace& workspace,
                                          std::vector<mot::Detection>* recovered) {
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

    return processRaw(output, candidate_count, options, workspace, recovered);
}

std::vector<mot::Detection> processOutput(const float* output, int candidate_count,
                                                  const OutputOptions& options) {
    OutputWorkspace workspace;
    return processOutput(output, candidate_count, options, workspace, nullptr);
}

}
