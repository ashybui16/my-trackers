#pragma once

// Pool bookkeeping shared by SORT-family trackers, generic over the track type.
// A track type needs: trackId(), bbox(), frameId(), startFrame() and applyWarp(const cv::Mat&).

#include "tracking/common/geometry.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mot {

template <typename T>
using TrackPtrOf = std::shared_ptr<T>;

namespace pool_detail {

// Sorted track IDs; for pools of a few hundred tracks this is faster than a hash set.
template <typename T>
std::vector<int> sortedIds(const std::vector<TrackPtrOf<T>>& tracks) {
    std::vector<int> ids;
    ids.reserve(tracks.size());
    for (const auto& track : tracks) ids.push_back(track->trackId());
    std::sort(ids.begin(), ids.end());
    return ids;
}

inline bool contains(const std::vector<int>& sorted_ids, int id) {
    return std::binary_search(sorted_ids.begin(), sorted_ids.end(), id);
}

// True when the boxes cannot overlap, so their IoU is exactly 0 (also for inverted boxes).
inline bool disjoint(const BBox& a, const BBox& b) {
    return a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1];
}

}

template <typename T>
std::vector<TrackPtrOf<T>> jointTracks(const std::vector<TrackPtrOf<T>>& first,
                                       const std::vector<TrackPtrOf<T>>& second) {
    // `first` wins on ID collisions; a duplicate ID inside `second` is kept once (first occurrence).
    const std::vector<int> first_ids = pool_detail::sortedIds<T>(first);
    const std::vector<int> second_ids = pool_detail::sortedIds<T>(second);
    const bool second_unique = std::adjacent_find(second_ids.begin(), second_ids.end()) == second_ids.end();
    std::vector<TrackPtrOf<T>> result = first;
    result.reserve(first.size() + second.size());
    std::unordered_set<int> added;  // Only needed when `second` repeats an ID.
    for (const auto& track : second) {
        const int id = track->trackId();
        if (pool_detail::contains(first_ids, id) || (!second_unique && !added.insert(id).second)) {
            continue;
        }
        result.push_back(track);
    }
    return result;
}

template <typename T>
std::vector<TrackPtrOf<T>> subtractTracks(const std::vector<TrackPtrOf<T>>& first,
                                          const std::vector<TrackPtrOf<T>>& second) {
    const std::vector<int> ids = pool_detail::sortedIds<T>(second);
    std::vector<TrackPtrOf<T>> result;
    result.reserve(first.size());
    for (const auto& track : first) {
        if (!pool_detail::contains(ids, track->trackId())) {
            result.push_back(track);
        }
    }
    return result;
}

template <typename T>
void applyGMC(const std::vector<TrackPtrOf<T>>& tracks, const cv::Mat& warp) {
    for (const auto& track : tracks) {
        track->applyWarp(warp);
    }
}

template <typename Ptr>
struct BasicDuplicateResult {
    std::vector<Ptr> tracked;
    std::vector<Ptr> lost;
    std::vector<Ptr> removed;
};

// Drops the shorter-lived track of each tracked/lost pair with IoU > iou_threshold (Ultralytics order).
template <typename T>
BasicDuplicateResult<TrackPtrOf<T>> removeDuplicateTracks(const std::vector<TrackPtrOf<T>>& tracked,
                                                          const std::vector<TrackPtrOf<T>>& lost,
                                                          double iou_threshold = 0.85) {
    if (iou_threshold < 0.0 || iou_threshold > 1.0) {
        throw std::invalid_argument("iou_threshold must be in [0, 1]");
    }
    BasicDuplicateResult<TrackPtrOf<T>> result{tracked, lost, {}};
    if (tracked.empty() || lost.empty()) {
        return result;
    }
    std::vector<BBox> tracked_boxes;
    std::vector<BBox> lost_boxes;
    tracked_boxes.reserve(tracked.size());
    lost_boxes.reserve(lost.size());
    for (const auto& track : tracked) tracked_boxes.push_back(track->bbox());
    for (const auto& track : lost) lost_boxes.push_back(track->bbox());
    std::unordered_set<int> drop_tracked;
    std::unordered_set<int> drop_lost;
    drop_tracked.reserve(tracked.size());
    drop_lost.reserve(lost.size());
    for (std::size_t tracked_index = 0; tracked_index < tracked.size(); ++tracked_index) {
        for (std::size_t lost_index = 0; lost_index < lost.size(); ++lost_index) {
            if (pool_detail::disjoint(tracked_boxes[tracked_index], lost_boxes[lost_index])) {
                continue;  // IoU 0: never a duplicate.
            }
            const double distance = 1.0F - static_cast<float>(iou(tracked_boxes[tracked_index], lost_boxes[lost_index]));
            if (distance >= 1.0 - iou_threshold) {
                continue;
            }
            const auto& active = tracked[tracked_index];
            const auto& stale = lost[lost_index];
            const int active_life = active->frameId() - active->startFrame();
            const int stale_life = stale->frameId() - stale->startFrame();
            if (active_life > stale_life) {
                drop_lost.insert(static_cast<int>(lost_index));
                result.removed.push_back(stale);
            } else {
                drop_tracked.insert(static_cast<int>(tracked_index));
                result.removed.push_back(active);
            }
        }
    }
    result.tracked.clear();
    result.lost.clear();
    for (std::size_t index = 0; index < tracked.size(); ++index) {
        if (drop_tracked.find(static_cast<int>(index)) == drop_tracked.end()) {
            result.tracked.push_back(tracked[index]);
        }
    }
    for (std::size_t index = 0; index < lost.size(); ++index) {
        if (drop_lost.find(static_cast<int>(index)) == drop_lost.end()) {
            result.lost.push_back(lost[index]);
        }
    }
    return result;
}

}
