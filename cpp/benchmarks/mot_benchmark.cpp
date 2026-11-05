#include "mot/pipeline/config.hpp"
#include "mot/tracking/factory.hpp"
#include "mot/visualization/annotator.hpp"
#ifdef MOT_BENCHMARK_GPU
#include "mot/tensorrt/detector.hpp"
#else
#include "detection/yolo/postprocess.hpp"
#include "tracking/common/assignment.hpp"
#include "tracking/common/gmc.hpp"
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    int warmup = 100, frames = 1000, repeats = 3;
    std::string config = "weights/config.yaml", video, output = "runs/benchmarks", only;
};
Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "--config YAML --warmup 100 --frames 1000 --repeats 3"
                         " [--only CPU_WORKLOAD] [--video VIDEO --output-dir DIR]\n";
            std::exit(0);
        }
        if (++i == argc)
            throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--only")
            o.only = value;
        else if (key == "--config")
            o.config = value;
        else if (key == "--video")
            o.video = value;
        else if (key == "--output-dir")
            o.output = value;
        else if (key == "--warmup" || key == "--frames" || key == "--repeats") {
            std::size_t end = 0;
            const int number = std::stoi(value, &end);
            if (end != value.size())
                throw std::invalid_argument("invalid integer: " + value);
            if (key == "--warmup")
                o.warmup = number;
            else if (key == "--frames")
                o.frames = number;
            else
                o.repeats = number;
        } else
            throw std::invalid_argument("unknown option: " + key);
    }
    if (o.warmup < 0 || o.frames < 1 || o.repeats < 1)
        throw std::invalid_argument("warmup >= 0, frames/repeats > 0 required");
    return o;
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void report(const std::string &name, int repeat, std::vector<double> times, std::size_t checksum = 0) {
    if (times.empty())
        throw std::runtime_error("no measured samples for " + name);
    const double total = std::accumulate(times.begin(), times.end(), 0.0);
    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();
    const double median = n % 2 ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2.0;
    const std::size_t p95 = static_cast<std::size_t>(std::ceil(0.95 * n)) - 1;
    std::cout << name << ',' << repeat << ',' << n << ',' << total / n << ',' << median << ',' << times[p95]
              << ',' << 1000.0 * n / total << ',' << checksum << '\n';
}
#ifndef MOT_BENCHMARK_GPU
template <class F> void measure(const std::string &name, int repeat, const Options &o, F &&f) {
    if (!o.only.empty() && name != o.only)
        return;
    std::vector<double> times;
    times.reserve(o.frames);
    std::size_t checksum = 0;
    for (int i = 0; i < o.warmup + o.frames; ++i) {
        const auto start = Clock::now();
        const auto value = f(i);
        const double ms = elapsed(start);
        if (i >= o.warmup) {
            times.push_back(ms);
            checksum += value;
        }
    }
    report(name, repeat, std::move(times), checksum);
}
void run(const Options &o, const mot::PipelineConfig &config) {
    // Component workloads use BoT-SORT parameters; other trackers fall back to defaults.
    const auto *selected = std::get_if<mot::BoTSORTConfig>(&config.tracker);
    const mot::BoTSORTConfig botsort = selected ? *selected : mot::BoTSORTConfig{};
    for (int rep = 1; rep <= o.repeats; ++rep) {
        for (auto shape : {std::pair<int, int>{16, 16}, {64, 96}, {128, 128}}) {
            std::vector<std::vector<double>> costs(shape.first, std::vector<double>(shape.second));
            for (int r = 0; r < shape.first; ++r)
                for (int c = 0; c < shape.second; ++c)
                    costs[r][c] = r == c ? 0.1 : 0.5 + ((r * 17 + c * 31) % 50) / 100.0;
            measure(
                "assignment_" + std::to_string(shape.first) + "x" + std::to_string(shape.second), rep, o,
                [&](int) {
                    return mot::linearAssignment(costs, botsort.match_thresh).matches.size();
                });
        }
        for (int n : {128, 2048, 34000}) {
            const int classes = config.detector.class_count;
            std::vector<float> raw(static_cast<std::size_t>(4 + classes) * n, 0.0F);
            for (int i = 0; i < n; ++i) {
                raw[i] = 10.0F + (i % 100) * 6;
                raw[n + i] = 10.0F + ((i / 100) % 100) * 6;
                raw[2 * n + i] = 20;
                raw[3 * n + i] = 20;
                raw[static_cast<std::size_t>(4 + i % classes) * n + i] = 0.1F + 0.8F * (i % 97) / 96.0F;
            }
            mot::detail::OutputOptions output;
            output.class_count = classes;
            output.conf_threshold = config.detector.conf_threshold;
            output.iou_threshold = config.detector.iou_threshold;
            output.max_det = config.detector.max_det;
            output.transform = {1, 0, 0, 1920, 1080};
            mot::detail::OutputWorkspace workspace;
            measure("nms_" + std::to_string(n), rep, o, [&](int) {
                return mot::detail::processOutput(raw.data(), n, output, workspace).size();
            });
        }
        for (int n : {30001, 34000, 100000}) {
            const int classes = config.detector.class_count;
            std::vector<float> raw(static_cast<std::size_t>(4 + classes) * n, 0.0F);
            for (int i = 0; i < n; ++i) {
                raw[i] = 50;
                raw[n + i] = 50;
                raw[2 * n + i] = 20;
                raw[3 * n + i] = 20;
                raw[4 * n + i] = 0.9F;
            }
            mot::detail::OutputOptions output;
            output.class_count = classes;
            output.conf_threshold = config.detector.conf_threshold;
            output.iou_threshold = config.detector.iou_threshold;
            output.max_det = config.detector.max_det;
            output.transform = {1, 0, 0, 100, 100};
            mot::detail::OutputWorkspace workspace;
            measure("nms_overlap_" + std::to_string(n), rep, o, [&](int) {
                return mot::detail::processOutput(raw.data(), n, output, workspace).size();
            });
        }
        for (int n : {16, 64, 128}) {
            auto tracker_config = config.tracker;
            if (auto *bot = std::get_if<mot::BoTSORTConfig>(&tracker_config))
                bot->gmc_method = "none"; // Isolate association/Kalman; GMC measured separately.
            const auto tracker = mot::createTracker(tracker_config);
            measure("tracker_no_gmc_" + std::to_string(n), rep, o, [&](int frame) {
                std::vector<mot::Detection> detections;
                detections.reserve(n);
                for (int i = 0; i < n; ++i)
                    if ((frame + i) % 11 >= 3) {
                        const double x = i * 16.0 + 0.1 * (frame % 20);
                        detections.emplace_back(mot::BBox{x, 0, x + 12, 20}, 0.9,
                                                i % config.detector.class_count);
                    }
                const auto tracks = tracker->update(detections);
                std::size_t sum = tracks.size();
                for (const auto &track : tracks)
                    sum += track.track_id;
                return sum;
            });
        }
        mot::GMC gmc(botsort.gmc_method, botsort.gmc_downscale);
        cv::Mat image(360, 640, CV_8UC3);
        cv::RNG rng(42);
        rng.fill(image, cv::RNG::UNIFORM, 0, 256);
        measure("gmc_640x360", rep, o, [&](int) {
            const auto warp = gmc.apply(image);
            return static_cast<std::size_t>(std::llround(warp.at<double>(0, 0) * 1000000));
        });
    }
}
#else
void run(const Options &o, const mot::PipelineConfig &config) {
    if (!o.only.empty())
        throw std::invalid_argument("--only applies to CPU workloads");
    if (o.video.empty())
        throw std::invalid_argument("--video is required");
    std::filesystem::create_directories(o.output);
    for (int rep = 1; rep <= o.repeats; ++rep) {
        mot::TensorRTDetector detector(config.model_path, config.detector, config.class_names);
        const auto tracker = mot::createTracker(config.tracker);
        mot::Annotator annotator(config.class_names);
        cv::VideoCapture capture(o.video);
        if (!capture.isOpened())
            throw std::runtime_error("cannot open video");
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0)
            fps = 30;
        cv::VideoWriter writer;
        std::vector<double> decode, detect, track, annotate, write, pipeline, total;
        int count = 0;
        for (; count < o.warmup + o.frames; ++count) {
            cv::Mat frame;
            const auto start = Clock::now();
            if (!capture.read(frame))
                break;
            const double a = elapsed(start);
            auto tick = Clock::now();
            const auto detections = detector.detect(frame);
            const double b = elapsed(tick);
            tick = Clock::now();
            const auto tracks = tracker->update(detections, frame);
            const double c = elapsed(tick);
            tick = Clock::now();
            annotator.draw(frame, tracks);
            const double d = elapsed(tick);
            if (!writer.isOpened()) {
                const auto path =
                    std::filesystem::path(o.output) / ("benchmark_" + std::to_string(rep) + ".mp4");
                writer.open(path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
                if (!writer.isOpened())
                    throw std::runtime_error("cannot open video writer");
            }
            tick = Clock::now();
            writer.write(frame);
            const double e = elapsed(tick);
            const double full = elapsed(start);
            if (count >= o.warmup) {
                decode.push_back(a);
                detect.push_back(b);
                track.push_back(c);
                annotate.push_back(d);
                write.push_back(e);
                pipeline.push_back(b + c);
                total.push_back(full);
            }
        }
        const auto finalize = Clock::now();
        writer.release();
        std::cerr << "repeat " << rep << ": processed=" << count << ", measured=" << total.size()
                  << ", writer_finalize_ms=" << elapsed(finalize) << '\n';
        if (total.size() < static_cast<std::size_t>(o.frames))
            std::cerr << "insufficient video frames for requested samples\n";
        report("decode", rep, decode);
        report("detector", rep, detect);
        report("tracker", rep, track);
        report("annotation", rep, annotate);
        report("write", rep, write);
        report("detection_tracking", rep, pipeline);
        report("end_to_end", rep, total);
    }
}
#endif
}
int main(int argc, char **argv) {
    try {
        const auto o = parse(argc, argv);
        const auto config = mot::PipelineConfig::fromFile(o.config);
        std::cout << std::fixed << std::setprecision(6)
                  << "workload,repeat,samples,mean_ms,median_ms,p95_ms,fps,checksum\n";
        run(o, config);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
