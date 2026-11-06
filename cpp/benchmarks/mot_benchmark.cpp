// VisDrone-MOT dataset speed benchmark: times decode/detect/track per frame.
#include "mot/pipeline/config.hpp"
#if defined(MOT_HAS_TENSORRT)
#include "mot/tensorrt/detector.hpp"
#endif
#if defined(MOT_HAS_ONNXRUNTIME)
#include "mot/onnxruntime/detector.hpp"
#endif
#include "mot/tracking/factory.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
    int warmup = 20, repeats = 1, max_frames = 0;
    std::string config = "weights/config.yaml", model, dataset = "data/VisDroneMOT/test",
                output = "runs/benchmarks/visdrone_test";
    std::vector<std::string> sequences;
};

const char *kUsage =
    "Usage: mot_benchmark [--config YAML] [--model PATH] [--dataset DIR] [--output-dir DIR]\n"
    "                     [--sequences a,b,...] [--warmup 20] [--repeats 1] [--max-frames N]\n"
    "  --dataset     VisDrone split dir with sequences/<seq>/0000001.jpg (default data/VisDroneMOT/test)\n"
    "  --model       overrides YAML model_path (.engine/.trt/.plan -> TensorRT, .onnx -> ONNX Runtime)\n"
    "  --warmup      untimed detector passes on the first frame before each repeat\n"
    "  --max-frames  per-sequence frame cap (0 = all)\n"
    "Writes the timing CSV to stdout and <output-dir>/timing.csv.\n";

int parseInt(const std::string &value) {
    std::size_t end = 0;
    const int number = std::stoi(value, &end);
    if (end != value.size())
        throw std::invalid_argument("invalid integer: " + value);
    return number;
}

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << kUsage;
            std::exit(0);
        }
        if (++i == argc)
            throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--config")
            o.config = value;
        else if (key == "--model" || key == "--engine")
            o.model = value;
        else if (key == "--dataset")
            o.dataset = value;
        else if (key == "--output-dir")
            o.output = value;
        else if (key == "--sequences") {
            std::stringstream stream(value);
            for (std::string name; std::getline(stream, name, ',');)
                if (!name.empty())
                    o.sequences.push_back(name);
        } else if (key == "--warmup")
            o.warmup = parseInt(value);
        else if (key == "--repeats")
            o.repeats = parseInt(value);
        else if (key == "--max-frames")
            o.max_frames = parseInt(value);
        else
            throw std::invalid_argument("unknown option: " + key + "\n" + kUsage);
    }
    if (o.warmup < 0 || o.repeats < 1 || o.max_frames < 0)
        throw std::invalid_argument("warmup >= 0, repeats > 0, max-frames >= 0 required");
    return o;
}

std::unique_ptr<mot::IDetector> makeDetector(const mot::PipelineConfig &config, std::string &backend) {
    std::string extension = fs::path(config.model_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".onnx") {
#if defined(MOT_HAS_ONNXRUNTIME)
        backend = "onnxruntime (CPU)";
        return std::make_unique<mot::OnnxRuntimeDetector>(config.model_path, config.detector,
                                                          config.class_names);
#else
        throw std::runtime_error("ONNX model given but mot_benchmark was built without ONNX Runtime: " +
                                 config.model_path);
#endif
    }
    if (extension == ".engine" || extension == ".trt" || extension == ".plan") {
#if defined(MOT_HAS_TENSORRT)
        backend = "tensorrt";
        return std::make_unique<mot::TensorRTDetector>(config.model_path, config.detector, config.class_names);
#else
        throw std::runtime_error("TensorRT engine given but mot_benchmark was built without TensorRT: " +
                                 config.model_path);
#endif
    }
    throw std::runtime_error("cannot pick a detector backend for '" + config.model_path +
                             "' (expected .engine/.trt/.plan or .onnx)");
}

struct Sequence {
    std::string name;
    std::vector<fs::path> frames;
};

std::vector<fs::path> sequenceFrames(const fs::path &dir) {
    std::vector<std::pair<long, fs::path>> numbered;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
            continue;
        const std::string stem = entry.path().stem().string();
        if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return std::isdigit(c); }))
            throw std::runtime_error("non-numeric frame name in " + dir.string() + ": " + stem);
        numbered.emplace_back(std::stol(stem), entry.path());
    }
    if (numbered.empty())
        throw std::runtime_error("no image frames in " + dir.string());
    std::sort(numbered.begin(), numbered.end());
    std::vector<fs::path> frames;
    frames.reserve(numbered.size());
    for (std::size_t i = 0; i < numbered.size(); ++i) {
        if (numbered[i].first != static_cast<long>(i + 1))
            throw std::runtime_error("frames must be numbered continuously from 1 in " + dir.string());
        frames.push_back(numbered[i].second);
    }
    return frames;
}

std::vector<Sequence> discover(const Options &o) {
    const fs::path root = fs::path(o.dataset) / "sequences";
    if (!fs::is_directory(root))
        throw std::runtime_error("missing sequences directory: " + root.string());
    std::vector<std::string> names = o.sequences;
    if (names.empty()) {
        for (const auto &entry : fs::directory_iterator(root))
            if (entry.is_directory())
                names.push_back(entry.path().filename().string());
        std::sort(names.begin(), names.end());
    }
    if (names.empty())
        throw std::runtime_error("no sequences found in " + root.string());
    std::vector<Sequence> sequences;
    for (const auto &name : names) {
        const fs::path dir = root / name;
        if (!fs::is_directory(dir))
            throw std::runtime_error("sequence not found: " + dir.string());
        auto frames = sequenceFrames(dir);
        if (o.max_frames > 0 && frames.size() > static_cast<std::size_t>(o.max_frames))
            frames.resize(o.max_frames);
        sequences.push_back({name, std::move(frames)});
    }
    return sequences;
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct StageTimes {
    std::vector<double> decode, detector, tracker, end_to_end;
    void append(const StageTimes &other) {
        auto add = [](std::vector<double> &to, const std::vector<double> &from) {
            to.insert(to.end(), from.begin(), from.end());
        };
        add(decode, other.decode);
        add(detector, other.detector);
        add(tracker, other.tracker);
        add(end_to_end, other.end_to_end);
    }
};

void reportStage(std::ostream &out, const std::string &scope, int repeat, const std::string &stage,
                 std::vector<double> times) {
    if (times.empty())
        throw std::runtime_error("no measured samples for " + scope + "/" + stage);
    const double total = std::accumulate(times.begin(), times.end(), 0.0);
    std::sort(times.begin(), times.end());
    const std::size_t n = times.size();
    const double median = n % 2 ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2.0;
    const std::size_t p95 = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
    out << scope << ',' << repeat << ',' << stage << ',' << n << ',' << total / n << ',' << median << ','
        << times[p95] << ',' << 1000.0 * n / total << '\n';
}

void reportAll(std::ostream &out, const std::string &scope, int repeat, const StageTimes &t) {
    reportStage(out, scope, repeat, "decode", t.decode);
    reportStage(out, scope, repeat, "detector", t.detector);
    reportStage(out, scope, repeat, "tracker", t.tracker);
    reportStage(out, scope, repeat, "end_to_end", t.end_to_end);
}

cv::Mat readFrame(const fs::path &path) {
    cv::Mat frame = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (frame.empty())
        throw std::runtime_error("cannot read image: " + path.string());
    return frame;
}

StageTimes runSequence(const Sequence &sequence, const mot::PipelineConfig &config, mot::IDetector &detector) {
    const auto tracker = mot::createTracker(config.tracker);
    StageTimes t;
    for (std::size_t index = 0; index < sequence.frames.size(); ++index) {
        const auto start = Clock::now();
        const cv::Mat frame = readFrame(sequence.frames[index]);
        const double decode = elapsed(start);
        auto tick = Clock::now();
        const auto detections = detector.detect(frame);
        const double detect = elapsed(tick);
        tick = Clock::now();
        tracker->update(detections, frame);
        const double track = elapsed(tick);
        t.decode.push_back(decode);
        t.detector.push_back(detect);
        t.tracker.push_back(track);
        t.end_to_end.push_back(elapsed(start));
    }
    return t;
}

void run(const Options &o, const mot::PipelineConfig &config) {
    const auto sequences = discover(o);
    std::size_t total_frames = 0;
    for (const auto &s : sequences)
        total_frames += s.frames.size();
    const fs::path output = o.output;
    fs::create_directories(output);
    std::ofstream timing(output / "timing.csv");
    if (!timing)
        throw std::runtime_error("cannot write " + (output / "timing.csv").string());
    for (auto *stream : {static_cast<std::ostream *>(&std::cout), static_cast<std::ostream *>(&timing)})
        *stream << std::fixed << std::setprecision(6)
                << "scope,repeat,stage,samples,mean_ms,median_ms,p95_ms,fps\n";

    for (int rep = 1; rep <= o.repeats; ++rep) {
        std::string backend;
        const auto detector = makeDetector(config, backend);
        std::cerr << "repeat " << rep << ": backend=" << backend << ", model=" << config.model_path
                  << ", sequences=" << sequences.size() << ", frames=" << total_frames << '\n';
        const cv::Mat first = readFrame(sequences.front().frames.front());
        for (int i = 0; i < o.warmup; ++i)
            detector->detect(first);

        StageTimes all;
        for (const auto &sequence : sequences) {
            const auto times = runSequence(sequence, config, *detector);
            for (auto *stream : {static_cast<std::ostream *>(&std::cout), static_cast<std::ostream *>(&timing)})
                reportAll(*stream, sequence.name, rep, times);
            all.append(times);
        }
        for (auto *stream : {static_cast<std::ostream *>(&std::cout), static_cast<std::ostream *>(&timing)})
            reportAll(*stream, "ALL", rep, all);
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        const auto o = parse(argc, argv);
        auto config = mot::PipelineConfig::fromFile(o.config);
        if (!o.model.empty())
            config.model_path = o.model;
        run(o, config);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
