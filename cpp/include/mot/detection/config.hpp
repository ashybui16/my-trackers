#pragma once

namespace mot {

// Tham số chung cho detector kiểu YOLO (đầu vào cố định, lọc điểm và NMS).
struct DetectorConfig {
    // Kích thước phải khớp input cố định [1,3,H,W] của engine.
    int input_width{640};
    int input_height{640};
    // Ngưỡng điểm tin cậy và IoU dùng để lọc detection/NMS.
    double conf_threshold{0.01};
    double iou_threshold{0.70};
    // Số detection tối đa sau lọc và số lớp mà engine dự đoán.
    int max_det{300};
    int class_count{10};

    // Kiểm tra kích thước, số lượng và các ngưỡng đầu vào.
    void validate() const;
};

}  // namespace mot
