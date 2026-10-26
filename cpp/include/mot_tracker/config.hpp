#pragma once

#include <string>

namespace mot_tracker {

struct BoTSORTConfig {
    // Ngưỡng điểm tin cậy cho lượt ghép chính và lượt ghép điểm thấp.
    double track_high_thresh{0.25};
    double track_low_thresh{0.10};
    // Chỉ tạo track mới khi điểm phát hiện đạt ngưỡng này.
    double new_track_thresh{0.25};
    // Số frame tối đa giữ một track ở trạng thái LOST.
    int track_buffer{30};
    // Giới hạn chi phí ghép; chi phí nhỏ hơn nghĩa là hai hộp giống nhau hơn.
    double match_thresh{0.80};
    // Kết hợp điểm tin cậy của detection vào chi phí ghép.
    bool fuse_score{true};
    // Phương pháp và tỉ lệ thu nhỏ ảnh khi bù chuyển động camera.
    std::string gmc_method{"sparseOptFlow"};
    int gmc_downscale{2};

    // Kiểm tra các ngưỡng và tùy chọn trước khi chạy tracker.
    void validate() const;
};

}  // namespace mot_tracker
