#pragma once

namespace mot {

struct DetectorConfig {
    int input_width{640};
    int input_height{640};
    double conf_threshold{0.10};
    double iou_threshold{0.70};
    int max_det{300};
    int class_count{10};

    void validate() const;
};

}
