//
// Created by phajder on 11/11/2022.
//

#ifndef KERAS_OCR_CONTAINERS_H_
#define KERAS_OCR_CONTAINERS_H_

#include <vector>
#include <opencv2/core/types.hpp>

namespace keras_ocr {
/**
 * Container representing single box.
 */
struct Box {
  std::vector<cv::Point2f> coords;
};

/**
 * Container representing single detection output from CRAFT model.
 */
struct DetectionOutput {
  std::vector<Box> boxes;
};

/**
 * Container representing single pipeline output.
 */
struct PipelineOutput {
  int frame;
  std::string text;
  Box box;
};
} // namespace keras_ocr

#endif // KERAS_OCR_CONTAINERS_H_
