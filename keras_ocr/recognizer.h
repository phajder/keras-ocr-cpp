//
// Created by phajder on 05/11/2022.
//

#ifndef KERAS_OCR_RECOGNIZER_H_
#define KERAS_OCR_RECOGNIZER_H_

#include "cppflow/model.h"
#include "cppflow/tensor.h"
#include "opencv2/imgproc.hpp"

namespace keras_ocr {
/**
 * @class Recognizer
 * @brief A cppflow::model wrapper for serving Text Recognition based on Convolutional Recurrent Neural (CRNN).
 */
class Recognizer {
 private:
  /**
   * Recognition model created using cppflow wrapper
   */
  cppflow::model model_;

  /**
   * Alphabet on which network was trained, used for output decoding.
   */
  const std::string alphabet_ = "0123456789abcdefghijklmnopqrstuvwxyz ";

  /**
   * Maximum word length on which network was trained.
   * Can be computed from output tensor.
   */
  const int max_length_ = 48;

  /**
   * Default name for input op.
   */
  const std::string default_input_op_ = "serving_default_input_2";
 public:
  /**
   * Creates recognizer using path from compiler definitions
   */
  Recognizer() : model_(RECOGNITION_MODEL_PATH) {}

  /**
   * Creates detector by initializing model using custom path to model.
   * Model must be in format supported by cppflow: Frozen_graph of Saved_model.
   *
   * @param modelPath path to recognizer model
   */
  explicit Recognizer(const std::string &modelPath) : model_(modelPath) {}

  /**
   * Crops original image to boxes containing text.
   * If box is rotated, it is warped to normal axis using cv::warpPerspective.
   * At the end box is padded to match model input layer dimensions.
   *
   * @param image image represented in CV_8UC(channels) type and RBG or GRAY color mode
   * @param boxes boxes coordinates containing text
   * @return cropped images with text
   */
  std::vector<cv::Mat> CropToBoxes(const cv::Mat &image, const std::vector<Box> &boxes) const;

  /**
   * Transforms text boxes from Mat to batch input tensor for at least one image.
   * Pixel values are converted to [0;1] to match model expectations.
   * Box images must be continuous, i.e. cannot be a cv::Mat slices,
   * otherwise memory access violation will occur during tensor creation.
   *
   * @param boxes boxes with text as images
   * @return batch input tensor used for inference
   */
  cppflow::tensor PreprocessInput(std::vector<cv::Mat> &boxes) const;

  /**
   * Performs input inference.
   *
   * @param input batch four-dimensional tensor, consisting of at least single image
   * @return model output as four-dimensional tensor
   */
  cppflow::tensor Inference(const cppflow::tensor &input);

  /**
   * Translates numeric output from network to meaningful text using alphabet.
   * Batches are split and represented as separate string, which are put to
   * output vector. If first detected letter from any batch is -1, then entire
   * output from such batch is skipped.
   *
   * @param output four-dimensional output tensor from inference
   * @return vector of recognition outputs, each with corresponding text string
   */
  std::vector<std::string> PostprocessOutput(const cppflow::tensor &output) const;
};
} // namespace keras_ocr


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace keras_ocr {
std::vector<cv::Mat> Recognizer::CropToBoxes(const cv::Mat &image, const std::vector<Box> &boxes) const {
  auto input_shape = model_.get_operation_shape(default_input_op_);
  std::vector<cv::Mat> crops;

  for (auto &[box] : boxes) {
    // Skipping get rotated box because it produces same output
    float w = (norm(box[0] - box[1]) + norm(box[2] - box[3])) / 2.,
        h = (norm(box[0] - box[3]) + norm(box[1] - box[2])) / 2.;
    int target_height = input_shape[1], target_width = input_shape[2];
    float scale = std::min(target_width / w, target_height / h);
    float m_transform[4][2] = {
        {0., 0.},
        {scale * w, 0.},
        {scale * w, scale * h},
        {0., scale * h}
    };
    cv::Mat crop, transform(4, 2, CV_32FC1, &m_transform);
    transform = getPerspectiveTransform(box, transform);
    warpPerspective(image, crop, transform, cv::Size_<float>{scale * w, scale * h});

    crop = PadImage(crop, target_height, target_width);
    crops.emplace_back(crop);
  }

  return crops;
}

cppflow::tensor Recognizer::PreprocessInput(std::vector<cv::Mat> &boxes) const {
  assert(!boxes.empty() && "Input batch can be created only if there is at least one box!");
  auto input_shape = model_.get_operation_shape(default_input_op_);
  int model_channels = static_cast<int>(input_shape[3]);
  int shape[4] = {
      1,
      boxes[0].size[0],
      boxes[0].size[1],
      model_channels
  };

  std::vector<cppflow::tensor> input_tensors;
  for (auto &box : boxes) {
    if (model_channels == 1 && box.channels() == 3) cvtColor(box, box, cv::COLOR_RGB2GRAY);
    box.convertTo(box, CV_32FC(model_channels), 1. / 255);
    assert(box.isContinuous() && "Input boxes must be continuous to create a batch - Mat slices are not allowed!");

    auto array = Mat2Vector<float>(box);
    input_tensors.emplace_back(cppflow::tensor(
        array,
        {shape, shape + 4}
    ));
  }

  return input_tensors.size() > 1
         ? cppflow::concat({0}, input_tensors)
         : input_tensors[0];
}

cppflow::tensor Recognizer::Inference(const cppflow::tensor &input) {
  auto output = model_(
      {{default_input_op_ + ":0", input}},
      {"StatefulPartitionedCall:0"}
  )[0];

  return output;
}

std::vector<std::string> Recognizer::PostprocessOutput(const cppflow::tensor &output) const {
  auto ocr_output = output.get_data<int64_t>();
  std::vector<std::string> recognized_text;
  std::string tmp;
  int i = 0;

  while (i < ocr_output.size()) {
    if (ocr_output[i] == -1 || tmp.length() == max_length_) {
      i += max_length_ - tmp.length(); // move iterator to the next word
      recognized_text.emplace_back(tmp);
      tmp.clear();
      continue;
    }
    tmp += alphabet_[ocr_output[i++]];
  }

  return recognized_text;
}
} // namespace keras_ocr

#endif // KERAS_OCR_RECOGNIZER_H_
