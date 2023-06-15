//
// Created by phajder on 09/11/2022.
//

#ifndef KERAS_OCR_PIPELINE_H_
#define KERAS_OCR_PIPELINE_H_

#include "containers.h"
#include "utils.h"
#include "detector.h"
#include "recognizer.h"

namespace keras_ocr {
/**
 * @class Pipeline
 * @brief Text Detection and Recognition pipeline using CRAFT detector and CRNN recognizer.
 */
class Pipeline {
 private:
  /**
   * CRAFT detector, used for text detection.
   */
  Detector detector_;

  /**
   * CRNN text recognizer, used usually with boxes obtained through detection.
   */
  Recognizer recognizer_;

  /**
   * Image scaling coefficient, used at the beginning of the pipeline.
   */
  double scale_ = 1.;

  /**
   * Maximum image size, used for computational and memory optimization.
   * Currently unused.
   */
  int max_image_size_ = 2048;
 public:
  Pipeline() = default;

  /**
   * Creates detector and recognizer using provided model paths.
   * Models must be in supported format, defined in cppflow: Frozen_graph or Saved_model.
   *
   * @param detector_path path to detector model
   * @param recognizer_path path to recognizer model
   */
  Pipeline(const std::string &detector_path, const std::string &recognizer_path)
      : detector_(detector_path), recognizer_(recognizer_path) {};

  /**
   * Performs prediction pipeline.
   * Pipeline starts from image resizing and padding with zeros to the highest width and length
   * in batch, because input tensor must have consistent shape between each of the batches.
   * Then, detection of the text is performed on modified images, which produces bounding boxes.
   * These boxes are used in the next step - creation of text crops used as input for text recognition.
   * Input is combined into single batch tensor. After, the recognition is performed, which produces
   * a tensor of size equals to max_word_length * batch_size. Alphabet is used to decode words.
   *
   * @param images input images of CV_8UC(channels) type and RBG or GRAY color mode
   * @return pipeline outputs consisting of bounding box coordinates and recognized text for frame
   */
  std::vector<PipelineOutput> Predict(const std::vector<cv::Mat> &images);
};
} // namespace keras_ocr


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace keras_ocr {
std::vector<PipelineOutput> Pipeline::Predict(const std::vector<cv::Mat> &images) {
  assert(!images.empty() && "Provide at least one input image to perform prediction!");
  std::vector<cv::Mat> processed_images;

  // compute max dimensions for padding
  auto max_height = std::max_element(
      images.begin(),
      images.end(),
      [](auto &m1, auto &m2) {
        return m1.size[0] < m2.size[0];
      })[0].size[0];
  auto max_width = std::max_element(
      images.begin(),
      images.end(),
      [](auto &m1, auto &m2) {
        return m1.size[1] < m2.size[1];
      })[0].size[1];

  // preprocess images for detection
  for (auto &image : images) {
    // TODO: limit maximum size of the image to improve the performance
    cv::Mat processed = ResizeImage(image, scale_);
    processed = PadImage(processed, max_height * scale_, max_width * scale_);
    processed_images.emplace_back(processed);
  }
  auto input_tensors = detector_.PreprocessMultipleImages(processed_images);
  cppflow::tensor detection_batch = images.size() > 1
                                    ? cppflow::concat({0}, input_tensors)
                                    : input_tensors[0];

  // detection
  auto detection_inference = detector_.Inference(detection_batch);
  auto detected_boxes = detector_.PostprocessOutput(detection_inference);

  // check if any text boxes were detected
  std::vector<PipelineOutput> output;
  bool non_empty = std::any_of(detected_boxes.begin(),
                               detected_boxes.end(),
                               [](auto &boxes) { return !boxes.boxes.empty(); });
  if (non_empty) {
    // preprocess for recognition
    std::vector<cv::Mat> cropped_boxes;
    for (int i = 0; i < detected_boxes.size(); i++) {
      auto crops = recognizer_.CropToBoxes(processed_images[i], detected_boxes[i].boxes);
      cropped_boxes.insert(cropped_boxes.end(), crops.begin(), crops.end());
    }
    auto ocr_input = recognizer_.PreprocessInput(cropped_boxes);

    // recognition
    auto recognition_inference = recognizer_.Inference(ocr_input);
    std::vector<std::string> recognized_text = recognizer_.PostprocessOutput(recognition_inference);

    // postprocess detection & recognition, pairing text with frames and boxes
    int text_boxes_counter = 0;
    for (int frame_num = 0; frame_num < detected_boxes.size(); frame_num++) {
      auto frame_boxes = detected_boxes[frame_num].boxes;
      AdjustBoxes(frame_boxes, 1. / scale_);
      for (auto &[frame_box] : frame_boxes) {
        output.emplace_back(PipelineOutput{
            frame_num,
            recognized_text[text_boxes_counter++],
            frame_box
        });
      }
    }
  }
  return output;
}
} // namespace keras_ocr

#endif // KERAS_OCR_PIPELINE_H_
