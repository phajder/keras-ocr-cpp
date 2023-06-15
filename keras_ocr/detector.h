//
// Created by phajder on 05/11/2022.
//

#ifndef KERAS_OCR_DETECTOR_H_
#define KERAS_OCR_DETECTOR_H_

#include "cppflow/tensor.h"
#include "cppflow/model.h"
#include "opencv2/imgproc.hpp"

namespace keras_ocr {
/**
 * @class Detector
 * @brief A cppflow::model wrapper for serving Text Detection model based on Character-Region Awareness For Text detection (CRAFT).
 */
class Detector {
 private:
  /**
   * Detection model created using cppflow wrapper
   */
  cppflow::model model_;

  // Thresholds used for connected components computations
  float detection_threshold_ = 0.7f;
  float text_threshold_ = 0.4f;
  float link_threshold_ = 0.4f;
  int size_threshold_ = 10;
  // =====================================================

  /**
   * Default name for input op.
   */
  const std::string default_input_op_ = "serving_default_input_1";
 public:
  /**
   * Creates detector using path from compiler definitions
   */
  Detector() : model_(DETECTION_MODEL_PATH) {}

  /**
   * Creates recognizer by initializing model using custom path to model.
   * Model must be in format supported by cppflow: Frozen_graph of Saved_model.
   *
   * @param modelPath path to detector model
   */
  explicit Detector(const std::string &model_path) : model_(model_path) {}

  /**
   * Preprocesses image and transforms it to four-dimensional input tensor.
   * Image must be continuous, i.e. cannot be a cv::Mat slice, otherwise
   * memory access violation will occur during tensor creation.
   *
   * @param image image represented in CV_8UC(channels) type and RBG or GRAY color mode
   * @return input tensor used for inference
   */
  cppflow::tensor PreprocessImage(const cv::Mat &image) const;

  /**
   * Preprocesses multiple images and transforms them to four-dimensional input tensor.
   * Images must be continuous, i.e. cannot be a cv::Mat slices, otherwise memory access
   * violation will occur during tensor creation.
   *
   * @param images vector of images represented in CV_8UC(channels) type and RGB or GRAY color mode
   * @return vector of input tensors used for inference
   */
  std::vector<cppflow::tensor> PreprocessMultipleImages(const std::vector<cv::Mat> &images) const;

  /**
   * Performs input inference.
   *
   * @param input batch four-dimensional tensor, consisting of at least one image
   * @return model output as four-dimensional tensor
   */
  cppflow::tensor Inference(cppflow::tensor input);

  /**
   * Computes bounding boxes coordinates using cv connected components.
   * Converts output tensor to multiple text and link maps, which are used
   * to create connected components with stats using cv::connectedComponentsWithStats
   * method. Box is returned if stats meet thresholds specified in class fields.
   *
   * @param inference four-dimensional output tensor from inference
   * @return vector of detection outputs, each with corresponding bounding boxes
   */
  std::vector<DetectionOutput> PostprocessOutput(const cppflow::tensor &inference) const;
};
} // namespace keras_ocr


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace keras_ocr {
cppflow::tensor Detector::PreprocessImage(const cv::Mat &image) const {
  assert(image.isContinuous() && "Input image must be continuous to create a tensor - Mat slices are not allowed!");
  int model_channels = static_cast<int>(model_.get_operation_shape(default_input_op_)[3]);
  cv::Mat source_image = image.clone();

  auto mean = cv::Scalar(0.485, 0.456, 0.406),
      variance = cv::Scalar(0.229, 0.224, 0.225);
  if (model_channels == 1 && image.channels() == 3) cv::cvtColor(source_image, source_image, cv::COLOR_RGB2GRAY);
  source_image.convertTo(source_image, CV_32FC(model_channels), 1. / 255.);
  source_image -= mean;
  source_image /= variance;

  auto array = Mat2Vector<float>(source_image);
  auto input = cppflow::tensor(array, {source_image.size[0], source_image.size[1], source_image.channels()});
  input = expand_dims(input, 0);

  return input;
}

std::vector<cppflow::tensor> Detector::PreprocessMultipleImages(const std::vector<cv::Mat> &images) const {
  std::vector<cppflow::tensor> input_tensors;
  for (auto &image : images) {
    cppflow::tensor tensor = PreprocessImage(image);
    input_tensors.emplace_back(tensor);
  }

  return input_tensors;
}

cppflow::tensor Detector::Inference(cppflow::tensor input) {
  auto output = model_(
      {{default_input_op_ + ":0", input}},
      {"StatefulPartitionedCall:0"}
  )[0];

  return output;
}

std::vector<DetectionOutput> Detector::PostprocessOutput(const cppflow::tensor &inference) const {
  auto batches = SplitTensor(inference);
  auto output_shape = batches[0].shape().get_data<int64_t>();
  int img_h = static_cast<int>(output_shape[0]),
      img_w = static_cast<int>(output_shape[1]);
  int shape[2] = {img_h, img_w};

  std::vector<DetectionOutput> detections;
  detections.reserve(batches.size());
  for (const auto &kBatch : batches) {
    auto map_tensors = SplitTensor(kBatch, 2);
    auto v_textmap = map_tensors[0].get_data<float>(),
        v_linkmap = map_tensors[1].get_data<float>();

    cv::Mat textmap(2, shape, CV_32FC1, v_textmap.data()),
        linkmap(2, shape, CV_32FC1, v_linkmap.data()),
        text_score, link_score;

    cv::threshold(textmap, text_score, text_threshold_, 1, cv::THRESH_BINARY);
    cv::threshold(linkmap, link_score, link_threshold_, 1, cv::THRESH_BINARY);

    cv::Mat tlss(text_score + link_score), labels, stats, centroids;
    // clip values outside [0;1] range
    cv::threshold(tlss, tlss, 1., 1., cv::THRESH_TRUNC);
    cv::threshold(tlss, tlss, 0., 1., cv::THRESH_TOZERO);

    // compute connected components of boolean image, input must be 8-bit single-channel
    tlss.convertTo(tlss, CV_8UC1);
    int num_components = connectedComponentsWithStats(tlss, labels, stats, centroids, 4);
    centroids.release(); // centroids are not required
    tlss.release(); // tlss is not required anymore

    std::vector<Box> boxes;
    for (int component_id = 1; component_id < num_components; component_id++) {
      // Skip component if its component_size is smaller than threshold
      int component_size = stats.at<int>(component_id, cv::CC_STAT_AREA);
      if (component_size < size_threshold_) continue;

      // If the maximum value within this connected component
      // is less than text threshold, skip it.
      cv::Mat textmap_mask(labels == component_id);
      double max;
      cv::minMaxLoc(textmap, nullptr, &max, nullptr, nullptr, textmap_mask);
      if (max < detection_threshold_) continue;

      // Make segmentation map: 1 - text, 0 otherwise
      cv::Mat segmap;
      textmap_mask.convertTo(segmap, CV_32FC1, 1. / 255.);
      segmap &= ~(link_score & text_score);

      // Expand segmentation map elements
      int w = stats.at<int>(component_id, cv::CC_STAT_WIDTH);
      int h = stats.at<int>(component_id, cv::CC_STAT_HEIGHT);
      int niter = static_cast<int>(std::sqrt(component_size * std::min(w, h) / (w * h)) * 2);
      cv::Mat se = getStructuringElement(cv::MORPH_RECT, {1 + niter, 1 + niter});
      cv::dilate(segmap, segmap, se);

      // Make rotated box from contour
      segmap.convertTo(segmap, CV_8UC1);
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(segmap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
      std::vector<cv::Point2f> box(4);
      cv::minAreaRect(contours[0]).points(box.data());

      // Check if box is diamond-shaped
      double norm_w = norm(box[0] - box[1]);
      double norm_h = norm(box[1] - box[2]);
      double box_ratio = std::max(norm_w, norm_h) / (std::min(norm_w, norm_h) + 1e-5);
      if (abs(1. - box_ratio) <= 0.1) {
        auto [left, right] = std::minmax_element(
            contours[0].begin(),
            contours[0].end(),
            [](auto const &u, auto const &v) {
              return u.x < v.x;
            });
        auto [top, bottom] = std::minmax_element(
            contours[0].begin(),
            contours[0].end(),
            [](auto const &u, auto const &v) {
              return u.y < v.y;
            });
        box[0] = cv::Point2f(static_cast<float>(left->x),
                             static_cast<float>(top->y));
        box[1] = cv::Point2f(static_cast<float>(right->x),
                             static_cast<float>(top->y));
        box[2] = cv::Point2f(static_cast<float>(right->x),
                             static_cast<float>(bottom->y));
        box[3] = cv::Point2f(static_cast<float>(left->x),
                             static_cast<float>(bottom->y));
      } else {
        // Find top left coord
        auto elem = std::min_element(
            box.begin(),
            box.end(),
            [](auto const &a, auto const &b) {
              return a.x + a.y < b.x + b.y;
            });
        // Roll box coords to match their proper ordering
        int d = static_cast<int>(std::distance(box.begin(), elem));
        std::rotate(box.begin(), box.begin() + d, box.end());
      }
      // Multiply coords by 2 - heatmap dims are twice smaller than base image
      std::transform(box.begin(),
                     box.end(),
                     box.begin(),
                     [](auto const &p) { return p * 2; });
      boxes.emplace_back(Box{box});
    }
    detections.emplace_back(DetectionOutput{boxes});
  }

  return detections;
}
} // namespace keras_ocr

#endif // KERAS_OCR_DETECTOR_H_
