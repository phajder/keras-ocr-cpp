//
// Created by phajder on 10/11/2022.
//

#ifndef KERAS_OCR_UTILS_H_
#define KERAS_OCR_UTILS_H_

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "cppflow/tensor.h"
#include "cppflow/raw_ops.h"

namespace keras_ocr {
/**
 * Resizes image by specified scale.
 *
 * @param image image to be resized
 * @param scale scaling parameter
 * @return
 */
cv::Mat ResizeImage(const cv::Mat &image, double scale) {
  if (scale == 1.) return image;
  cv::Mat resized;
  cv::resize(image, resized, {0, 0}, scale, scale);

  return resized;
}

/**
 * Extends image to specified size and pads it with padding value.
 *
 * @param image image to be padded
 * @param height output height of the image
 * @param width output width of the image
 * @param padding_val padding value for additional pixels
 * @return extended and padded image
 */
cv::Mat PadImage(const cv::Mat &image, int height, int width, int padding_val = 0) {
  cv::Mat padded;
  assert(height >= image.size[0] && "Input height must be smaller than output height!");
  assert(width >= image.size[1] && "Input width must be smaller than output width!");
  int height_padding = height - image.size[0], width_padding = width - image.size[1];
  cv::copyMakeBorder(
      image,
      padded,
      0,
      height_padding,
      0,
      width_padding,
      cv::BORDER_CONSTANT,
      padding_val
  );

  return padded;
}

/**
 * Adjusts boxes from detector to original scale.
 *
 * @param boxes bounding boxes to be adjusted
 * @param scale inverted scale factor
 */
void AdjustBoxes(std::vector<Box> &boxes, double scale) {
  for (auto &kBox : boxes) {
    std::transform(
        kBox.coords.begin(),
        kBox.coords.end(),
        kBox.coords.begin(),
        [=](auto &p) {
          return p * scale;
        });
  }
}

/**
 * Splits single batch tensor to multiple tensors, usually squeezed by batch dimension.
 *
 * @param batch_tensor tensor to be split
 * @param split_dim dimension by which tensor is to be split
 * @param squeezed if output tensor should be squeezed by selected dimension
 * @return vector with multiple batch tensors of batch_size = 1
 */
std::vector<cppflow::tensor> SplitTensor(const cppflow::tensor &batch_tensor, int split_dim = 0, bool squeezed = true) {
  std::vector<int64_t> shape = batch_tensor.shape().get_data<int64_t>();
  int tensor_dims = static_cast<int>(shape.size());
  std::vector<int32_t> v_shape(shape.begin(), shape.end()), v_start(shape.size(), 0);
  v_shape[split_dim] = 1;
  cppflow::tensor size(v_shape, {tensor_dims});
  std::vector<cppflow::tensor> tensors;
  for (int i = 0; i < shape[split_dim]; i++) {
    v_start[split_dim] = i;
    cppflow::tensor start(v_start, {tensor_dims}),
        sliced = cppflow::slice(batch_tensor, start, size, TF_FLOAT);
    if (squeezed) sliced = cppflow::squeeze(sliced, {split_dim});
    tensors.emplace_back(sliced);
  }

  return tensors;
}

/**
 * Prints single box to standard output.
 *
 * @param box box coordinates
 * @param prefix prefix value to standard output
 */
void PrintBox(const Box &box, const std::string &prefix = "\t\t") {
  for (int i = 0; i < box.coords.size(); i++) {
    std::cout << prefix << "P" << i << ": " << box.coords[i] << "," << std::endl;
  }
}

/**
 * Prints pipeline output boxes and text to standard output.
 *
 * @param predictions predicted values
 */
void PrintPipelineOutput(const std::vector<PipelineOutput> &predictions) {
  std::cout << "[" << std::endl;
  for (auto &[frame, text, box] : predictions) {
    std::cout << "{" << std::endl;
    std::cout << "\tframe: " << frame << "," << std::endl;
    std::cout << "\tboxes: [" << std::endl;
    PrintBox(box);
    std::cout << "\t]," << std::endl;
    std::cout << "\ttext: " << "\"" << text << "\"" << std::endl;
    std::cout << "}," << std::endl;
  }
  std::cout << "]" << std::endl;
}

/**
 * Draws bounding boxes and predicted text in cv Windows using cv::highui.
 *
 * @param images images to be drawn
 * @param predictions predicted values
 */
void DrawOutput(const std::vector<cv::Mat> &images, const std::vector<PipelineOutput> &predictions) {
  for (int i = 0; i < images.size(); i++) {
    cv::Mat modified;
    cv::cvtColor(images[i], modified, cv::COLOR_RGB2BGR);
    for (auto &[frame, text, box] : predictions) {
      if (frame == i) {
        cv::polylines(
            modified,
            std::vector<cv::Point>(
                box.coords.begin(),
                box.coords.end()
            ), true,
            {0, 0, 255}
        );
        cv::putText(
            modified,
            text,
            box.coords[0],
            cv::FONT_HERSHEY_COMPLEX_SMALL,
            1,
            {0, 0, 255}
        );
      }
    }
    cv::imshow("Detected boxes", modified);
    cv::waitKey(0);
  }
  cv::destroyAllWindows();
}

/**
 * Transforms cv::Mat data into vector with data of type T
 * It is assumed that image is continuous, i.e., obtained by cv::imread or cv::Mat::clone.
 *
 * @tparam T datatype stored in mat as uchar (usually float)
 * @param mat source image to be transformed into vector
 * @return mat data as vector
 */
template<typename T>
std::vector<T> Mat2Vector(const cv::Mat &mat) {
  auto source_image_data = reinterpret_cast<T *>(mat.data);
  std::vector<float> array;
  assert(mat.isContinuous() && "Mat must be continuous to use assign method! Mats with borrowed content not allowed!");
  array.assign(source_image_data, source_image_data + mat.total() * mat.channels());

  return array;
}
} // namespace keras_ocr

#endif // KERAS_OCR_UTILS_H_
