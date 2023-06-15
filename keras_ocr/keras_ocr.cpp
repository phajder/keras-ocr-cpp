//
// Created by phajder on 05/11/2022.
//

#include "keras_ocr.h"

int main() {
  using namespace keras_ocr;
  using std::string, std::vector;
  using cv::Mat, cv::COLOR_BGR2RGB, cv::imread;
  vector<string> image_paths = {
      "../examples/img1.jpg",
      "../examples/img2.jpg"
  };
  vector<Mat> images;
  for (auto &path : image_paths) {
    Mat image = imread(path);
    cvtColor(image, image, COLOR_BGR2RGB);
    images.emplace_back(image);
  }

  Pipeline p;
  vector<PipelineOutput> predictions = p.Predict(images);

  PrintPipelineOutput(predictions);
  DrawOutput(images, predictions);
}
