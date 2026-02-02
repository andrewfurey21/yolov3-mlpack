#include "./yolov3/serialization.hpp"
#include "./yolov3/image.hpp"
#include "./yolov3/boundingbox.hpp"
#include "./yolov3/labels.hpp"

#include <mlpack.hpp>
#include <mlpack/core/data/image_layout.hpp>
#include <mlpack/methods/ann/layer/grouped_convolution.hpp>

int main() {
  const std::string inputFile = "./images/dog.jpg";
  const std::string outputFile = "./images/output.jpg";
  const size_t imgSize = 320;
  const size_t numBoxes = 6300; // 2535, 6300, 10647, 22743
  const double ignoreThresh = 0.7;
  const std::vector<float> anchors = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};

  // 80 classes, 3 predictions per cell.
  mlpack::YOLOv3<mlpack::EmptyLoss, mlpack::RandomInitialization, arma::fmat> model;
  mlpack::Load("./weights/mlpack/yolov3-320.bin", "yolov3-320", model);

  Image image;
  Image input(imgSize, imgSize, 3);
  LoadImage(inputFile, image);

  Image groupedImage = image;
  groupedImage.data = mlpack::GroupChannels(groupedImage.data, groupedImage.info);
  LetterBox(groupedImage, input);

  arma::fmat detections;
  model.Predict(input.data, detections);

  auto alphabet = GetAlphabet("./data/labels");
  auto labels = GetLabels("./data/coco.names", 80); // 80 for coco

  DrawBoxes(detections, numBoxes, ignoreThresh, 4, imgSize, 1.5, labels, alphabet, image);
  SaveImage(outputFile, image);
  return 0;
}
