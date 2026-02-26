#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

int main(int argc, const char **argv) {
  // Step 1: load the pretrained arma::mat weights.
  // Download: https://models.mlpack.org/yolo/yolov3-320-coco-f64.bin
  mlpack::YOLOv3<arma::mat> model;
  mlpack::Load("yolov3-320-coco-f64.bin", model);

  // Step 2: load the image into an arma::fmat.
  // Download: https://models.mlpack.org/yolo/dog.jpg
  // Note: the image type must also be `arma::mat`
  arma::mat inputImage, outputImage;
  mlpack::ImageOptions opts;
  mlpack::Load("./cat.jpg", inputImage, opts);

  // Step 3: Preprocess the `inputImage`, detect bounding boxes and draw them onto `outputImage`.
  model.Predict(inputImage, opts, outputImage, true);

  // Step 4: Save to "output.jpg"
  mlpack::Save("cat_output.jpg", outputImage, opts, true);
}
