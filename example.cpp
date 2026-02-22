#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#include <mlpack.hpp>

// Used for deserializing the model and loading the weights
CEREAL_REGISTER_TYPE(mlpack::Layer<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::Identity<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::MultiLayer<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::Convolution<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::BatchNorm<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::LeakyReLU<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::Padding<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::MaxPooling<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::NearestInterpolation<arma::fmat>)
CEREAL_REGISTER_TYPE(mlpack::YOLOv3Layer<arma::fmat>)

int main(int argc, const char** argv) {
  if (argc != 4) {
    std::cout << "usage: ./yolov3 <weights_file> <input_image> <output_image>\n";
    return -1;
  }

  const std::string inputFile = argv[2];
  const std::string outputFile = argv[3];
  const std::string modelFile = argv[1];

  const double ignoreThreshold = 0.7; // 0.5 for yolov3-tiny

  // Load model
  mlpack::YOLOv3 model;
  bool modelSuccess = mlpack::Load(modelFile, model);
  if (!modelSuccess) {
    std::cout << "Error: could not load " + modelFile;
    return -1;
  }

  // Load image
  arma::fmat image;
  mlpack::ImageInfo info;
  bool imageSuccess = mlpack::Load(inputFile, image, info, true);
  if (!imageSuccess) {
    std::cout << "Error: could not load " + inputFile;
    return -1;
  }

  // Inference
  model.Predict(image, info, ignoreThreshold);

  // Save image with bounding boxes.
  bool saveSuccess = mlpack::Save(outputFile, image, info, true);
  if (!saveSuccess) {
    std::cout << "Error: could not save " + outputFile << "\n";
    return -1;
  }

  return 0;
}
