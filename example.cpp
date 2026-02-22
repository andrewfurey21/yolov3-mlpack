#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#include <mlpack.hpp>

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
  if (argc != 5) {
    std::cout << "usage: ./yolov3 <weights_file> <input_image> <output_image>\n";
    return -1;
  }

  const std::string inputFile = argv[2];
  const std::string outputFile = argv[3];
  const std::string modelFile = argv[1];

  mlpack::YOLOv3 model;
  bool modelSuccess = mlpack::Load(modelFile, model);
  if (!modelSuccess) {
    std::cout << "Error: could not load " + modelFile;
    return -1;
  }

  arma::fmat image, outputImage;
  mlpack::ImageInfo info;
  bool imageSuccess = mlpack::Load(inputFile, image, info, true);
  if (!imageSuccess) {
    std::cout << "Error: could not load " + inputFile;
    return -1;
  }

  model.Predict(image, info, outputImage);

  bool saveSuccess = mlpack::Save(outputFile, outputImage, info, true);
  if (!saveSuccess) {
    std::cout << "Error: could not save " + outputFile << "\n";
    return -1;
  }

  return 0;
}
