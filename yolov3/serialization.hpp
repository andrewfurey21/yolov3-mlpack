#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING

#include <mlpack.hpp>
// #include "yolov3_layer.hpp"

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
