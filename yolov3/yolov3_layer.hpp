#ifndef YOLOV3_LAYER_HPP
#define YOLOV3_LAYER_HPP

#include <mlpack.hpp>

namespace mlpack {

template <typename MatType = arma::fmat>
class YOLOv3Layer : public Layer<MatType>
{
 public:
  YOLOv3Layer() { /* Nothing to do. */ }

  YOLOv3Layer(size_t imgSize,
              size_t numAttributes,
              size_t gridSize,
              size_t predictionsPerCell,
              std::vector<typename MatType::elem_type> anchors);

  YOLOv3Layer* Clone() const override { return new YOLOv3Layer(*this); }

  // Copy the given YOLOv3Layer.
  YOLOv3Layer(const YOLOv3Layer& other);
  // Take ownership of the given YOLOv3Layer.
  YOLOv3Layer(YOLOv3Layer&& other);
  // Copy the given YOLOv3Layer.
  YOLOv3Layer& operator=(const YOLOv3Layer& other);
  // Take ownership of the given YOLOv3Layer.
  YOLOv3Layer& operator=(YOLOv3Layer&& other);

  void ComputeOutputDimensions() override;

  // Output format: cx, cy, w, h
  void Forward(const MatType& input, MatType& output) override;

  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g) override;

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:

  using Type = typename MatType::elem_type;

  using CubeType = typename GetCubeType<MatType>::type;

  size_t imgSize;

  size_t numAttributes;

  size_t gridSize;
  // Cached gridSize * gridSize
  size_t grid;

  MatType w;

  MatType h;

  size_t predictionsPerCell;
};


} // namespace mlpack

#include "yolov3_layer_impl.hpp"

#endif
