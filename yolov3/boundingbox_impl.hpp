#ifndef BOUNDINGBOX_IMPL_HPP
#define BOUNDINGBOX_IMPL_HPP

#include "boundingbox.hpp"

inline double LineOverlap(double a, double aw, double b, double bw) {
  return std::abs(std::max(a - aw / 2, b - bw / 2) -
                  std::min(a + aw / 2, b + bw / 2));
}

inline void DrawBoxes(const arma::fmat& modelOutput,
               const size_t numBoxes,
               const double ignoreProb,
               const double borderSize,
               const size_t imgSize,
               const double letterSize,
               const std::vector<std::string>& labels,
               const std::unordered_map<char, Image>& alphabet,
               Image& image)
{
  CheckImage(image);
  if (modelOutput.n_cols != 1)
  {
    std::ostringstream errMessage;
    errMessage << "modelOutput should have 1 column, but you gave "
               << modelOutput.n_cols << " columns.";
    throw std::logic_error(errMessage.str());
  }

  const size_t rem = modelOutput.n_rows % numBoxes;
  if (rem != 0)
  {
    std::ostringstream errMessage;
    errMessage << "modelOutput rows should be divisible by numBoxes, but "
               << modelOutput.n_rows << " % " << numBoxes << " == " << rem;
    throw std::logic_error(errMessage.str());
  }

  double xRatio = (double)image.info.Width() / imgSize;
  double yRatio = (double)image.info.Height() / imgSize;

  const size_t predictionSize = modelOutput.n_rows / numBoxes;
  for (size_t box = 0; box < numBoxes; box++)
  {
    arma::fmat prediction;
    mlpack::MakeAlias(prediction, modelOutput, predictionSize, 1,
      box * predictionSize);
    float objectness = prediction.at(4, 0);
    if (objectness < ignoreProb)
      continue;
    double x, y, w, h;
    x = prediction.at(0, 0) * xRatio;
    y = prediction.at(1, 0) * yRatio;
    w = prediction.at(2, 0) * xRatio;
    h = prediction.at(3, 0) * yRatio;
    const arma::fmat& classProbs =
      prediction.submat(5, 0, prediction.n_rows - 1, 0);
    const size_t classIndex = classProbs.index_max();
    const float objectProb = objectness * classProbs.at(classIndex);

    if (objectProb < ignoreProb)
      continue;
    std::cout << labels[classIndex] << ": " << roundf(objectProb * 100) << "%\n";
    BoundingBox bbox(x, y, w, h, classIndex, objectProb, labels.size());
    bbox.Draw(image, borderSize, labels, alphabet, letterSize);
  }
}

#endif
