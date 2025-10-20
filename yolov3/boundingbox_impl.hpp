#ifndef BOUNDINGBOX_IMPL_HPP
#define BOUNDINGBOX_IMPL_HPP

#include "boundingbox.hpp"

inline void DrawLetter(Image& image, const unsigned char letter, const size_t x, const size_t y, const size_t s)
{
  for (char i = 0; i < 8; i++)
  {
    for (char j = 0; j < 8; j++)
    {
      int set = font8x8_basic[letter][i] & (unsigned char)(1 << j);
      for (int k = 0; k < s * s; k++)
      {
        int px = x + (j * s) + (k % s);
        int py = y + (i * s) + (k / s);
        for (int c = 0; c < 3; c++)
          image.SetPixel(px, py, c, !set);
      }
    }
  }
}

inline double Intersection(const BoundingBox& a, const BoundingBox& b)
{
  const double w = std::max((std::min(a.x2, b.x2) - std::max(a.x1, b.x1)), 0.0);
  const double h = std::max((std::min(a.y2, b.y2) - std::max(a.y1, b.y1)), 0.0);
  return w * h;
}

inline double Union(const BoundingBox& a, const BoundingBox& b)
{
  const double aw = a.x2 - a.x1;
  const double ah = a.y2 - a.y1;
  const double bw = b.x2 - b.x1;
  const double bh = b.y2 - b.y1;
  return (aw * ah) + (bw * bh) - Intersection(a, b);
}

inline void NonMaxSuppression(std::vector<BoundingBox>& bboxes,
                              const double threshold)
{
  if (bboxes.size() == 0)
    return;

  std::sort(bboxes.begin(), bboxes.end(), std::greater<BoundingBox>());
  for (size_t i = 0; i < bboxes.size() - 1; i++)
  {
    for (size_t j = i + 1; j < bboxes.size(); j++)
    {
      const BoundingBox& a = bboxes[i];
      BoundingBox& b = bboxes[j];

      if (a.objectClass != b.objectClass)
        continue;

      const double iou = Intersection(a, b) / Union(a, b);
      if (iou > threshold)
        b.objectProb = 0;
    }
  }
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
  std::vector<BoundingBox> bboxes;

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
    bboxes.push_back(bbox);
  }

  NonMaxSuppression(bboxes);
  for (auto& bbox : bboxes)
    bbox.Draw(image, borderSize, labels, alphabet, letterSize);
}

#endif
