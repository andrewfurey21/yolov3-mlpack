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

struct BoundingBox {
  arma::fcolvec pos;
  arma::fcolvec color;
  std::string objectClass;
  float objectProb;
};

double Intersection(const BoundingBox& a, const BoundingBox& b)
{
  const double w = std::max((std::min(a.pos(2), b.pos(2)) - std::max(a.pos(0), b.pos(0))), 0.0f);
  const double h = std::max((std::min(a.pos(3), b.pos(3)) - std::max(a.pos(1), b.pos(1))), 0.0f);
  return w * h;
}

double Union(const BoundingBox& a, const BoundingBox& b)
{
  const double aw = a.pos(2) - a.pos(0);
  const double ah = a.pos(3) - a.pos(1);
  const double bw = b.pos(2) - b.pos(0);
  const double bh = b.pos(3) - b.pos(1);
  return (aw * ah) + (bw * bh) - Intersection(a, b);
}

void NonMaxSuppression(std::vector<BoundingBox>& bboxes, const double threshold)
{
  if (bboxes.size() == 0)
    return;

  std::sort(bboxes.begin(), bboxes.end(),
      [](const BoundingBox& a, const BoundingBox& b) {
        return a.objectProb > b.objectProb;
      });
  for (size_t i = 0; i < bboxes.size() - 1; i++)
  {
    for (size_t j = i + 1; j < bboxes.size(); j++)
    {
      const BoundingBox& a = bboxes[i];
      BoundingBox& b = bboxes[j];

      if (a.objectClass != b.objectClass)
        continue;

      const double iou = Intersection(a, b) / Union(a, b);
      if (iou > threshold) {
        b.objectProb = 0;
      }
    }
  }
}

std::vector<std::string> GetLabels(const std::string& path, size_t numClasses)
{
  std::ifstream file(path);
  std::vector<std::string> labels;
  if (!file)
  {
    std::ostringstream errMessage;
    errMessage << "Could not open " << path << ".";
    throw std::logic_error(errMessage.str());
  }

  std::string line;
  while (std::getline(file, line))
    labels.push_back(line);

  if (labels.size() != numClasses)
  {
    std::ostringstream errMessage;
    errMessage << "Expected " << numClasses
               << " classes, but got " << labels.size() << ".";
    throw std::logic_error(errMessage.str());
  }
  return labels;
}

int main(int argc, const char** argv) {
  if (argc != 5)
    throw std::logic_error("usage: ./yolov3 <weights_file> <coco_names> <input_image> <output_image>");

  const std::string inputFile = argv[3];
  const std::string outputFile = argv[4];
  const std::string modelFile = argv[1];
  const std::string labelsFile = argv[2];
  const size_t imgSize = 320;
  const size_t numBoxes = 6300; // 2535, 6300, 10647, 22743
  const double ignoreThresh = 0.7;
  const std::vector<float> anchors = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
  auto labels = GetLabels(labelsFile, 80);

  // 80 classes, 3 predictions per cell.
  mlpack::YOLOv3<mlpack::EmptyLoss, mlpack::RandomInitialization, arma::fmat> model;
  bool modelSuccess = mlpack::Load(modelFile, model);
  if (!modelSuccess) {
    std::cout << "Error: could not load " + modelFile;
    return -1;
  }

  arma::fmat image;
  mlpack::ImageInfo info;
  bool imageSuccess = mlpack::Load(inputFile, image, info, true);
  if (!imageSuccess) {
    std::cout << "Error: could not load " + inputFile;
    return -1;
  }


  arma::fmat preprocessedImage = image / 255.0f;
  mlpack::ImageInfo preprocessedInfo = info;
  mlpack::LetterboxImages(preprocessedImage, preprocessedInfo, imgSize, imgSize, 0.5f);
  preprocessedImage = mlpack::GroupChannels(preprocessedImage, preprocessedInfo);

  arma::fmat detections;
  model.Predict(preprocessedImage, detections);

  const size_t width = info.Width();
  const size_t height = info.Height();

  double xRatio = (double)width / imgSize;
  double yRatio = (double)height / imgSize;

  double xOffset = 0;
  double yOffset = 0;

  if (width > height) {
    // landscape
    yRatio =  (double)width / imgSize;
    yOffset = (imgSize - (height * imgSize / (double)width)) / 2;
  } else {
    // portrait
    xRatio =  (double)height / imgSize;
    xOffset = (imgSize - (width * imgSize / (double)height)) / 2;
  }

  const size_t predictionSize = detections.n_rows / numBoxes;
  std::vector<BoundingBox> bboxes;
  for (size_t box = 0; box < numBoxes; box++)
  {
    arma::fmat prediction;
    mlpack::MakeAlias(prediction, detections, predictionSize, 1,
      box * predictionSize);
    float objectness = prediction.at(4, 0);
    if (objectness < ignoreThresh)
      continue;
    float x1, y1, x2, y2;
    x1 = (prediction.at(0, 0) - prediction.at(2, 0) / 2 - xOffset) * xRatio;
    y1 = (prediction.at(1, 0) - prediction.at(3, 0) / 2 - yOffset) * yRatio;
    x2 = (prediction.at(0, 0) + prediction.at(2, 0) / 2 - xOffset) * xRatio;
    y2 = (prediction.at(1, 0) + prediction.at(3, 0) / 2 - yOffset) * yRatio;
    const arma::fmat& classProbs =
      prediction.submat(5, 0, prediction.n_rows - 1, 0);
    const size_t classIndex = classProbs.index_max();
    const float objectProb = objectness * classProbs.at(classIndex);

    if (objectProb < ignoreThresh)
      continue;

    arma::fcolvec pos = {x1, y1, x2, y2};
    arma::fcolvec red = {255.0f, 0, 0};
    BoundingBox bbox = {
      .pos = pos,
      .color = red,
      .objectClass = labels[classIndex],
      .objectProb = objectProb
    };
    bboxes.push_back(bbox);
  }

  NonMaxSuppression(bboxes, 0.5);
  for (auto& bbox : bboxes) {
    if (bbox.objectProb < ignoreThresh) continue;
    mlpack::BoundingBoxImage(image, info, bbox.pos, bbox.color, 1, bbox.objectClass, 2);
    std::cout << bbox.objectClass << ": " << roundf(bbox.objectProb * 100) << "%\n";
  }

  bool saveSuccess = mlpack::Save(outputFile, image, info, true);
  if (!saveSuccess) {
    std::cout << "Error: could not save " + outputFile << "\n";
    return -1;
  }

  return 0;
}
