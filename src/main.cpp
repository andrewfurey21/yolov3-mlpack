#include <algorithm>
#include <mlpack.hpp>
#include "../models/models/yolov3_tiny/yolov3_tiny.hpp"
#include <armadillo>
#include <mlpack/core/data/image_info.hpp>
#include <mlpack/methods/ann/layer/nearest_interpolation.hpp>

using namespace mlpack::data;


void load(const std::string& file, arma::Mat<double>& data, ImageInfo& info) {
  Load(file, data, info, true);
  data /= 255.0f;
}

void save(const std::string& file, arma::Mat<double> data, ImageInfo& info) {
  data *= 255;
  Save(file, data, info, true);
}

void fill(arma::Mat<double>& data, double value) {
  value = std::clamp<double>(value, 0.0f, 1.0f);
  data.fill(value);
}

arma::Mat<double> resize(const arma::Mat<double>& data, 
                         ImageInfo& info, 
                         ImageInfo& outputInfo) {

  size_t newWidth = outputInfo.Width();
  size_t newHeight = outputInfo.Height();

  assert(data.n_rows == info.Width() * info.Height() * 3 && data.n_cols == 1);
  assert(newWidth > 1);
  assert(newHeight > 1);
  arma::Mat<double> resized(newWidth * newHeight * 3, 1);
  outputInfo = ImageInfo(newWidth, newHeight, 3);

  double xRatio = (double)(info.Width() - 1) / (newWidth - 1);
  double yRatio = (double)(info.Height() - 1) / (newHeight - 1);

  for (size_t channel = 0; channel < info.Channels(); channel++) {
    for (size_t i = 0; i < newWidth; i++) {
      for (size_t j = 0; j < newHeight; j++) {
        size_t xLow = std::floor(xRatio * i);
        size_t yLow = std::floor(yRatio * j);

        size_t xHigh = std::ceil(xRatio * i);
        size_t yHigh = std::ceil(yRatio * j);

        size_t xWeight = (xRatio * i) - xLow;
        size_t yWeight = (yRatio * j) - yLow;

        double a = data.at(yLow * info.Width() * info.Channels() + xLow * info.Channels() + channel);
        double b = data.at(yLow * info.Width() * info.Channels() + xHigh * info.Channels() + channel);
        double c = data.at(yHigh * info.Width() * info.Channels() + xLow * info.Channels() + channel);
        double d = data.at(yHigh * info.Width() * info.Channels() + xHigh * info.Channels() + channel);

        double value = 
                a * (1 - xWeight) * (1 - yWeight) +
                b * xWeight * (1 - yWeight) +
                c * yWeight * (1 - xWeight) +
                d * xWeight * yWeight;

        resized.at(j * newWidth * info.Channels() + i * info.Channels() + channel) = value;
      }
    }
  }
  return resized;
}

void embed(arma::Mat<double>& source, 
           ImageInfo& sourceInfo, 
           arma::Mat<double>& dest,
           ImageInfo& destInfo,
           size_t dx,
           size_t dy) {

  assert(sourceInfo.Channels() == destInfo.Channels());
  size_t width = std::min(sourceInfo.Width() + dx, destInfo.Width());
  size_t height = std::min(sourceInfo.Height() + dy, destInfo.Height());

  for (size_t c = 0; c < sourceInfo.Channels(); c++) {
    for (size_t i = 0; i < sourceInfo.Width(); i++) {
      if (dx + i > destInfo.Width()) break;
      for (size_t j = 0; j < sourceInfo.Height(); j++) {
        if (dy + j > destInfo.Height()) break;
        size_t sourceIndex = j*sourceInfo.Channels()*sourceInfo.Width() + i * sourceInfo.Channels() + c;
        size_t destIndex = (j + dy)*destInfo.Channels()*destInfo.Width() + (i + dx) * destInfo.Channels() + c;
        dest.at(destIndex) = source.at(sourceIndex);
      }
    }
  }
}

void letterbox(arma::Mat<double>&source,
               ImageInfo sourceInfo,
               arma::Mat<double>& dest,
               ImageInfo& destInfo) {
  size_t width, height;
  if (destInfo.Width() / sourceInfo.Width() > destInfo.Height() / sourceInfo.Height()) {
    height = destInfo.Height();
    width = sourceInfo.Width() * destInfo.Height() / sourceInfo.Height();
  } else {
    width = destInfo.Width();
    height = sourceInfo.Height() * destInfo.Width() / sourceInfo.Width();
  }

  ImageInfo resizedInfo(width, height, 3);
  arma::Mat<double> resized = resize(source, sourceInfo, resizedInfo);

  dest = arma::Mat<double>(destInfo.Width() * destInfo.Height() * destInfo.Channels(), 1);
  fill(dest, .3);
  embed(resized, resizedInfo, dest, destInfo, (destInfo.Width() - width)/2, (destInfo.Height() - height)/2);
}

void tile(arma::Mat<double>& a,
          ImageInfo aInfo,
          arma::Mat<double>& b,
          ImageInfo& bInfo,
          arma::Mat<double>& output,
          ImageInfo& outputInfo,
          size_t dx) {
  assert(aInfo.Channels() == bInfo.Channels());
  size_t height = std::max(aInfo.Height(), bInfo.Height());
  size_t width = dx + aInfo.Width() + bInfo.Width();
  output = arma::Mat<double>(width * height * aInfo.Channels(), 1);
  outputInfo = ImageInfo(width, height, aInfo.Channels());
  fill(output, 1.0f);
  embed(a, aInfo, output, outputInfo, 0, 0);
  embed(b, bInfo, output, outputInfo, aInfo.Width()+dx, 0);
}

void border(arma::Mat<double>& source,
            ImageInfo sourceInfo,
            arma::Mat<double>& dest,
            ImageInfo& destInfo,
            size_t borderSize) {
  destInfo = ImageInfo(sourceInfo.Width() + 2 * borderSize, sourceInfo.Height() + 2 * borderSize, sourceInfo.Channels());
  dest = arma::Mat<double>(destInfo.Width() * destInfo.Height() * destInfo.Channels(), 1);
  fill(dest, 1.0f);
  embed(source, sourceInfo, dest, destInfo, borderSize, borderSize);
}

double lineOverlap(double a, double aw, double b, double bw) {
  return std::abs(std::max(a - aw/2, b - bw/2) - std::min(a + aw/2, b + bw/2));
}

struct box {
  double x, y, w, h;
};

struct detection {
  box boundingBox;
  std::vector<double> classProbabilities;
  double objectness;
};

double boxIntersection(box& a, box& b) {
  double w = lineOverlap(a.x, a.w, b.x, b.w);
  double h = lineOverlap(a.y, a.h, b.y, b.h);
  assert(w >= 0 && h >= 0);
  return w * h;
}

double boxUnion(box& a, box& b) {
  return a.w * a.h + b.w * b.h - boxIntersection(a,b);
}
float iou(box& a, box& b) { return boxIntersection(a, b) / boxUnion(a, b); }

void correctBox(box& b, size_t imageWidth, size_t imageHeight, size_t netWidth, size_t netHeight) {
    int newW = 0;
    int newH = 0;
    if (((float)netWidth/imageWidth) < ((float)netHeight/imageHeight)) {
        newW = netWidth;
        newH = (imageHeight * netWidth)/imageWidth;
    } else {
        newH = netHeight;
        newW = (imageWidth * netHeight)/imageHeight;
    }
    b.x = (b.x - (netWidth - newW)/2./netWidth) / ((float)newW/netWidth);
    b.y = (b.y - (netHeight - newH)/2./netHeight) / ((float)newH/netHeight);
    b.w *= (float)netWidth/newW;
    b.h *= (float)netHeight/newH;
}

size_t outputIndex(size_t mask, size_t cell, size_t offset, size_t width, size_t height, size_t classes) {
  assert(mask >= 0 && mask < 3);//3 should be const?
  assert(cell >= 0 && cell < width * height);
  assert(offset >= 0 && offset < classes + 5);
  return mask * width * height * (4 + 1 + classes) + offset * width * height + cell;
}

// TODO: batchSize = 1
box getBox(const arma::mat& output, 
           size_t mask, 
           size_t cell, 
           size_t layerWidth, 
           size_t layerHeight, 
           size_t width, 
           size_t height, 
           size_t classes,
           std::pair<size_t, size_t>& anchor) {
  box b;
  // x,y : add row/col, / 13
  // w, h: use anchors here, then / 416

  size_t i = cell / layerHeight;
  size_t j = cell % layerHeight;

  size_t xidx = outputIndex(mask, cell, 0, layerWidth, layerHeight, classes);
  size_t yidx = outputIndex(mask, cell, 1, layerWidth, layerHeight, classes);
  size_t widx = outputIndex(mask, cell, 2, layerWidth, layerHeight, classes);
  size_t hidx = outputIndex(mask, cell, 3, layerWidth, layerHeight, classes);

  std::cout << "Before yolo: " << b.x << ", " << b.y << ", " << b.w << ", " << b.h << "\n";
  b.x = (i + output(xidx, 0))/layerWidth;
  b.y = (j + output(yidx, 0))/layerHeight;
  b.w = anchor.first * std::exp(output(widx, 0))/width;
  b.h = anchor.second * std::exp(output(hidx, 0))/height;
  std::cout << "After yolo: " << b.x << ", " << b.y << ", " << b.w << ", " << b.h << "\n";
  return b;
}

//assuming batchSize of 1 again
void getDetections(arma::mat& output, 
                   std::vector<detection>& detections,
                   std::vector<size_t>& outputDims, 
                   std::vector<std::pair<size_t, size_t>>& anchors,
                   size_t width,
                   size_t height,
                   size_t imageWidth,
                   size_t imageHeight,
                   double ignoreThresh) {
  size_t gridWidth = outputDims[0];
  size_t gridHeight = outputDims[1];
  size_t gridDepth = outputDims[2];

  // TODO: assuming 3 masks of course. fix pls
  size_t classes = gridDepth / 3 - 5;

  for (size_t i = 0; i < gridWidth * gridHeight; i++) {//column major
    for (size_t n = 0; n < anchors.size(); n++) {
      size_t oidx = outputIndex(n, i, 4, gridWidth, gridHeight, classes);
      double objectness = output(oidx, 0);
      if (objectness < ignoreThresh) {
        continue;
      }
      detection d;
      d.objectness = objectness;
      d.boundingBox = getBox(output, n, i, gridWidth, gridHeight, width, height, classes, anchors[i]);
      correctBox(d.boundingBox, imageWidth, imageHeight, width, height);
      d.classProbabilities.resize(classes);
      for (size_t j = 0; j < classes; j++) {
        size_t cidx = outputIndex(n, i, 5 + j, gridWidth, gridHeight, classes);
        double probability = objectness * output(cidx, 0);
        d.classProbabilities[j] = probability * (probability > ignoreThresh);
      }
      detections.push_back(d);
    }
  }
}

void columnMajorLayout(const arma::mat &src, const ImageInfo &info, arma::mat &dest) {
  size_t width = info.Width();
  size_t height = info.Height();
  size_t channels = info.Channels();
  std::vector<double> data(width * height * channels);
  for(size_t c = 0; c < channels; c++) {
    for(size_t w = 0; w < width; w++) {
      for(size_t h = 0; h < height; h++) {
        data[(c * height * width) + (w * height) + h] = src((h * channels * width) + (w * channels) + c, 0);
      }
    }
  }
  dest = arma::mat(data);
}

void imageLayout(const arma::mat& src, const ImageInfo& info, arma::mat& dest) {
  size_t width = info.Width();
  size_t height = info.Height();
  size_t channels = info.Channels();
  std::vector<double> data(width * height * channels);
  for(size_t c = 0; c < channels; c++) {
    for(size_t w = 0; w < width; w++) {
      for(size_t h = 0; h < height; h++) {
        data[(h * channels * width) + (w * channels) + c] = src((c * height * width) + (w * height) + h, 0);
      }
    }
  }
  dest = arma::mat(data);
}

void nms(std::vector<detection>& detections, size_t classes, double iouThresh) {
  for (size_t i = 0; i < detections.size(); i++) {
    if (detections[i].objectness <= 0) {
      detections.erase(std::next(detections.begin(), i));
    }
  }
  for (size_t c = 0; c < classes; c++) {
    std::sort(detections.begin(), detections.end(), [=](const detection& a, const detection& b) {
        return a.classProbabilities[c] > b.classProbabilities[c];
    });
    for (size_t i = 0; i < detections.size(); i++) {
      if (detections[i].classProbabilities[c] == 0) continue;
      for (size_t j = 0; j < detections.size(); j++) {
        if (i != j && iou(detections[i].boundingBox, detections[j].boundingBox) > iouThresh) {
          detections[j].classProbabilities[c] = 0;
        }
      }
    }
  }
};

// NOTE: boxes have weird values
void drawBox(arma::mat& imageData, ImageInfo& imageInfo, int x1, int y1, int x2, int y2, double r) {
  assert(y1 <= y2);
  assert(x1 <= x2);

  x1 = std::clamp<int>(x1, 0, imageInfo.Width()-1);
  x2 = std::clamp<int>(x2, 0, imageInfo.Width()-1);
  y1 = std::clamp<int>(y1, 0, imageInfo.Height()-1);
  y2 = std::clamp<int>(y2, 0, imageInfo.Height()-1);

  for (int i = x1; i <= x2; i++) {
    size_t side1 = i * imageInfo.Height() + y1;
    size_t side2 = i * imageInfo.Height() + y2;
    imageData(side1, 0) = r;
    imageData(side2, 0) = r;

    imageData(side1 + imageInfo.Height() * imageInfo.Width(), 0) = 0;//g
    imageData(side2 + imageInfo.Height() * imageInfo.Width(), 0) = 0;
    imageData(side1 + imageInfo.Height() * imageInfo.Width() * 2, 0) = 0;//b
    imageData(side2 + imageInfo.Height() * imageInfo.Width() * 2, 0) = 0;
  }
  for (int i = y1; i <= y2; i++) {
    size_t side1 = x1 * imageInfo.Height() + i;
    size_t side2 = x2 * imageInfo.Height() + i;
    imageData(side1, 0) = r;
    imageData(side2, 0) = r;

    imageData(side1 + imageInfo.Height() * imageInfo.Width(), 0) = 0;//g
    imageData(side2 + imageInfo.Height() * imageInfo.Width(), 0) = 0;
    imageData(side1 + imageInfo.Height() * imageInfo.Width() * 2, 0) = 0;//b
    imageData(side2 + imageInfo.Height() * imageInfo.Width() * 2, 0) = 0;
  }
}

void drawBoundingBox(arma::mat& imageData, ImageInfo& imageInfo, box& bbox, size_t width) {
  width = std::clamp<size_t>(width, 0, 10);

  int x1 = bbox.x - bbox.w/2.0f;
  int x2 = bbox.x + bbox.w/2.0f;
  int y1 = bbox.y - bbox.h/2.0f;
  int y2 = bbox.y + bbox.h/2.0f;
  for (int i = 0; i < width; i++) {
    drawBox(imageData, imageInfo, x1 + i, y1 + i, x2 - i, y2 - i, 1.0f);
  }
}

// TODO: draw labels when drawing detections
void drawDetections(arma::mat& imageData, ImageInfo& imageInfo, std::vector<detection>& detections, size_t maxObjects) {
  for (size_t i = 0; i < maxObjects && i < detections.size(); i++) {
    drawBoundingBox(imageData, imageInfo, detections[i].boundingBox, 4);
  }
}

void columnMajorWeights() {}

void loadWeights(const std::string weights, mlpack::models::YoloV3Tiny<arma::mat> &model) {

}

using namespace mlpack;
int main(void) {
  const std::string inputFile = "input.jpg";
  const std::string outputFile = "output.jpg";
 
  double ignoreThresh = 0.5f;
  std::vector<std::pair<size_t, size_t>> anchors = { 
    {10, 14},
    {23, 27},
    {37, 58},
    {81, 82},
    {135, 169},
    {344, 319}
  };
  size_t maxObjects = 1;

  std::vector<size_t> largeDims = {13, 13, 255, 1};
  std::vector<size_t> smallDims = {26, 26, 255, 1};
  
  arma::mat inputData;
  ImageInfo inputInfo;
  
  arma::mat letterboxedInput;
  ImageInfo letterboxedInputInfo(416, 416, 3);
  arma::mat modelInput;
 
  arma::mat largeOutput;
  arma::mat smallOutput;
  load(inputFile, inputData, inputInfo);

  letterbox(inputData, inputInfo, letterboxedInput, letterboxedInputInfo);
  columnMajorLayout(letterboxedInput, letterboxedInputInfo, modelInput);
  
  mlpack::models::YoloV3Tiny<arma::mat> model(anchors);
  model.printModel();
  model.Predict(modelInput, largeOutput, smallOutput);

  std::vector<detection> detections;

  using Anchors = std::vector<std::pair<size_t, size_t>>;
  Anchors largeAnchors = Anchors(anchors.begin()+3, anchors.end());
  getDetections(largeOutput, 
                detections,
                largeDims,
                largeAnchors,
                letterboxedInputInfo.Width(),
                letterboxedInputInfo.Height(),
                inputInfo.Width(),
                inputInfo.Height(),
                ignoreThresh);

  Anchors smallAnchors = Anchors(anchors.begin(), anchors.begin()+3);
  getDetections(smallOutput, 
                detections,
                smallDims,
                smallAnchors,
                letterboxedInputInfo.Width(),
                letterboxedInputInfo.Height(),
                inputInfo.Width(),
                inputInfo.Height(),
                ignoreThresh);

  std::cout << "detections length: " << detections.size() << "\n";

  arma::mat columnMajorInputData;
  columnMajorLayout(inputData, inputInfo, columnMajorInputData);

  box& b = detections[0].boundingBox;
  std::cout << "First detection: " << b.x << ", " << b.y << ", " << b.w << ", " << b.h << "\n";

  drawDetections(columnMajorInputData, inputInfo, detections, maxObjects);
  imageLayout(columnMajorInputData, inputInfo, inputData);
  save(outputFile, inputData, inputInfo);
  return 0;
}
