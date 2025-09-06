#include <mlpack.hpp>
#include <stdexcept>

void CheckImage(const mlpack::data::ImageInfo& info,
                const arma::mat& data)
{
  if (data.n_rows != info.Width() * info.Height() * info.Channels() ||
    data.n_cols != 1 || info.Channels() != 3)
  {
    throw std::logic_error("Image is the incorrect shape.");
  }
}

/*
 *  Loads an image and normalizes it values.
 */
void LoadImage(const std::string& file,
               mlpack::data::ImageInfo& info,
               arma::mat& data)
{
  Load(file, data, info, true);
  data /= 255.0f;
}

/*
 *  Saves an image back, converting the RGB values from the range 0-1 to 0-255.
 */
void SaveImage(const std::string& file,
               mlpack::data::ImageInfo& info,
               arma::mat data)
{
  CheckImage(info, data);
  data *= 255;
  Save(file, data, info, true);
}

/*
 *  Resizes an image using `resizedInfo`.
 */
void ResizeImage(const mlpack::data::ImageInfo& oldInfo,
                 const arma::mat& oldImage,
                 const mlpack::data::ImageInfo& resizedInfo,
                 arma::mat& resizedImage)
{
  size_t newWidth = resizedInfo.Width();
  size_t newHeight = resizedInfo.Height();

  size_t oldWidth = oldInfo.Width();
  size_t oldHeight = oldInfo.Height();

  CheckImage(oldInfo, oldImage);

  resizedImage.clear();
  resizedImage = arma::mat(newWidth * newHeight * 3, 1);

  double xRatio = (double)(oldWidth - 1) / (newWidth - 1);
  double yRatio = (double)(oldHeight - 1) / (newHeight - 1);

  for (size_t channel = 0; channel < 3; channel++)
  {
    for (size_t w = 0; w < newWidth; w++)
    {
      for (size_t h = 0; h < newHeight; h++)
      {
        size_t xLow = std::floor(xRatio * w);
        size_t yLow = std::floor(yRatio * h);

        size_t xHigh = std::ceil(xRatio * w);
        size_t yHigh = std::ceil(yRatio * h);

        size_t xWeight = (xRatio * w) - xLow;
        size_t yWeight = (yRatio * h) - yLow;

        double a = oldImage.at(yLow * oldWidth * 3 + xLow * 3 + channel);
        double b = oldImage.at(yLow * oldWidth * 3 + xHigh * 3 + channel);
        double c = oldImage.at(yHigh * oldWidth * 3 + xLow * 3 + channel);
        double d = oldImage.at(yHigh * oldWidth * 3 + xHigh * 3 + channel);

        double value =
                a * (1 - xWeight) * (1 - yWeight) +
                b * xWeight * (1 - yWeight) +
                c * yWeight * (1 - xWeight) +
                d * xWeight * yWeight;

        resizedImage.at(h * newWidth * 3 + w * 3 + channel) = value;
      }
    }
  }
}

/*
 *  Embed `src` image within `dst` image starting at offset (dx, dy).
 */
void EmbedImage(const mlpack::data::ImageInfo& srcInfo, const arma::mat& src,
                const mlpack::data::ImageInfo& dstInfo, arma::mat& dst,
                const size_t dx, const size_t dy) {

  CheckImage(srcInfo, src);
  CheckImage(dstInfo, dst);

  size_t width = std::min(srcInfo.Width() + dx, dstInfo.Width());
  size_t height = std::min(srcInfo.Height() + dy, dstInfo.Height());

  for (size_t c = 0; c < srcInfo.Channels(); c++)
  {
    for (size_t i = 0; i < srcInfo.Width(); i++)
    {
      if (dx + i >= dstInfo.Width())
        break;

      for (size_t j = 0; j < srcInfo.Height(); j++)
      {
        if (dy + j >= dstInfo.Height())
          break;

        size_t sourceIndex = j * srcInfo.Channels() * srcInfo.Width() +
          i * srcInfo.Channels() + c;
        size_t destIndex = (j + dy) * dstInfo.Channels() * dstInfo.Width() +
          (i + dx) * dstInfo.Channels() + c;
        dst.at(destIndex) = src.at(sourceIndex);
      }
    }
  }
}

/*
 *  Resize `src` to a square image such that the aspect ratio is the same.
 *  Blank space will then be filled in with `grayValue`.
 *
 *  The original yolov3-tiny model trained by Redmon et al. used this method
 *  of resizing images to keep them within 416x416 but also keeping
 *  the aspect ratio of the original image, instead of simply resizing
 *  to 416x416.
 *
 *  XXX: The original model was trained on images whose blank
 *  space was filled with the rgb value (0.5, 0.5, 0.5). If inference
 *  is done with the same weights and other gray values, this will worsen
 *  the results of the network.
 */
void LetterBox(const mlpack::data::ImageInfo& srcInfo, const arma::mat& src,
               const mlpack::data::ImageInfo& dstInfo, arma::mat& dst,
               const double grayValue = 0.5)
{
  CheckImage(srcInfo, src);
  CheckImage(dstInfo, dst);

  size_t width, height;
  if (dstInfo.Width() / srcInfo.Width() > dstInfo.Height() / srcInfo.Height())
  {
    height = dstInfo.Height();
    width = srcInfo.Width() * dstInfo.Height() / srcInfo.Height();
  }
  else
  {
    width = dstInfo.Width();
    height = srcInfo.Height() * dstInfo.Width() / srcInfo.Width();
  }

  dst.fill(grayValue);
  arma::mat resizedSrc;
  mlpack::data::ImageInfo resizedInfo(width, height, srcInfo.Channels());
  ResizeImage(srcInfo, src, resizedInfo, resizedSrc);
  EmbedImage(resizedInfo, resizedSrc, dstInfo, dst, (dstInfo.Width() - width)/2,
    (dstInfo.Height() - height)/2);
}

/*
 *  Return how much a vertical or horizontal line overlap, where `a` and `b` are
 *  mid-points and `aw` and `bw` are widths/heights of those lines.
 */
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

void columnMajorLayout(const arma::mat &src, const mlpack::data::ImageInfo &info, arma::mat &dest) {
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

void imageLayout(const arma::mat& src, const mlpack::data::ImageInfo& info, arma::mat& dest) {
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

// NOTE: boxes have weird values
void drawBox(arma::mat& imageData, mlpack::data::ImageInfo& imageInfo, int x1, int y1, int x2, int y2, double r) {
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

void drawBoundingBox(arma::mat& imageData, mlpack::data::ImageInfo& imageInfo, box& bbox, size_t width) {
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
void drawDetections(arma::mat& imageData, mlpack::data::ImageInfo& imageInfo, std::vector<detection>& detections, size_t maxObjects) {
  for (size_t i = 0; i < maxObjects && i < detections.size(); i++) {
    drawBoundingBox(imageData, imageInfo, detections[i].boundingBox, 4);
  }
}

int main(void) {
  const std::string inputFile = "./images/dog.jpg";
  const std::string outputFile = "output.jpg";

  mlpack::data::ImageInfo info;
  arma::mat image;

  mlpack::data::ImageInfo newInfo(416, 416, 3);
  arma::mat newImage;
  newImage.resize(newInfo.Width() * newInfo.Height() * 3, 1);

  LoadImage(inputFile, info, image);
  LetterBox(info, image, newInfo, newImage);
  SaveImage(outputFile, newInfo, newImage);

  // double ignoreThresh = 0.5f;
  // std::vector<std::pair<size_t, size_t>> anchors = {
  //   {10, 14},
  //   {23, 27},
  //   {37, 58},
  //   {81, 82},
  //   {135, 169},
  //   {344, 319}
  // };
  // size_t maxObjects = 1;
  //
  // std::vector<size_t> largeDims = {13, 13, 255, 1};
  // std::vector<size_t> smallDims = {26, 26, 255, 1};
  //
  // mat inputData;
  // mlpack::data::ImageInfo inputInfo;
  //
  // mat letterboxedInput;
  // mlpack::data::ImageInfo letterboxedInputInfo(416, 416, 3);
  // mat modelInput;
  //
  // mat largeOutput;
  // mat smallOutput;
  // load(inputFile, inputData, inputInfo);
  //
  // letterbox(inputData, inputInfo, letterboxedInput, letterboxedInputInfo);
  // columnMajorLayout(letterboxedInput, letterboxedInputInfo, modelInput);
  //
  // std::vector<detection> detections;
  //
  // using Anchors = std::vector<std::pair<size_t, size_t>>;
  // Anchors largeAnchors = Anchors(anchors.begin()+3, anchors.end());
  // Anchors smallAnchors = Anchors(anchors.begin(), anchors.begin()+3);
  //
  // std::cout << "detections length: " << detections.size() << "\n";
  //
  // arma::mat columnMajorInputData;
  // columnMajorLayout(inputData, inputInfo, columnMajorInputData);
  //
  // box& b = detections[0].boundingBox;
  // std::cout << "First detection: " << b.x << ", " << b.y << ", " << b.w << ", " << b.h << "\n";
  //
  // drawDetections(columnMajorInputData, inputInfo, detections, maxObjects);
  // imageLayout(columnMajorInputData, inputInfo, inputData);
  // save(outputFile, inputData, inputInfo);
  // return 0;
}
