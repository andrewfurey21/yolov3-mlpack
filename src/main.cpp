#include <mlpack.hpp>
#include <armadillo>

#include "../models/models/yolov3_tiny/yolov3_tiny.hpp"

struct box {
  double x, y, w, h;
};

struct detection {
  box boundingBox;
  std::vector<double> classProbabilities;
  double objectness;
};

void load(const std::string& file, arma::Mat<double>& data, mlpack::data::ImageInfo& info) {
  mlpack::data::Load(file, data, info, true);
  data /= 255.0f;
}

void save(const std::string& file, arma::Mat<double> data, mlpack::data::ImageInfo& info) {
  data *= 255;
  mlpack::data::Save(file, data, info, true);
}

void fill(arma::Mat<double>& data, double value) {
  value = std::clamp<double>(value, 0.0f, 1.0f);
  data.fill(value);
}

arma::Mat<double> resize(const arma::Mat<double>& data, 
                         mlpack::data::ImageInfo& info, 
                         mlpack::data::ImageInfo& outputInfo) {

  size_t newWidth = outputInfo.Width();
  size_t newHeight = outputInfo.Height();

  assert(data.n_rows == info.Width() * info.Height() * 3 && data.n_cols == 1);
  assert(newWidth > 1);
  assert(newHeight > 1);
  arma::Mat<double> resized(newWidth * newHeight * 3, 1);
  outputInfo = mlpack::data::ImageInfo(newWidth, newHeight, 3);

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
           mlpack::data::ImageInfo& sourceInfo, 
           arma::Mat<double>& dest,
           mlpack::data::ImageInfo& destInfo,
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
               mlpack::data::ImageInfo sourceInfo,
               arma::Mat<double>& dest,
               mlpack::data::ImageInfo& destInfo) {
  size_t width, height;
  if (destInfo.Width() / sourceInfo.Width() > destInfo.Height() / sourceInfo.Height()) {
    height = destInfo.Height();
    width = sourceInfo.Width() * destInfo.Height() / sourceInfo.Height();
  } else {
    width = destInfo.Width();
    height = sourceInfo.Height() * destInfo.Width() / sourceInfo.Width();
  }

  mlpack::data::ImageInfo resizedInfo(width, height, 3);
  arma::Mat<double> resized = resize(source, sourceInfo, resizedInfo);

  dest = arma::Mat<double>(destInfo.Width() * destInfo.Height() * destInfo.Channels(), 1);
  fill(dest, .3);
  embed(resized, resizedInfo, dest, destInfo, (destInfo.Width() - width)/2, (destInfo.Height() - height)/2);
}

void tile(arma::Mat<double>& a,
          mlpack::data::ImageInfo aInfo,
          arma::Mat<double>& b,
          mlpack::data::ImageInfo& bInfo,
          arma::Mat<double>& output,
          mlpack::data::ImageInfo& outputInfo,
          size_t dx) {
  assert(aInfo.Channels() == bInfo.Channels());
  size_t height = std::max(aInfo.Height(), bInfo.Height());
  size_t width = dx + aInfo.Width() + bInfo.Width();
  output = arma::Mat<double>(width * height * aInfo.Channels(), 1);
  outputInfo = mlpack::data::ImageInfo(width, height, aInfo.Channels());
  fill(output, 1.0f);
  embed(a, aInfo, output, outputInfo, 0, 0);
  embed(b, bInfo, output, outputInfo, aInfo.Width()+dx, 0);
}

void border(arma::Mat<double>& source,
            mlpack::data::ImageInfo sourceInfo,
            arma::Mat<double>& dest,
            mlpack::data::ImageInfo& destInfo,
            size_t borderSize) {
  destInfo = mlpack::data::ImageInfo(sourceInfo.Width() + 2 * borderSize, sourceInfo.Height() + 2 * borderSize, sourceInfo.Channels());
  dest = arma::Mat<double>(destInfo.Width() * destInfo.Height() * destInfo.Channels(), 1);
  fill(dest, 1.0f);
  embed(source, sourceInfo, dest, destInfo, borderSize, borderSize);
}



double lineOverlap(double a, double aw, double b, double bw) {
  return std::max(a - aw/2, b - bw/2) - std::min(a + aw/2, b + bw/2);
}

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
    int new_w = 0;
    int new_h = 0;
    if (((float)netWidth/imageWidth) < ((float)netHeight/imageHeight)) {
        new_w = netWidth;
        new_h = (imageHeight * netWidth)/imageWidth;
    } else {
        new_h = netHeight;
        new_w = (imageWidth * netHeight)/imageHeight;
    }
    b.x = (b.x - (netWidth - new_w)/2./netWidth) / ((float)new_w/netWidth);
    b.y = (b.y - (netHeight - new_h)/2./netHeight) / ((float)new_h/netHeight);
    b.w *= (float)netWidth/new_w;
    b.h *= (float)netHeight/new_h;
}

std::vector<detection> getDetections(arma::mat& output, std::vector<size_t> outputDims) {
  return {}; 
}

std::vector<detection> nms_sort(std::vector<detection> detections) {return {};};
void drawDetections(arma::mat& imageData, mlpack::data::ImageInfo& imageInfo, std::vector<detection>& detections) {}
void printLayer(mlpack::Layer<arma::mat>* layer, size_t layerIndex) {
  int width = layer->OutputDimensions()[0];
  int height = layer->OutputDimensions()[1];
  int channels = layer->OutputDimensions()[2];
  int batch = layer->OutputDimensions()[3];
  printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", (int)layerIndex, width, height, channels, batch);
  //std::cout << "Layer " << layerIndex << ", output shape: " << width << " x " << height << " x " << channels << " x " << batch << "\n";
}

int main(void) {
  const std::string input = "input.jpg";
  const std::string output = "output.jpg";
  
  arma::mat inputData;
  arma::mat predictions;
  mlpack::data::ImageInfo inputInfo;
  mlpack::data::ImageInfo resizeInfo(416, 416, 3);
  
  load(input, inputData, inputInfo);
  arma::mat resizeData = resize(inputData, inputInfo, resizeInfo);
  
  mlpack::models::YoloV3Tiny<arma::mat> yolo({ 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 });
  
  auto model = yolo.Model();
  model.InputDimensions() = std::vector<size_t>({resizeInfo.Width(), resizeInfo.Height(), resizeInfo.Channels(), 1});

  std::cout << "Number of layers: " << model.Network().size() << "\n";
  std::cout << "resized dims: " << resizeInfo.Width() << " x " << resizeInfo.Height() << " x " << resizeInfo.Channels() << "\n";
  model.Predict(resizeData, predictions, 1);
  for (size_t i = 0; i < model.Network().size(); i++) {
    if (i == 16 || i == 17 || i == 20) continue;
    auto layer = model.Network()[i];
    printLayer(layer, i);
  }

  return 0;
}
