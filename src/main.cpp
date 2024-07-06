#include <mlpack.hpp>
#include <armadillo>

struct detection {
  double x, y, w, h;
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
                         mlpack::data::ImageInfo& outputInfo,
                         double widthScale, 
                         double heightScale) {
  assert(widthScale > 0 && heightScale > 0);
  size_t newWidth = std::floor(info.Width() * widthScale);
  size_t newHeight = std::floor(info.Height() * heightScale);

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

        //double a = data.at(yLow * info.Width() + xLow);
        double a = data.at(yLow * info.Width() * info.Channels() + xLow * info.Channels() + channel);
        //double b = data.at(yLow * info.Width() + xHigh);
        double b = data.at(yLow * info.Width() * info.Channels() + xHigh * info.Channels() + channel);
        //double c = data.at(yHigh * info.Width() + xLow);
        double c = data.at(yHigh * info.Width() * info.Channels() + xLow * info.Channels() + channel);
        //double d = data.at(yHigh * info.Width() + xHigh);
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

int main(void) {
  const std::string input = "input.jpg";
  const std::string output = "output.jpg";

  mlpack::data::ImageInfo inputInfo;
  mlpack::data::ImageInfo outputInfo;

  arma::mat inputData;
  load(input, inputData, inputInfo);
  save(output, inputData, inputInfo);

  arma::mat outputData = resize(inputData, inputInfo, outputInfo, 2, 2);
  //fill(outputData, 1);
  //embed(inputData, inputInfo, outputData, outputInfo, 600, 801);

  save(output, outputData, outputInfo);

  return 0;
}
