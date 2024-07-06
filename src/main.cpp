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

int main(void) {
  const std::string input = "input.jpg";
  const std::string output = "output.jpg";

  mlpack::data::ImageInfo inputInfo;
  mlpack::data::ImageInfo outputInfo(1000, 600, 3);

  arma::mat inputData;
  load(input, inputData, inputInfo);
  save(output, inputData, inputInfo);

  arma::mat outputData = resize(inputData, inputInfo, outputInfo);
  //fill(outputData, 1);
  //embed(inputData, inputInfo, outputData, outputInfo, 600, 801);
  letterbox(inputData, inputInfo, outputData, outputInfo);

  save(output, outputData, outputInfo);

  return 0;
}
