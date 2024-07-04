#include <cstdint>
#include <mlpack.hpp>
#include <armadillo>
#include "../models/models/yolov3_tiny/yolov3_tiny.hpp"

#define INPUT "input.jpg"
#define OUTPUT "output.jpg"
#define WEIGHTS "yolov3-tiny.weights"

using namespace mlpack::models;

struct detection {
  double x, y, w, h;
  std::vector<double> classProbabilities;
  double objectness;
};

//needs indexing stuff
// detection getDetection(arma::mat yoloOutputs, std::vector<double> yoloDimensions) {
// }

class Image {
public:
  Image(size_t width, size_t height, size_t channels) :
  width(width),
  height(height),
  channels(channels)
  {
    data = std::vector<double>(width * height * channels);
  }

  void load(const char* fileName);
  void save(const char* fileName);
  void embed(Image& source, Image& dest, size_t dx, size_t dy);
  void resize(Image& source, Image& dest, size_t newWidth, size_t newHeight);
  void letterboxImage(size_t newWidth, size_t newHeight);
  void drawBox(double x, double y, double w, double h);

  void setPixel(size_t x, size_t y, size_t c, double value) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < width);
    assert(c >= 0 && c < channels);

    size_t index = c * width * height + y * width + x;
    data[index] = value;
  }

  double getPixel(size_t x, size_t y, size_t c, double value) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < width);
    assert(c >= 0 && c < channels);

    size_t index = c * width * height + y * width + x;
    return data[index];
  }
private:
  size_t width, height, channels;
  std::vector<double> data;
};

int main(void) {

  return 0;
}
