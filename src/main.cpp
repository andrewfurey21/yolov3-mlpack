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
  void embed(Image& source, Image& dest, size_t dx, size_t dy) {
    for (size_t c = 0; c < source.channels; c++) {
      for (size_t y = 0; y < source.height; y++) {
        for (size_t x = 0; x < source.width; x++) {
          double value = source.getPixel(x, y, c);
          dest.setPixel(x+dx, y+dy, c, value);
        }
      }
    }
  }
  void resize(Image& source, Image& dest, size_t newWidth, size_t newHeight);
  void letterboxImage(size_t newWidth, size_t newHeight);

  void drawBox(size_t x, size_t y, size_t w, size_t h, size_t border) {
    if (x < 0) x = 0;
    if (x >= this->width) x = this->width - 1;

    if (y < 0) y = 0;
    if (y >= this->height) y = this->height - 1;

    if (x+w > this->width) w = this->width - x;
    if (x+w > this->width) w = this->width - x;

    if (y+h > this->height) h = this->height - y;
    if (y+h > this->height) h = this->height - y;

    for (size_t b = 0; b < border; b++) {
      for (size_t i = x; i < w; i++) {
        for (size_t j = y; j < h; j++) {
          setPixel(i, j, 0, 255);//TODO:dont fill
          setPixel(i, j, 1, 255);
          setPixel(i, j, 2, 255);
        }
      }
    }
  }

  void setPixel(size_t x, size_t y, size_t c, double value) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < width);
    assert(c >= 0 && c < channels);

    size_t index = c * width * height + y * width + x;
    data[index] = value;
  }

  double getPixel(size_t x, size_t y, size_t c) {
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
