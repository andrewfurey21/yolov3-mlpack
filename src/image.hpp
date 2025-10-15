#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <mlpack.hpp>

class Image
{
 public:
  Image() {}

  Image(const size_t width, const size_t height, const size_t channels)
  {
    info = mlpack::data::ImageInfo(width, height, channels);
    data = arma::fmat(width * height * channels, 1, arma::fill::zeros);
  }

  Image(const arma::fmat& data, const mlpack::data::ImageInfo& info) :
    data(data), info(info)
  {}

  void SetPixel(int x, int y, int c, float val)
  {
    if (x < 0 && x >= info.Width()) return;
    if (y < 0 && y >= info.Height()) return;
    data.at(x + y * info.Width() + c * info.Width() * info.Height(), 0) = val;
  }

  float GetPixel(int x, int y, int c) const
  {
    assert(x >= 0 && x < info.Width());
    assert(y >= 0 && y < info.Height());
    return data.at(x + y * info.Width() + c * info.Width() * info.Height(), 0);
  }

  mlpack::data::ImageInfo info;
  arma::fmat data;
};

void CheckImage(const Image& image);
void LoadImage(const std::string& file,
               Image& image,
               const bool grayscale = false);
void SaveImage(const std::string& file, Image& image);
void ResizeImage(const Image& input, Image& output);
void EmbedImage(const Image& src, Image& dst, const size_t dx, const size_t dy);
void LetterBox(const Image& src, Image& dst);

#include "image_impl.hpp"

#endif
