#ifndef IMAGE_IMPL_HPP
#define IMAGE_IMPL_HPP

#include "image.hpp"

void CheckImage(const Image& image)
{
  const arma::fmat& data = image.data;
  const mlpack::data::ImageInfo& info = image.info;
  size_t expectedRows = info.Width() * info.Height() * info.Channels();
  if (data.n_rows != expectedRows || data.n_cols != 1)
  {
    std::ostringstream errMessage;
    errMessage << "Image is the incorrect shape. \n"
               << "data.n_rows: " << data.n_rows << "\n"
               << "Width() * Height() * Channels() => "
               << info.Width() << " * " << info.Height()
               << " * " << info.Channels() << " = "
               << expectedRows << "\n"
               << "data.n_cols: " << data.n_cols << "\n";
    throw std::logic_error(errMessage.str());
  }
}

/*
 *  Loads an image, normalize it values and convert to mlpack layout.
 */
void LoadImage(const std::string& file,
               Image& image,
               const bool grayscale)
{
  if (grayscale)
    image.info.Channels() = 1;
  Load(file, image.data, image.info, true);
  image.data /= 255.0f;
  image.data = mlpack::data::GroupChannels(image.data, image.info);

  // Hack so that other image processing functions work.
  if (grayscale)
  {
    image.data = arma::repmat(image.data, 3, 1);
    image.info.Channels() = 3;
  }

  if (image.info.Channels() == 4)
  {
    size_t width = image.info.Width();
    size_t height = image.info.Height();
    image.info.Channels() = 3;
    image.data = image.data.submat(0, 0, width * height * 3 - 1, 0);
  }
}

/*
 *  Saves an image back, reset RGB range back to 0-255 and convert to stb layout
 */
void SaveImage(const std::string& file, Image& image)
{
  CheckImage(image);
  arma::fmat stbData = mlpack::data::InterleaveChannels(image.data, image.info);
  stbData *= 255;
  Save(file, stbData, image.info, true);
}

/*
 *  Resizes an image using `resizedInfo`.
 *
 *  XXX: This is the same resizing used as darknet. If you load
 *  in the pretrained weights and use a different resizing method
 *  you will possibly get worse results.
 *
 *  TODO: try mlpack resize
 */
void ResizeImage(const Image& input, Image& output)
{
  CheckImage(input);
  CheckImage(output);

  std::cout << input.info.Channels() << "\n";
  std::cout << output.info.Channels() << "\n";
  assert(input.info.Channels() == output.info.Channels());
  const size_t channels = input.info.Channels();
  const size_t w = output.info.Width();
  const size_t h = output.info.Height();

  Image part(output.info.Width(), input.info.Height(), channels);
  float wScale = (float)(input.info.Width() - 1) / (w - 1);
  float hScale = (float)(input.info.Height() - 1) / (h - 1);

  for (size_t k = 0; k < channels; k++)
  {
    for (size_t r = 0; r < input.info.Height(); r++)
    {
      for (size_t c = 0; c < w; c++)
      {
        float val = 0;
        if (c == w - 1 || input.info.Width() == 1)
        {
          val = input.GetPixel(input.info.Width() - 1, r, k);
        }
        else
        {
          float sx = c * wScale;
          int ix = (int)sx;
          float dx = sx - ix;
          val = (1 - dx) * input.GetPixel(ix, r, k) +
            dx * input.GetPixel(ix + 1, r, k);
        }
        part.SetPixel(c, r, k, val);
      }
    }
  }

  for (int k = 0; k < channels; k++)
  {
    for (int r = 0; r < h; r++)
    {
      float sy = r * hScale;
      int iy = (int)sy;
      float dy = sy - iy;
      for (int c = 0; c < w; c++)
      {
        float val = (1 - dy) * part.GetPixel(c, iy, k);
        output.SetPixel(c, r, k, val);
      }

      if (r == h - 1 || input.info.Height() == 1)
        continue;

      for (int c = 0; c < w; c++)
      {
        float val = dy * part.GetPixel(c, iy + 1, k);
        output.SetPixel(c, r, k, output.GetPixel(c, r, k) + val);
      }
    }
  }
}

/*
 *  Embed `src` image within `dst` image starting at offset (dx, dy).
 */
void EmbedImage(const Image& src, Image& dst, const size_t dx, const size_t dy)
{
  CheckImage(src);
  CheckImage(dst);

  size_t width = std::min(src.info.Width() + dx, dst.info.Width());
  size_t height = std::min(src.info.Height() + dy, dst.info.Height());

  for (size_t c = 0; c < src.info.Channels(); c++)
  {
    for (size_t i = 0; i < src.info.Width(); i++)
    {
      if (dx + i >= dst.info.Width() || dx + i < 0)
        break;

      for (size_t j = 0; j < src.info.Height(); j++)
      {
        if (dy + j >= dst.info.Height() || dy + j < 0)
          break;

        dst.SetPixel(i + dx, j + dy, c, src.GetPixel(i, j, c));
      }
    }
  }
}

/*
 *  Resize `src` to a square image such that the aspect ratio is the same.
 *  Blank space will then be filled in with `grayValue`.
 *
 *  The original yolov3 model trained by Redmon et al. used this method
 *  of resizing images to keep them within 416x416 but also keeping
 *  the aspect ratio of the original image, instead of simply resizing
 *  to 416x416.
 */
void LetterBox(const Image& src, Image& dst)
{
  const double grayValue = 0.5;
  CheckImage(src);

  dst.data.clear();
  dst.data = arma::fmat(dst.info.Width() * dst.info.Height() * dst.info.Channels(), 1);
  dst.data.fill(grayValue);

  size_t width, height;
  if ((float)dst.info.Width() / src.info.Width() > (float)dst.info.Height() / src.info.Height())
  {
    height = dst.info.Height();
    width = src.info.Width() * dst.info.Height() / src.info.Height();
  }
  else
  {
    width = dst.info.Width();
    height = src.info.Height() * dst.info.Width() / src.info.Width();
  }

  Image resizedImage(width, height, dst.info.Channels());
  ResizeImage(src, resizedImage);
  EmbedImage(resizedImage, dst, (dst.info.Width() - width) / 2,
    (dst.info.Height() - height) / 2);
}

#endif
