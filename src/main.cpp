#include <limits>
#include <mlpack.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>

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


template <typename MatType = arma::mat>
class yolov3tiny {
public:
  yolov3tiny()
  {
    model = mlpack::DAGNetwork();
  }

private:
  
  using Type = typename MatType::elem_type;

  size_t Convolution(size_t maps, size_t kernel, bool batchNorm,
    Type negativeSlope)
  {
    if (kernel != 3 || kernel != 1)
      throw std::logic_error("Kernel size for convolutions in yolov3-tiny"
        "must be 3 or 1");

    size_t pad = kernel == 3 ? 1 : 0;
    mlpack::MultiLayer<MatType> block;
    block.template Add<mlpack::Convolution<MatType>>(
      maps, kernel,kernel, 1, 1, pad, pad, "none", !batchNorm);

    if (batchNorm)
    {
      // set epsilon to zero, couldn't find it used in darknet/ggml.
      block.template Add<mlpack::BatchNorm<MatType>>(2, 2, 0, false, 0.1f);
    }
    block.template Add<mlpack::LeakyReLU<MatType>>(negativeSlope);
    return model.Add(block);
  }

  size_t MaxPool2D(size_t stride)
  {
    // All max pool layers have kernel size 2
    mlpack::MultiLayer<MatType> block;
    if (stride == 1)
    {
      // One layer with odd width and height input has kernel size 2, stride 1,
      // so padding on the right and bottom are needed.
      Type min = -std::numeric_limits<Type>();
      block.template Add<mlpack::Padding<MatType>>(0, 1, 0, 1, min);
    }
    block.template Add<mlpack::MaxPooling<MatType>>(2, 2, stride, stride);
    return model.Add(block);
  }


  mlpack::DAGNetwork<> model;
};

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

  return 0;
}
