/*
 *
 * @article{DBLP:journals/corr/abs-1804-02767,
 *   author       = {Joseph Redmon and Ali Farhadi},
 *   title        = {YOLOv3: An Incremental Improvement},
 *   journal      = {CoRR},
 *   volume       = {abs/1804.02767},
 *   year         = {2018},
 *   url          = {http://arxiv.org/abs/1804.02767},
 *   eprinttype    = {arXiv},
 *   eprint       = {1804.02767},
 *   timestamp    = {Mon, 13 Aug 2018 16:48:24 +0200},
 *   biburl       = {https://dblp.org/rec/journals/corr/abs-1804-02767.bib},
 *   bibsource    = {dblp computer science bibliography, https://dblp.org}
 * }
 *
 */

#include <mlpack.hpp>

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
  return std::abs(std::max(a - aw / 2, b - bw / 2) -
                  std::min(a + aw / 2, b + bw / 2));
}

template <typename MatType = arma::mat>
class YOLOv3Layer : public mlpack::Layer<MatType>
{
 public:
  YOLOv3Layer(size_t imgSize,
              size_t numAttributes,
              std::vector<typename MatType::elem_type> anchors) :
    imgSize(imgSize), numAttributes(numAttributes), anchorsSetup(false)
  {
    if (anchors.size() != 6)
      throw std::logic_error("YOLOv3-tiny must have 3 (w, h) anchors.");

    w = MatType(1, 3, arma::fill::none);
    h = MatType(1, 3, arma::fill::none);

    for (size_t i = 0; i < 3; i++)
    {
      w.at(0, i) = anchors[i * 2];
      h.at(0, i) = anchors[i * 2 + 1];
    }
  }

  YOLOv3Layer* Clone() const override { return new YOLOv3Layer(*this); }

  void ComputeOutputDimensions() override
  {
    if (this->inputDimensions.size() != 3) 
    {
      std::ostringstream errMessage;
      errMessage << "YOLOv3Layer::ComputeOutputDimensions(): "
                 << "Input dimensions must be 3D, but there are "
                 << this->inputDimensions.size() << " input "
                 << "dimensions";
      throw std::logic_error(errMessage.str());
    }

    if (this->inputDimensions[0] != this->inputDimensions[1])
      throw std::logic_error("YOLOv3Layer::ComputeOutputDimensions(): "
        "Input dimensions must be square.");
    gridSize = this->inputDimensions[0] * this->inputDimensions[1];
    this->outputDimensions = { gridSize * numAttributes * 3 };
    // this->outputDimensions = { gridSize * 3, numAttributes }; // TODO: should be this
    // this->outputDimensions = { numAttributes, gridSize * 3 }; // or this?

    if (!anchorsSetup)
    {
      w = arma::repmat(w, gridSize, 1);
      h = arma::repmat(h, gridSize, 1);
      anchorsSetup = true;
    }
  }

  void Forward(const MatType& input, MatType& output) override
  {
    Type stride = imgSize / (Type)(gridSize);
    size_t batchSize = input.n_cols;
    output.set_size(input.n_rows, batchSize);

    CubeType inputCube;
    mlpack::MakeAlias(inputCube, input, gridSize * numAttributes, 3,
      batchSize);

    CubeType outputCube;
    mlpack::MakeAlias(outputCube, output, gridSize * numAttributes, 3,
      batchSize);

    // TODO: move repcubes to be in place s.t armadillo optimizes stuff better
    MatType offset = arma::regspace(0, this->inputDimensions[0] - 1);
    CubeType xOffset = arma::repcube(offset, this->inputDimensions[0], 3, batchSize);
    // Theres probably a better way.
    CubeType yOffset = arma::repcube(arma::vectorise(arma::repmat(offset.t(), this->inputDimensions[0], 1)),
      1, 3, batchSize);

    // x
    outputCube.tube(0, 0, gridSize - 1, 2) =
      (xOffset + 1 / (1 + arma::exp(inputCube.tube(0, 0, gridSize - 1, 2)))) * stride;

    // y
    outputCube.tube(gridSize, 0, gridSize * 2 - 1, 2) =
      (yOffset + 1 / (1 + arma::exp(inputCube.tube(gridSize, 0, gridSize * 2 - 1, 2)))) * stride;

    // w
    outputCube.tube(gridSize * 2, 0, gridSize * 3 - 1, 2) =
      arma::repcube(w, 1, 1, batchSize) % arma::exp(inputCube.tube(gridSize * 2, 0, gridSize * 3 - 1, 2));

    // h
    outputCube.tube(gridSize * 3, 0, gridSize * 4 - 1, 2) =
      arma::repcube(h, 1, 1, batchSize) % arma::exp(inputCube.tube(gridSize * 3, 0, gridSize * 4 - 1, 2));

    // Copy objects and classification logits.
    outputCube.tube(gridSize * 4, 0, outputCube.n_rows - 1, 2) =
      inputCube.tube(gridSize * 4, 0, inputCube.n_rows - 1, 2);
  }

  void Backward(const MatType& input, const MatType& output,
    const MatType& gy, MatType& g) override
  {
    throw std::runtime_error("YOLOv3tiny::Backward() not implemented.");
  }

 private:

  using Type = typename MatType::elem_type;

  using CubeType = typename GetCubeType<MatType>::type;

  size_t imgSize;
  size_t numAttributes;
  size_t gridSize;
  MatType w;
  MatType h;

  bool anchorsSetup;

};

template <typename MatType = arma::mat>
class YOLOv3tiny {
 public:
  YOLOv3tiny(size_t imgSize, size_t classes) :
    imgSize(imgSize), classes(classes)
  {
    // x, y, w, h, objectness score, n classes (COCO = 80)
    numAttributes = 5 + classes;
    scale = { 2.0, 2.0 };

    model = mlpack::DAGNetwork();
    model.SetNetworkMode(false);
    model.InputDimensions() = { imgSize, imgSize, 3 };

    size_t convolution0 = Convolution(16, 3);
    size_t maxPool1 = MaxPool2x2(2);
    size_t convolution2 = Convolution(32, 3);
    size_t maxPool3 = MaxPool2x2(2);
    size_t convolution4 = Convolution(64, 3);
    size_t maxPool5 = MaxPool2x2(2);
    size_t convolution6 = Convolution(128, 3);
    size_t maxPool7 = MaxPool2x2(2);
    size_t convolution8 = Convolution(256, 3);
    size_t maxPool9 = MaxPool2x2(2);
    size_t convolution10 = Convolution(512, 3);
    size_t maxPool11 = MaxPool2x2(1);
    size_t convolution12 = Convolution(1024, 3);
    size_t convolution13 = Convolution(256, 1);

    // Detection head for larger objects.
    size_t convolution14 = Convolution(512, 3);
    size_t convolution15 = Convolution(3 * numAttributes, 1, false);
    size_t detections16 = YOLO(imgSize, { 81, 82, 135, 169, 344, 319 });

    size_t convolution17 = Convolution(128, 1);
    // Upsample for more fine-grained detections.
    size_t upsample18 = model.Add<mlpack::NearestInterpolation<MatType>>(scale);

    // Detection head for smaller objects.
    size_t convolution19 = Convolution(256, 3);
    size_t convolution20 = Convolution(3 * numAttributes, 1, false);
    size_t detections21 = YOLO(imgSize, { 10, 14, 23, 27, 37, 58 });

    // the DAGNetwork class requires a layer for concatenations, so we use
    // the Identity layer for pure concatentation, and no other compute.
    size_t concatLayer22 = model.Add<mlpack::Identity<MatType>>();

    model.Connect(convolution0, maxPool1);
    model.Connect(maxPool1, convolution2);
    model.Connect(convolution2, maxPool3);
    model.Connect(maxPool3, convolution4);
    model.Connect(convolution4, maxPool5);
    model.Connect(maxPool5, convolution6);
    model.Connect(convolution6, maxPool7);
    model.Connect(maxPool7, convolution8);

    model.Connect(convolution8, maxPool9);
    model.Connect(maxPool9, convolution10);
    model.Connect(convolution10, maxPool11);
    model.Connect(maxPool11, convolution12);
    model.Connect(convolution12, convolution13);

    model.Connect(convolution13, convolution14);
    model.Connect(convolution14, convolution15);
    model.Connect(convolution15, detections16);

    model.Connect(convolution13, convolution17);
    model.Connect(convolution17, upsample18);

    // Concat convolution8 + upsample18 => convolution19
    model.Connect(upsample18, convolution19); // TODO: double check order
    model.Connect(convolution8, convolution19);
    // Set axis not necessary, since default is channels.

    model.Connect(convolution19, convolution20); // TODO: double check order
    model.Connect(convolution20, detections21);
    // Again, set axis not necessary, since default is channels.
 
    // Concatenation order shouldn't matter.
    model.Connect(detections16, concatLayer22);
    model.Connect(detections21, concatLayer22);
    model.SetAxis(concatLayer22, 0);

    model.Reset();
  }

  void Training(const bool training)
  {
    model.SetNetworkMode(training);
  }

  void Predict(const MatType& input, MatType& output)
  {
    model.Predict(input, output);
  }

 private:

  using Type = typename MatType::elem_type;

  size_t Convolution(const size_t maps, const size_t kernel,
    const bool batchNorm = true, const Type reluSlope = 0.1)
  {
    if (kernel != 3 && kernel != 1)
    {
      std::ostringstream errMessage;
      errMessage << "Kernel size for convolutions in yolov3-tiny must be 3"
        "or 1, but you supplied " << kernel << ".\n";
      throw std::logic_error(errMessage.str());
    }

    size_t pad = kernel == 3 ? 1 : 0;
    mlpack::MultiLayer<MatType> block;
    block.template Add<mlpack::Convolution<MatType>>(
      maps, kernel,kernel, 1, 1, pad, pad, "none", !batchNorm);

    if (batchNorm)
    {
      // set epsilon to zero, couldn't find it used in darknet/ggml.
      block.template Add<mlpack::BatchNorm<MatType>>(2, 2, 0, false, 0.1f);
    }
    block.template Add<mlpack::LeakyReLU<MatType>>(reluSlope);
    return model.Add(block);
  }

  size_t MaxPool2x2(const size_t stride)
  {
    // All max pool layers have kernel size 2
    mlpack::MultiLayer<MatType> block;
    if (stride == 1)
    {
      // One layer with odd width and height input has kernel size 2, stride 1,
      // so padding on the right and bottom are needed.
      Type min = -arma::datum::inf;
      block.template Add<mlpack::Padding<MatType>>(0, 1, 0, 1, min);
    }
    block.template Add<mlpack::MaxPooling<MatType>>(2, 2, stride, stride);
    return model.Add(block);
  }

  size_t YOLO(const size_t imgSize, const std::vector<double>& anchors) {
    return model.Add<YOLOv3Layer<MatType>>(imgSize, numAttributes, anchors);
  }


  size_t imgSize;
  size_t classes;
  size_t numAttributes;
  std::vector<double> scale;

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

  arma::mat detections;
  YOLOv3tiny model(416, 80);
  model.Training(false);
  model.Predict(image, detections);

  // detections shape should be (85, 2535)

  // LetterBox(info, image, newInfo, newImage);
  // SaveImage(outputFile, newInfo, newImage);

  return 0;
}
