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
  size_t expectedRows = info.Width() * info.Height() * info.Channels();
  if (data.n_rows != expectedRows || data.n_cols != 1 || info.Channels() != 3)
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
               mlpack::data::ImageInfo& info,
               arma::mat& data)
{
  Load(file, data, info, true);
  data /= 255.0f;
  data = mlpack::data::ImageLayout(data, info);
}

/*
 *  Saves an image back, reset RGB range back to 0-255 and convert to stb layout
 */
void SaveImage(const std::string& file,
               mlpack::data::ImageInfo& info,
               arma::mat& data)
{
  CheckImage(info, data);
  arma::mat stbData = mlpack::data::STBLayout(data, info);
  stbData *= 255;
  Save(file, stbData, info, true);
}

/*
 *  Resizes an image using `resizedInfo`.
 */
void ResizeImage(const mlpack::data::ImageInfo& info,
                 const arma::mat& image,
                 const mlpack::data::ImageInfo& resizedInfo,
                 arma::mat& resizedImage)
{
  size_t newWidth = resizedInfo.Width();
  size_t newHeight = resizedInfo.Height();

  size_t width = info.Width();
  size_t height = info.Height();

  CheckImage(info, image);

  resizedImage.clear();
  resizedImage = arma::mat(newWidth * newHeight * resizedInfo.Channels(), 1);

  double xRatio = (double)(width - 1) / (newWidth - 1);
  double yRatio = (double)(height - 1) / (newHeight - 1);

  for (size_t channel = 0; channel < info.Channels(); channel++)
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

        double a = image.at(yLow + xLow * height + channel * width * height);
        double b = image.at(yLow + xHigh * height + channel * width * height);
        double c = image.at(yHigh + xLow * height + channel * width * height);
        double d = image.at(yHigh + xHigh * height + channel * width * height);

        double value =
                a * (1 - xWeight) * (1 - yWeight) +
                b * xWeight * (1 - yWeight) +
                c * yWeight * (1 - xWeight) +
                d * xWeight * yWeight;

        resizedImage.at(h + w * newHeight + channel * newWidth * newHeight) =
          value;
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
  dst.clear();
  dst = arma::mat(dstInfo.Width() * dstInfo.Height() * dstInfo.Channels(), 1);

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

        size_t sourceIndex = j + i * srcInfo.Height() +
          c * srcInfo.Height() * srcInfo.Width();
        size_t destIndex = (j + dy) + (i + dx) * dstInfo.Height() +
          c * dstInfo.Height() * dstInfo.Width();
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
  dst.clear();
  dst = arma::mat(dstInfo.Width() * dstInfo.Height() * dstInfo.Channels(), 1);

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
              size_t gridSize,
              size_t predictionsPerCell,
              std::vector<typename MatType::elem_type> anchors) :
    imgSize(imgSize),
    numAttributes(numAttributes),
    gridSize(gridSize),
    grid(gridSize * gridSize),
    predictionsPerCell(predictionsPerCell)
  {
    if (anchors.size() != 2 * predictionsPerCell)
    {
      std::ostringstream errMessage;
      errMessage << "YOLOv3-tiny must have " << predictionsPerCell
                  << " (w, h) anchors but you gave "
                  << anchors.size() / 2 << ".";
      throw std::logic_error(errMessage.str());
    }

    w = MatType(grid, predictionsPerCell, arma::fill::none);
    h = MatType(grid, predictionsPerCell, arma::fill::none);

    // TODO: Could maybe use .each_row()?
    for (size_t i = 0; i < predictionsPerCell; i++)
    {
      w.col(i).fill(anchors[i * 2]);
      h.col(i).fill(anchors[i * 2 + 1]);
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

    if (grid != this->inputDimensions[0] * this->inputDimensions[1])
      throw std::logic_error("YOLOv3Layer::ComputeOutputDimensions(): "
        "Grid is the wrong size.");
    this->outputDimensions = { numAttributes, grid * predictionsPerCell };
  }

  // Output format: cx, cy, w, h
  void Forward(const MatType& input, MatType& output) override
  {
    Type stride = imgSize / (Type)(gridSize);
    size_t batchSize = input.n_cols;
    output.set_size(input.n_rows, batchSize);

    CubeType inputCube;
    mlpack::MakeAlias(inputCube, input, grid * numAttributes, predictionsPerCell,
      batchSize);

    CubeType outputCube(grid * numAttributes, predictionsPerCell, batchSize,
      arma::fill::none);

    CubeType reshapedCube;
    mlpack::MakeAlias(reshapedCube, output, numAttributes,
      predictionsPerCell * grid, batchSize);

    MatType offset = arma::regspace(0, this->inputDimensions[0] - 1);
    CubeType xOffset = arma::repcube(offset, this->inputDimensions[0],
      predictionsPerCell, batchSize);
    CubeType yOffset = arma::repcube(arma::vectorise(arma::repmat(offset.t(),
      this->inputDimensions[0], 1)), 1, predictionsPerCell, batchSize);

    const size_t cols = predictionsPerCell - 1;
    // x
    outputCube.tube(0, 0, grid - 1, cols) =
      (xOffset + 1 / (1 + arma::exp(-inputCube.tube(0, 0, grid - 1, cols))))
      * stride;

    // y
    outputCube.tube(grid, 0, grid * 2 - 1, cols) =
      (yOffset + 1 / (1 + arma::exp(-inputCube.tube(grid, 0, grid * 2 - 1, cols))
      )) * stride;

    // w
    outputCube.tube(grid * 2, 0, grid * 3 - 1, cols) =
      arma::repcube(w, 1, 1, batchSize) %
      arma::exp(inputCube.tube(grid * 2, 0, grid * 3 - 1, cols));

    // h
    outputCube.tube(grid * 3, 0, grid * 4 - 1, cols) =
      arma::repcube(h, 1, 1, batchSize) %
      arma::exp(inputCube.tube(grid * 3, 0, grid * 4 - 1, cols));

    // apply logistic sigmoid to objectness and classification logits.
    outputCube.tube(grid * 4, 0, outputCube.n_rows - 1, cols) =
      1 / (1 + arma::exp(-inputCube.tube(grid * 4, 0, inputCube.n_rows - 1, cols)));

    for (size_t i = 0; i < outputCube.n_slices; i++)
    {
      reshapedCube.slice(i) =
        arma::reshape(
          arma::reshape(
            outputCube.slice(i), grid, numAttributes * predictionsPerCell
          ).t(),
          numAttributes, predictionsPerCell * grid
        );
    }
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
  size_t grid; // number of boxes in a grid (13 * 13 or 26 * 26)
  MatType w;
  MatType h;

  size_t predictionsPerCell;
};

template <typename MatType = arma::mat>
class YOLOv3tiny {
 public:
  YOLOv3tiny(size_t imgSize, size_t classes, size_t predictionsPerCell) :
    imgSize(imgSize),
    classes(classes),
    predictionsPerCell(predictionsPerCell)
  {
    // x, y, w, h, objectness score, n classes (e.g. COCO = 80)
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
    size_t convolution15 = Convolution(predictionsPerCell * numAttributes, 1, false);
    size_t detections16 = YOLO(imgSize, 13, { 81, 82, 135, 169, 344, 319 });

    size_t convolution17 = Convolution(128, 1);
    // Upsample for more fine-grained detections.
    size_t upsample18 = model.Add<mlpack::NearestInterpolation<MatType>>(scale);

    // Detection head for smaller objects.
    size_t convolution19 = Convolution(256, 3);
    size_t convolution20 = Convolution(predictionsPerCell * numAttributes, 1, false);
    size_t detections21 = YOLO(imgSize, 26, { 10, 14, 23, 27, 37, 58 });

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
    model.Connect(upsample18, convolution19);
    model.Connect(convolution8, convolution19);
    // Set axis not necessary, since default is channels.

    model.Connect(convolution19, convolution20);
    model.Connect(convolution20, detections21);
    // Again, set axis not necessary, since default is channels.
 
    // Concatenation order shouldn't matter.
    model.Connect(detections16, concatLayer22);
    model.Connect(detections21, concatLayer22);

    model.Reset();
  }

  void Training(const bool training)
  {
    model.SetNetworkMode(training);
  }

  void Predict(const MatType& input, MatType& output)
  {
    CheckImage(mlpack::data::ImageInfo(imgSize, imgSize, 3), input);
    model.Predict(input, output);
  }

  std::string OutputDimensions()
  {
    std::vector<size_t> outputDims = model.OutputDimensions();
    std::ostringstream strDims;
    for (size_t i = 0; i < outputDims.size() - 1; i++)
      strDims << outputDims[i] << ", ";
    strDims << outputDims.back();
    return strDims.str();
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
        "or 1, but you gave " << kernel << ".\n";
      throw std::logic_error(errMessage.str());
    }

    size_t pad = kernel == 3 ? 1 : 0;
    mlpack::MultiLayer<MatType> block;
    block.template Add<mlpack::Convolution<MatType>>(
      maps, kernel,kernel, 1, 1, pad, pad, "none", !batchNorm);

    if (batchNorm)
    {
      // set epsilon to zero, couldn't find it used in darknet/ggml versions.
      block.template Add<mlpack::BatchNorm<MatType>>(2, 2, 0, false);
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

  size_t YOLO(const size_t imgSize, size_t gridSize,
              const std::vector<double>& anchors)
  {
    return model.Add<YOLOv3Layer<MatType>>(imgSize, numAttributes, gridSize,
      predictionsPerCell, anchors);
  }


  size_t imgSize;
  size_t classes;
  size_t predictionsPerCell;
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

  LoadImage(inputFile, info, image);
  LetterBox(info, image, newInfo, newImage);
  SaveImage(outputFile, newInfo, newImage);

  arma::mat detections;
  YOLOv3tiny model(416, 80, 3);
  model.Training(false);
  model.Predict(newImage, detections);

  // detections shape should be (85, 2535)
  std::cout << "Model output shape: " << model.OutputDimensions() << "\n";

  return 0;
}
