/*
 * YOLOv3-tiny, based on the paper below, written with mlpack.
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

struct Image
{
  Image() {}

  Image(const size_t width, const size_t height, const size_t channels)
  {
    info = mlpack::data::ImageInfo(width, height, channels);
    data.set_size(width * height * channels, 1);
  }

  Image(const arma::mat& data, const mlpack::data::ImageInfo& info) :
    data(data), info(info)
  {}

  mlpack::data::ImageInfo info;
  arma::mat data;
};

void CheckImage(const Image& image)
{
  const arma::mat data = image.data;
  const mlpack::data::ImageInfo info = image.info;
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
               const bool grayscale = false)
{
  if (grayscale)
    image.info.Channels() = 1;
  Load(file, image.data, image.info, true);
  image.data /= 255.0f;
  image.data = mlpack::data::ImageLayout(image.data, image.info);

  // Hack so that other image processing functions work.
  if (grayscale)
  {
    image.data = arma::repmat(image.data, 3, 1);
    image.info.Channels() = 3;
  }
}

/*
 *  Saves an image back, reset RGB range back to 0-255 and convert to stb layout
 */
void SaveImage(const std::string& file, Image& image)
{
  CheckImage(image);
  arma::mat stbData = mlpack::data::STBLayout(image.data, image.info);
  stbData *= 255;
  Save(file, stbData, image.info, true);
}

/*
 *  Get object labels from path.
 */
std::vector<std::string> GetLabels(const std::string& path, size_t numClasses)
{
  std::ifstream file(path);
  std::vector<std::string> labels;
  if (!file)
  {
    std::ostringstream errMessage;
    errMessage << "Could not open " << path << ".";
    throw std::logic_error(errMessage.str());
  }

  std::string line;
  while (std::getline(file, line))
    labels.push_back(line);

  if (labels.size() != numClasses)
  {
    std::ostringstream errMessage;
    errMessage << "Expected " << numClasses
               << " classes, but got " << labels.size() << ".";
    throw std::logic_error(errMessage.str());
  }
  return labels;
}

std::string AlphabetKey(char letter, size_t size)
{
  return std::to_string((int)letter) + "_" + std::to_string(size);
}

// There should be 8 sizes per letter.
// each png should start with letter in ascii decimal
// example d size 7: dir/100_7.png
std::unordered_map<char, Image> GetAlphabet(const std::string& dir)
{
  std::unordered_map<char, Image> alphabet;
  // Loops through all printable ascii
  for (char letter = ' '; letter <= '~'; letter++)
  {
    std::string filename = dir + "/" + AlphabetKey(letter, 1) + ".png";
    Image image;
    LoadImage(filename, image, true);
    alphabet.insert({ letter, image });
  }
  return alphabet;
}

/*
 *  Resizes an image using `resizedInfo`.
 */
void ResizeImage(const Image& input, Image& output)
{
  size_t newWidth = output.info.Width();
  size_t newHeight = output.info.Height();

  size_t width = input.info.Width();
  size_t height = input.info.Height();

  CheckImage(input);

  output.data.clear();
  output.data = arma::mat(newWidth * newHeight * output.info.Channels(), 1);

  double xRatio = (double)(width - 1) / (newWidth - 1);
  double yRatio = (double)(height - 1) / (newHeight - 1);

  for (size_t channel = 0; channel < input.info.Channels(); channel++)
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

        double a = input.data.at(xLow + yLow * width + channel * width * height);
        double b = input.data.at(xLow + yHigh * width + channel * width * height);
        double c = input.data.at(xHigh + yLow * width + channel * width * height);
        double d = input.data.at(xHigh + yHigh * width + channel * width * height);

        double value =
                a * (1 - xWeight) * (1 - yWeight) +
                b * xWeight * (1 - yWeight) +
                c * yWeight * (1 - xWeight) +
                d * xWeight * yWeight;

        output.data.at(w + h * newWidth + channel * newWidth * newHeight) =
          value;
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
      if (dx + i >= dst.info.Width())
        break;

      for (size_t j = 0; j < src.info.Height(); j++)
      {
        if (dy + j >= dst.info.Height())
          break;

        size_t sourceIndex = i + j * src.info.Width() +
          c * src.info.Height() * src.info.Width();

        size_t destIndex = (i + dx) + (j + dy) * dst.info.Width() +
          c * dst.info.Height() * dst.info.Width();
        dst.data.at(destIndex) = src.data.at(sourceIndex);
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
void LetterBox(const Image& src, Image& dst, const double grayValue = 0.5)
{
  CheckImage(src);
  dst.data.clear();
  dst.data = arma::mat(dst.info.Width() * dst.info.Height() * dst.info.Channels(), 1);

  size_t width, height;
  if (dst.info.Width() / src.info.Width() > dst.info.Height() / src.info.Height())
  {
    height = dst.info.Height();
    width = src.info.Width() * dst.info.Height() / src.info.Height();
  }
  else
  {
    width = dst.info.Width();
    height = src.info.Height() * dst.info.Width() / src.info.Width();
  }

  dst.data.fill(grayValue);
  arma::mat resizedSrc;
  mlpack::data::ImageInfo resizedInfo(width, height, src.info.Channels());

  Image resizedImage(resizedSrc, resizedInfo);
  ResizeImage(src, resizedImage);
  EmbedImage(resizedImage, dst, (dst.info.Width() - width)/2,
    (dst.info.Height() - height)/2);
}

class BoundingBox
{
 public:
  // Expects format: cx, cy, w, h
  BoundingBox(const double cx, const double cy, const double w, const double h,
              const arma::mat& classProbs)
  {
    x1 = cx - w / 2.0;
    x2 = cx + w / 2.0;
    y1 = cy - h / 2.0;
    y2 = cy + h / 2.0;

    // TODO: get colors
    red = 0.98;
    green = 0.90;
    blue = 0.15;

    objectIndex = classProbs.index_max();
    objectProb = classProbs.at(objectIndex);
  }

  void Draw(Image& image,
            const size_t borderSize,
            const std::vector<std::string>& labels,
            const std::unordered_map<char, Image> &alphabet,
            const double letterSize)
  {
    const mlpack::data::ImageInfo& info = image.info;
    double x1 = std::clamp<double>(this->x1, 0, info.Width() - 1);
    double x2 = std::clamp<double>(this->x2, 0, info.Width() - 1);
    double y1 = std::clamp<double>(this->y1, 0, info.Height() - 1);
    double y2 = std::clamp<double>(this->y2, 0, info.Height() - 1);

    if (x1 > x2 || y1 > y2)
      throw std::logic_error("Bounding box has a bad shape.");

    // Assumes image is layed out planar, i.e r, r, ... g, g, ... b, b
    for (int b = 0; b < borderSize; b++)
    {
      for (int x = x1; x <= x2; x++)
      {
        // Top
        int yTop = y1 - b;
        // Bottom
        int yBot = y2 + b;

        int rTop = x + yTop * info.Width();
        int gTop = x + yTop * info.Width() + info.Height() * info.Width();
        int bTop = x + yTop * info.Width() + info.Height() * info.Width() * 2;
        image.data(rTop, 0) = red;
        image.data(gTop, 0) = green;
        image.data(bTop, 0) = blue;

        int rBot = x + yBot * info.Width();
        int gBot = x + yBot * info.Width() + info.Height() * info.Width();
        int bBot = x + yBot * info.Width() + info.Height() * info.Width() * 2;
        image.data(rBot, 0) = red;
        image.data(gBot, 0) = green;
        image.data(bBot, 0) = blue;
      }

      for (int y = y1; y <= y2; y++)
      {
        // Left
        int xL = x1 + b;
        // Right
        int xR = x2 - b;

        int rL = xL + y * info.Width();
        int gL = xL + y * info.Width() + info.Height() * info.Width();
        int bL = xL + y * info.Width() + info.Height() * info.Width() * 2;
        image.data(rL, 0) = red;
        image.data(gL, 0) = green;
        image.data(bL, 0) = blue;

        int rR = xR + y * info.Width();
        int gR = xR + y * info.Width() + info.Height() * info.Width();
        int bR = xR + y * info.Width() + info.Height() * info.Width() * 2;
        image.data(rR, 0) = red;
        image.data(gR, 0) = green;
        image.data(bR, 0) = blue;
      }
    }
    std::cout << labels[objectIndex] << ": " << objectProb * 100 << "%\n";
    DrawLabel(image, labels[objectIndex], letterSize, alphabet);
  }

  void DrawLabel(Image& image,
                 const std::string& label,
                 const double size,
                 const std::unordered_map<char, Image>& alphabet)
  {
    double dx = x1;
    for (size_t i = 0; i < label.size(); i++)
    {
      char letter = label[i];
      Image letterImage = alphabet.at(letter);
      Image resized;
      resized.info = mlpack::data::ImageInfo(letterImage.info.Width() * size,
        letterImage.info.Height() * size, 3);

      ResizeImage(letterImage, resized);
      EmbedImage(resized, image, dx, y1);
      dx += resized.info.Width();
      if (dx > image.info.Width())
        break;
    }
  }

 private:
  double x1;
  double y1;
  double x2;
  double y2;
  double red;
  double green;
  double blue;
  size_t objectIndex;
  double objectProb;
};

/*
 * Draw boxes onto image, only if the boxes objectness score is > `maxProb`.
 */
void DrawBoxes(const arma::mat& modelOutput,
               const size_t numBoxes,
               const double maxProb,
               const double borderSize,
               const size_t imgSize,
               const double letterSize,
               const std::vector<std::string>& labels,
               const std::unordered_map<char, Image>& alphabet,
               Image& image)
{
  CheckImage(image);
  if (modelOutput.n_cols != 1)
  {
    std::ostringstream errMessage;
    errMessage << "modelOutput should have 1 column, but you gave "
               << modelOutput.n_cols << " columns.";
    throw std::logic_error(errMessage.str());
  }

  const size_t rem = modelOutput.n_rows % numBoxes;
  if (rem != 0)
  {
    std::ostringstream errMessage;
    errMessage << "modelOutput rows should be divisible by numBoxes, but "
               << modelOutput.n_rows << " % " << numBoxes << " == " << rem;
    throw std::logic_error(errMessage.str());
  }

  double xRatio = (double)image.info.Width() / imgSize;
  double yRatio = (double)image.info.Height() / imgSize;

  const size_t predictionSize = modelOutput.n_rows / numBoxes;
  for (size_t box = 0; box < numBoxes; box++)
  {
    arma::mat prediction;
    mlpack::MakeAlias(prediction, modelOutput, predictionSize, 1,
      box * predictionSize);
    if (prediction.at(4, 0) < maxProb)
      continue;

    double x, y, w, h;
    x = prediction.at(0, 0) * xRatio;
    y = prediction.at(1, 0) * yRatio;
    w = prediction.at(2, 0) * xRatio;
    h = prediction.at(3, 0) * yRatio;
    const arma::mat& classProbs =
      prediction.submat(5, 0, prediction.n_rows - 1, 0);

    BoundingBox bbox(x, y, w, h, classProbs);
    bbox.Draw(image, borderSize, labels, alphabet, letterSize);
  }
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

    MatType offset = arma::regspace<MatType>(0, this->inputDimensions[0] - 1);
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

template <typename MatType = arma::fmat>
class YOLOv3tiny {
 public:
  YOLOv3tiny(size_t imgSize, size_t classes, size_t predictionsPerCell,
             const std::string& weightsFile) :
    imgSize(imgSize),
    classes(classes),
    predictionsPerCell(predictionsPerCell)
  {
    // x, y, w, h, objectness score, n classes (e.g. COCO = 80)
    numAttributes = 5 + classes;
    scale = { 2.0, 2.0 };

    model = mlpack::DAGNetwork<
      mlpack::EmptyLoss,
      mlpack::RandomInitialization,
      MatType
    >();

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
    size_t upsample18 = model.template Add<mlpack::NearestInterpolation<MatType>>(scale);

    // Detection head for smaller objects.
    size_t convolution19 = Convolution(256, 3);
    size_t convolution20 = Convolution(predictionsPerCell * numAttributes, 1, false);
    size_t detections21 = YOLO(imgSize, 26, { 10, 14, 23, 27, 37, 58 });

    // the DAGNetwork class requires a layer for concatenations, so we use
    // the Identity layer for pure concatentation, and no other compute.
    size_t concatLayer22 = model.template Add<mlpack::Identity<MatType>>();

    layers = {
      convolution0,
      convolution2,
      convolution4,
      convolution6,
      convolution8,
      convolution10,
      convolution12,
      convolution13,
      convolution14,
      convolution15,
      convolution17,
      convolution19,
      convolution20
    };

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

    LoadWeights(weightsFile);
  }

  void Training(const bool training)
  {
    model.SetNetworkMode(training);
  }

  void Predict(const MatType& input, MatType& output)
  {
    Image image;
    image.data = input;
    image.info = mlpack::data::ImageInfo(imgSize, imgSize, 3);
    CheckImage(image);
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

  size_t YOLO(const size_t imgSize, const size_t gridSize,
              const std::vector<typename MatType::elem_type>& anchors)
  {
    return model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      gridSize, predictionsPerCell, anchors);
  }

  size_t LoadConvolution(std::ifstream& f,
                         const size_t layer,
                         const size_t inChannels,
                         const size_t outChannels,
                         const size_t kernelSize,
                         const bool output = false,
                         const bool batchNorm = true)
  {
    size_t weightsSize = kernelSize * kernelSize * inChannels * outChannels;

    std::vector<float> biases(outChannels);
    std::vector<float> scales(outChannels);
    std::vector<float> rollingMeans(outChannels);
    std::vector<float> rollingVars(outChannels);
    std::vector<float> weights(weightsSize);
    f.read(reinterpret_cast<char *>(biases.data()), outChannels * sizeof(float));
    if (batchNorm)
    {
      f.read(reinterpret_cast<char *>(scales.data()), outChannels * sizeof(float));
      f.read(reinterpret_cast<char *>(rollingMeans.data()), outChannels * sizeof(float));
      f.read(reinterpret_cast<char *>(rollingVars.data()), outChannels * sizeof(float));
    }
    f.read(reinterpret_cast<char *>(weights.data()), weightsSize * sizeof(float));

    if (f.eof()) throw std::runtime_error("Reached end of weights.");
    if (f.fail()) throw std::runtime_error("Reading weights failed.");
    if (f.bad()) throw std::runtime_error("Stream corrupted");

    if (output) {
      std::cout << "-------------------------------------\n";
      std::cout << "Biases: " << biases[3] << "\n";
      if (batchNorm) {
        std::cout << "Scales: " << scales[3] << "\n";
        std::cout << "Rolling mean: " << rollingMeans[3] << "\n";
        std::cout << "Rolling var: " << rollingVars[3] << "\n";
      }
      std::cout << "Convolution: " << weights[3] << "\n";
    }

    return (batchNorm ? outChannels * 4 : outChannels) + weightsSize;
  }

  /*
   * Load weights from the darknet .weights format.
   *
   * XXX: Only works for yolov3-tiny config, from
   * https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
   *
   */
  void LoadWeights(const std::string& file)
  {
    parameters.clear();
    parameters.set_size(model.WeightSize());

    std::ifstream weightsFile(file, std::ios::binary);
    if (!weightsFile)
      throw std::runtime_error("Could not open " + file);

    // Skip header.
    weightsFile.seekg(20, std::ios::cur);

    size_t total = 0;
    total += LoadConvolution(weightsFile, layers[0], 3, 16, 3, true);
    total += LoadConvolution(weightsFile, layers[1], 16, 32, 3, true);
    total += LoadConvolution(weightsFile, layers[2], 32, 64, 3, true);
    total += LoadConvolution(weightsFile, layers[3], 64, 128, 3, true);
    total += LoadConvolution(weightsFile, layers[4], 128, 256, 3, true);
    total += LoadConvolution(weightsFile, layers[5], 256, 512, 3, true);
    total += LoadConvolution(weightsFile, layers[6], 512, 1024, 3, true);
    total += LoadConvolution(weightsFile, layers[7], 1024, 256, 1, true);
    total += LoadConvolution(weightsFile, layers[8], 256, 512, 3, true);
    total += LoadConvolution(weightsFile, layers[9], 512, 255, 1, true, false);
    total += LoadConvolution(weightsFile, layers[10], 256, 128, 1, true);
    total += LoadConvolution(weightsFile, layers[11], 384, 256, 3, true);
    total += LoadConvolution(weightsFile, layers[12], 256, 255, 1, true, false);

    model.Parameters() = parameters;
    std::cout << "Total Weights: " << total << "\n";
    weightsFile.close();
  }

  size_t imgSize;
  size_t classes;
  size_t predictionsPerCell;
  size_t numAttributes;
  std::vector<double> scale;
  std::vector<size_t> layers;

  mlpack::DAGNetwork<mlpack::EmptyLoss, mlpack::RandomInitialization, MatType>
    model;

  MatType parameters;
};

int main(int argc, const char** argv) {
  // Settings
  const size_t numClasses = 80;
  const size_t imgSize = 416;
  const size_t imgChannels = 3;
  const size_t predictionsPerCell = 3;
  const size_t numBoxes = 13 * 13 * 3 + 26 * 26 * 3;
  const double ignoreProb = 0.6;
  const size_t borderSize = 1;
  const double letterSize = 0.5;
  const std::string lettersDir = "../data/labels";
  const std::string labelsFile = "../data/coco.names";
  const std::string weightsFile = "../weights/yolov3-tiny.weights";
  // const std::vector<double> anchors =
  //   { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };

  if (argc != 3)
    throw std::logic_error("usage: ./main <input_image> <output_image>");

  const std::string inputFile = argv[1];
  const std::string outputFile = argv[2];

  const std::unordered_map<char, Image> alphabet = GetAlphabet(lettersDir);
  const std::vector<std::string> labels = GetLabels(labelsFile, numClasses);

  Image image;
  Image input(imgSize, imgSize, imgChannels);
  arma::mat detections;

  LoadImage(inputFile, image);
  LetterBox(image, input);

  YOLOv3tiny model(imgSize, numClasses, predictionsPerCell, weightsFile);
  model.Training(false);
  // model.Predict(input.data, detections);
  //
  // std::cout << "Model output shape: " << model.OutputDimensions() << "\n";
  //
  // DrawBoxes(detections,
  //           numBoxes,
  //           ignoreProb,
  //           borderSize,
  //           imgSize,
  //           letterSize,
  //           labels,
  //           alphabet,
  //           image);
  //
  // std::cout << "Saving to " << outputFile << ".\n";
  // SaveImage(outputFile, image);
  return 0;
}
