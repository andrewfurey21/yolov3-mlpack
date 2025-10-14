/*
 * YOLOv3, based on the paper below, written with mlpack.
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
 * TODO: add link to cfg
 *
 */

// #define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
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
 *  TODO: fix pngs
 */
void LoadImage(const std::string& file,
               Image& image,
               const bool grayscale = false)
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
 *
 *  XXX: This is the same resizing used as darknet. If you load
 *  in the pretrained weights and use a different resizing method
 *  you will get worse results.
 *
 *  TODO: try mlpack resize
 */
void ResizeImage(const Image& input, Image& output)
{
  CheckImage(input);
  CheckImage(output);
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
 *  The original yolov3-tiny model trained by Redmon et al. used this method
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

class BoundingBox
{
 public:
  // Expects format: cx, cy, w, h
  BoundingBox(const double cx, const double cy, const double w, const double h,
              const size_t objectClass) : objectClass(objectClass)
  {
    x1 = cx - w / 2.0;
    x2 = cx + w / 2.0;
    y1 = cy - h / 2.0;
    y2 = cy + h / 2.0;

    // TODO: get colors
    red = 0.98;
    green = 0.90;
    blue = 0.15;
  }

  void Draw(Image& image,
            const size_t borderSize,
            const std::vector<std::string>& labels,
            const std::unordered_map<char, Image> &alphabet,
            const double letterSize)
  {
    const mlpack::data::ImageInfo& info = image.info;

    double x1 = std::clamp<double>(this->x1, 0, info.Width() - 1);
    double y1 = std::clamp<double>(this->y1, 0, info.Height() - 1);
    double x2 = std::clamp<double>(this->x2, 0, info.Width() - 1);
    double y2 = std::clamp<double>(this->y2, 0, info.Height() - 1);

    if (x1 > x2 || y1 > y2)
      throw std::logic_error("Bounding box has a bad shape.");

    // Assumes image is layed out planar, i.e r, r, ... g, g, ... b, b
    for (int b = 0; b < borderSize; b++)
    {
      for (int x = x1; x <= x2; x++)
      {
        // Top
        int yT = y1 + b;
        // Bottom
        int yB = y2 - b;

        image.SetPixel(x, yT, 0, red);
        image.SetPixel(x, yT, 1, green);
        image.SetPixel(x, yT, 2, blue);

        image.SetPixel(x, yB, 0, red);
        image.SetPixel(x, yB, 1, green);
        image.SetPixel(x, yB, 2, blue);
      }

      for (int y = y1; y <= y2; y++)
      {
        // Left
        int xL = x1 + b;
        // Right
        int xR = x2 - b;

        image.SetPixel(xL, y, 0, red);
        image.SetPixel(xL, y, 1, green);
        image.SetPixel(xL, y, 2, blue);

        image.SetPixel(xR, y, 0, red);
        image.SetPixel(xR, y, 1, green);
        image.SetPixel(xR, y, 2, blue);
      }
    }
    DrawLabel(image, labels[objectClass], letterSize, alphabet);
  }

  void DrawLabel(Image& image,
                 const std::string& label,
                 const double size,
                 const std::unordered_map<char, Image>& alphabet)
  {
    double x1 = std::clamp<double>(this->x1, 0, image.info.Width() - 1);
    double y1 = std::clamp<double>(this->y1, 0, image.info.Height() - 1);

    double dx = x1;
    for (size_t i = 0; i < label.size(); i++)
    {
      char letter = label[i];
      Image letterImage = alphabet.at(letter);
      Image resized(letterImage.info.Width() * size, letterImage.info.Height() * size, 3);

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
  size_t objectClass;
};

/*
 * Draw boxes onto image, only if the boxes objectness score is > `ignoreProb`.
 */
void DrawBoxes(const arma::fmat& modelOutput,
               const size_t numBoxes,
               const double ignoreProb,
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
    arma::fmat prediction;
    mlpack::MakeAlias(prediction, modelOutput, predictionSize, 1,
      box * predictionSize);
    float objectness = prediction.at(4, 0);
    if (objectness < ignoreProb)
    {
      // std::cout << objectness << ", ";
      continue;
    }
    double x, y, w, h;
    x = prediction.at(0, 0) * xRatio;
    y = prediction.at(1, 0) * yRatio;
    w = prediction.at(2, 0) * xRatio;
    h = prediction.at(3, 0) * yRatio;
    const arma::fmat& classProbs =
      prediction.submat(5, 0, prediction.n_rows - 1, 0);
    const size_t classIndex = classProbs.index_max();
    const float objectProb = objectness * classProbs.at(classIndex);

    if (objectProb < ignoreProb)
    {
      // std::cout << objectProb << ", ";
      continue;
    }
    std::cout << labels[classIndex] << ": " << roundf(objectProb * 100) << "%\n";
    BoundingBox bbox(x, y, w, h, classIndex);
    bbox.Draw(image, borderSize, labels, alphabet, letterSize);
  }
}

/*
 *  Return how much a vertical or horizontal line overlap, where `a` and `b` are
 *  mid-points and `aw` and `bw` are widths/heights of those lines.
 */
double LineOverlap(double a, double aw, double b, double bw) {
  return std::abs(std::max(a - aw / 2, b - bw / 2) -
                  std::min(a + aw / 2, b + bw / 2));
}

template <typename MatType = arma::fmat>
class YOLOv3Layer : public mlpack::Layer<MatType>
{
 public:
  YOLOv3Layer() {}
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
    {
      throw std::logic_error("YOLOv3Layer::ComputeOutputDimensions(): "
        "Grid is the wrong size.");
    }

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
      arma::fill::zeros);

    CubeType reshapedCube;
    mlpack::MakeAlias(reshapedCube, output, numAttributes,
      predictionsPerCell * grid, batchSize);

    const size_t cols = predictionsPerCell - 1;
    MatType offset = arma::regspace<MatType>(0, this->inputDimensions[0] - 1);
    CubeType xOffset = arma::repcube(offset, this->inputDimensions[0],
      predictionsPerCell, batchSize);
    CubeType yOffset = arma::repcube(arma::vectorise(arma::repmat(offset.t(),
      this->inputDimensions[0], 1)), 1, predictionsPerCell, batchSize);
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
      1. / (1. + arma::exp(-inputCube.tube(grid * 4, 0, inputCube.n_rows - 1, cols)));

    // Reshape, for each batch item.
    for (size_t i = 0; i < reshapedCube.n_slices; i++)
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
    throw std::runtime_error("YOLOv3::Backward() not implemented.");
  }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<mlpack::Layer<MatType>>(this));
    ar(CEREAL_NVP(imgSize));
    ar(CEREAL_NVP(numAttributes));
    ar(CEREAL_NVP(gridSize));
    ar(CEREAL_NVP(grid));
    ar(CEREAL_NVP(w));
    ar(CEREAL_NVP(h));
    ar(CEREAL_NVP(predictionsPerCell));
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

// CEREAL_REGISTER_TYPE(mlpack::Layer<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::Identity<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::MultiLayer<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::Convolution<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::BatchNorm<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::LeakyReLU<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::Padding<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::MaxPooling<arma::fmat>)
// CEREAL_REGISTER_TYPE(mlpack::NearestInterpolation<arma::fmat>)
// CEREAL_REGISTER_TYPE(YOLOv3Layer<arma::fmat>)

template <typename MatType = arma::fmat>
class YOLOv3 {
 public:
  YOLOv3(size_t imgSize, size_t classes, size_t predictionsPerCell,
             const std::string& weightsFile, bool spp = false) :
    imgSize(imgSize),
    classes(classes),
    predictionsPerCell(predictionsPerCell),
    spp(spp)
  {
    // x, y, w, h, objectness score, n classes (e.g. COCO = 80)
    numAttributes = 5 + classes;
    scale = { 2.0, 2.0 };

    model = Model();
    model.InputDimensions() = { imgSize, imgSize, 3 };

    size_t convolution0 = Convolution(32, 3);

    size_t layer4 = Downsample(convolution0, 64, 1);
    size_t layer11 = Downsample(layer4, 128, 2);
    size_t layer36 = Downsample(layer11, 256, 8);
    size_t layer61 = Downsample(layer36, 512, 8);
    size_t layer74 = Downsample(layer61, 1024, 4);

    size_t layer75 = Convolution(512, 1);
    size_t layer76 = Convolution(1024, 3);
    size_t layer77 = Convolution(512, 1);

    size_t layer1 = MaxPool(13);
    size_t layer2 = MaxPool(9);
    size_t layer3 = MaxPool(5);
    size_t sppConvolution = Convolution(512, 1);

    size_t layer78 = Convolution(1024, 3);
    size_t layer79 = Convolution(512, 1);
    size_t layer80 = Convolution(1024, 3);
    size_t layer81 = Convolution(255, 1, 1, false); // coco
    // size_t detection0 = YOLO(imgSize, 19, {116, 90, 156, 198, 373, 326}); // 608
    // size_t detection0 = YOLO(imgSize, 13, {116, 90, 156, 198, 373, 326}); // 416
    size_t detection0 = YOLO(imgSize, 10, {116, 90, 156, 198, 373, 326}); // 320

    model.Connect(layer74, layer75);
    model.Connect(layer75, layer76);
    model.Connect(layer76, layer77);

    // if (!spp)
    //   model.Connect(layer77, layer78);

    model.Connect(layer77, layer1);
    model.Connect(layer77, layer2);
    model.Connect(layer77, layer3);
    model.Connect(layer1, sppConvolution);
    model.Connect(layer2, sppConvolution);
    model.Connect(layer3, sppConvolution);
    model.Connect(layer77, sppConvolution);

    std::cout << "Order: " << layer1 << ", " << layer2 << ", " << layer3 << ", " << layer77 << "\n";
    model.Connect(sppConvolution, layer78);

    //

    model.Connect(layer78, layer79);
    model.Connect(layer79, layer80);
    model.Connect(layer80, layer81);
    model.Connect(layer81, detection0);

    size_t layer82 = Convolution(256, 1);
    size_t upsample82 = model.template Add<mlpack::NearestInterpolation<MatType>>(scale);
    model.Connect(layer79, layer82);
    model.Connect(layer82, upsample82);

    size_t layer84 = Convolution(256, 1);
    size_t layer85 = Convolution(512, 3);
    size_t layer86 = Convolution(256, 1);
    size_t layer87 = Convolution(512, 3);
    size_t layer88 = Convolution(256, 1);
    size_t layer89 = Convolution(512, 3);
    size_t layer90 = Convolution(255, 1, 1, false); // coco
    // size_t detection1 = YOLO(imgSize, 38, {30, 61, 62, 45, 59, 119}); // 608
    // size_t detection1 = YOLO(imgSize, 26, {30, 61, 62, 45, 59, 119}); // 416
    size_t detection1 = YOLO(imgSize, 20, {30, 61, 62, 45, 59, 119}); // 320

    // Concat
    model.Connect(upsample82, layer84);
    model.Connect(layer61, layer84); // default is concat along last axis.

    model.Connect(layer84, layer85);
    model.Connect(layer85, layer86);
    model.Connect(layer86, layer87);
    model.Connect(layer87, layer88);
    model.Connect(layer88, layer89);
    model.Connect(layer89, layer90);
    model.Connect(layer90, detection1);

    size_t layer91 = Convolution(128, 1);
    size_t upsample91 = model.template Add<mlpack::NearestInterpolation<MatType>>(scale);
    model.Connect(layer88, layer91);
    model.Connect(layer91, upsample91);

    size_t layer93 = Convolution(128, 1);
    size_t layer94 = Convolution(256, 3);
    size_t layer95 = Convolution(128, 1);
    size_t layer96 = Convolution(256, 3);
    size_t layer97 = Convolution(128, 1);
    size_t layer98 = Convolution(256, 3);
    size_t layer99 = Convolution(255, 1, 1, false); // coco
    // size_t detection2 = YOLO(imgSize, 76, {10, 13, 16, 30, 33, 23}); // 608
    // size_t detection2 = YOLO(imgSize, 52, {10, 13, 16, 30, 33, 23}); // 416
    size_t detection2 = YOLO(imgSize, 40, {10, 13, 16, 30, 33, 23}); // 320

    // Concat
    model.Connect(upsample91, layer93);
    model.Connect(layer36, layer93); // default is concat along last axis.

    model.Connect(layer93, layer94);
    model.Connect(layer94, layer95);
    model.Connect(layer95, layer96);
    model.Connect(layer96, layer97);
    model.Connect(layer97, layer98);
    model.Connect(layer98, layer99);
    model.Connect(layer99, detection2);

    // do spp logic here, because loading darknet weights would be annoying
    // move when using cereal
    // if (spp)
    // {
    //   size_t sppConvolution = Convolution(512, 1);
    //   SpatialPyramidPooling(layer77, sppConvolution);
    //   model.Connect(sppConvolution, layer78);
    // }

    // Concat outputs.
    size_t lastLayer = model.template Add<mlpack::Identity>();
    model.Connect(detection0, lastLayer);
    model.Connect(detection1, lastLayer);
    model.Connect(detection2, lastLayer);

    model.Reset();

    std::cout << "Weight size: " << model.WeightSize() << "\n";
    LoadWeights(weightsFile);
  }

  ~YOLOv3() {}

  using Model =
    mlpack::DAGNetwork<mlpack::EmptyLoss,
      mlpack::RandomInitialization, MatType>;

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

  size_t Downsample(const size_t previousLayer, const size_t maps,
                    const size_t shortcuts = 1)
  {
    assert(shortcuts >= 1);

    size_t convolution0 = Convolution(maps, 3, 2); // stride = 2
    model.Connect(previousLayer, convolution0);

    size_t previous = convolution0;
    for (size_t i = 0; i < shortcuts; i++)
    {
      size_t convolution1 = Convolution(maps / 2, 1);
      size_t convolution2 = Convolution(maps, 3);

      size_t residual = model.template Add<mlpack::Identity>();
      model.SetConnection(residual, ADDITION);

      model.Connect(previous, convolution1);
      model.Connect(convolution1, convolution2);
      model.Connect(convolution2, residual);
      model.Connect(previous, residual);

      previous = residual;
    }
    return previous;
  }

  size_t MaxPool(const size_t kernel)
  {
    const size_t pad = kernel / 2;
    mlpack::MultiLayer<MatType> block;
    Type min = -arma::datum::inf;
    block.template Add<mlpack::Padding<MatType>>(pad, pad, pad, pad, min);
    block.template Add<mlpack::MaxPooling<MatType>>(kernel, kernel, 1, 1);
    return model.Add(block);
  }

  void SpatialPyramidPooling(const size_t input, const size_t output)
  {
    size_t layer1 = MaxPool(5);
    size_t layer2 = MaxPool(9);
    size_t layer3 = MaxPool(13);

    model.Connect(input, layer1);
    model.Connect(input, layer2);
    model.Connect(input, layer3);

    model.Connect(layer1, output);
    model.Connect(layer2, output);
    model.Connect(layer3, output);
    model.Connect(input, output);
  }

  size_t Convolution(const size_t maps, const size_t kernel, const size_t stride = 1,
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
      maps, kernel, kernel, stride, stride, pad, pad, "none", !batchNorm);

    if (batchNorm)
    {
      // set epsilon to zero, couldn't find it used in darknet/ggml versions.
      block.template Add<mlpack::BatchNorm<MatType>>(2, 2, 0, false);
    }
    block.template Add<mlpack::LeakyReLU<MatType>>(reluSlope);

    size_t layer = model.Add(block);
    layers.push_back(layer);
    return layer;
  }

  size_t YOLO(const size_t imgSize, const size_t gridSize,
              const std::vector<typename MatType::elem_type>& anchors)
  {
    return model.template Add<YOLOv3Layer<MatType>>(imgSize, numAttributes,
      gridSize, predictionsPerCell, anchors);
  }

  using CubeType = typename GetCubeType<MatType>::type;

  size_t LoadConvolution(std::ifstream& f,
                         const size_t layer,
                         const size_t inChannels,
                         const size_t outChannels,
                         const size_t kernelSize,
                         const size_t offset,
                         const bool batchNorm = true)
  {
    size_t weightsSize = kernelSize * kernelSize * inChannels * outChannels;
    size_t total = (outChannels * batchNorm) + outChannels + weightsSize;

    // Should just use armadillo instead.
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

    MatType layerParams;
    mlpack::MakeAlias<MatType>(layerParams, parameters, total, 1, offset);

    // All layers must be MultiLayer, otherwise undefined behaviour.
    mlpack::MultiLayer<MatType>* layerPtr =
      static_cast<mlpack::MultiLayer<MatType>*>(model.Network()[layer]);

    if (layerPtr->WeightSize() != total)
    {
      std::ostringstream errMessage;
      errMessage << "Layer weight size ( " << layerPtr->WeightSize()
                 <<  " ) is not the same as total weight size ( "
                 << total << " ).";
      throw std::logic_error(errMessage.str());
    }

    std::vector<float> totalWeights;
    totalWeights.reserve(total);
    if(batchNorm)
    {
      // Layer[1] should be a BatchNorm layer.
      mlpack::BatchNorm<MatType>* bn =
        static_cast<mlpack::BatchNorm<MatType>*>(layerPtr->Network()[1]);
      bn->TrainingMean() = MatType(rollingMeans);
      bn->TrainingVariance() = MatType(rollingVars);

      totalWeights.insert(totalWeights.end(), weights.begin(), weights.end());
      totalWeights.insert(totalWeights.end(), scales.begin(), scales.end());
      totalWeights.insert(totalWeights.end(), biases.begin(), biases.end());
    }
    else
    {
      totalWeights.insert(totalWeights.end(), weights.begin(), weights.end());
      totalWeights.insert(totalWeights.end(), biases.begin(), biases.end());
    }

    layerParams = MatType(totalWeights);
    return total;
  }

  /*
   * Load weights from the darknet .weights format.
   *
   * XXX: Only works for yolov3 config, from
   * https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   *
   */
  void LoadWeights(const std::string& file)
  {
    parameters.clear();
    parameters.set_size(model.WeightSize(), 1);

    std::ifstream weightsFile(file, std::ios::binary);
    if (!weightsFile)
      throw std::runtime_error("Could not open " + file);

    // Skip header.
    weightsFile.seekg(20, std::ios::cur);

    if (this->spp)
      assert(layers.size() == 76);
    else
      assert(layers.size() == 75);

    size_t total = 0;
    total += LoadConvolution(weightsFile, layers[0], 3, 32, 3, total);

    total += LoadConvolution(weightsFile, layers[1], 32, 64, 3, total);
    total += LoadConvolution(weightsFile, layers[2], 64, 32, 1, total);
    total += LoadConvolution(weightsFile, layers[3], 32, 64, 3, total);

    total += LoadConvolution(weightsFile, layers[4], 64, 128, 3, total);
    total += LoadConvolution(weightsFile, layers[5], 128, 64, 1, total);
    total += LoadConvolution(weightsFile, layers[6], 64, 128, 3, total);
    total += LoadConvolution(weightsFile, layers[7], 128, 64, 1, total);
    total += LoadConvolution(weightsFile, layers[8], 64, 128, 3, total);

    total += LoadConvolution(weightsFile, layers[9], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[10], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[11], 128, 256, 3, total); //layer 14
    total += LoadConvolution(weightsFile, layers[12], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[13], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[14], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[15], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[16], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[17], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[18], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[19], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[20], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[21], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[22], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[23], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[24], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[25], 128, 256, 3, total);

    total += LoadConvolution(weightsFile, layers[26], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[27], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[28], 256, 512, 3, total); //layer39

    total += LoadConvolution(weightsFile, layers[29], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[30], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[31], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[32], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[33], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[34], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[35], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[36], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[37], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[38], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[39], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[40], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[41], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[42], 256, 512, 3, total);

    total += LoadConvolution(weightsFile, layers[43], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[44], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[45], 512, 1024, 3, total); //layer64
    total += LoadConvolution(weightsFile, layers[46], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[47], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[48], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[49], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[50], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[51], 512, 1024, 3, total);

    total += LoadConvolution(weightsFile, layers[52], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[53], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[54], 1024, 512, 1, total);

    // if (spp)
    // {
    //   std::cout << "Layer: " << layers[75] << "\n";
    // }

    // total += LoadConvolution(weightsFile, layers[55], 512, 1024, 3, total);
    // total += LoadConvolution(weightsFile, layers[56], 1024, 512, 1, total);
    // total += LoadConvolution(weightsFile, layers[57], 512, 1024, 3, total);
    // total += LoadConvolution(weightsFile, layers[58], 1024, 255, 1, total, false); // coco
    //
    // total += LoadConvolution(weightsFile, layers[59], 512, 256, 1, total);
    // total += LoadConvolution(weightsFile, layers[60], 768, 256, 1, total);
    // total += LoadConvolution(weightsFile, layers[61], 256, 512, 3, total);
    // total += LoadConvolution(weightsFile, layers[62], 512, 256, 1, total);
    // total += LoadConvolution(weightsFile, layers[63], 256, 512, 3, total);
    // total += LoadConvolution(weightsFile, layers[64], 512, 256, 1, total);
    // total += LoadConvolution(weightsFile, layers[65], 256, 512, 3, total);
    // total += LoadConvolution(weightsFile, layers[66], 512, 255, 1, total, false); // coco
    //
    // total += LoadConvolution(weightsFile, layers[67], 256, 128, 1, total);
    // total += LoadConvolution(weightsFile, layers[68], 384, 128, 1, total);
    // total += LoadConvolution(weightsFile, layers[69], 128, 256, 3, total);
    // total += LoadConvolution(weightsFile, layers[70], 256, 128, 1, total);
    // total += LoadConvolution(weightsFile, layers[71], 128, 256, 3, total);
    // total += LoadConvolution(weightsFile, layers[72], 256, 128, 1, total);
    // total += LoadConvolution(weightsFile, layers[73], 128, 256, 3, total);
    // total += LoadConvolution(weightsFile, layers[74], 256, 255, 1, total, false); // coco

    total += LoadConvolution(weightsFile, layers[55], 2048, 512, 1, total);

    total += LoadConvolution(weightsFile, layers[56], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[57], 1024, 512, 1, total);
    total += LoadConvolution(weightsFile, layers[58], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[59], 1024, 255, 1, total, false); // coco

    total += LoadConvolution(weightsFile, layers[60], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[61], 768, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[62], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[63], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[64], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[65], 512, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[66], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[67], 512, 255, 1, total, false); // coco

    total += LoadConvolution(weightsFile, layers[68], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[69], 384, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[70], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[71], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[72], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[73], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[74], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[75], 256, 255, 1, total, false); // coco

    model.Parameters() = parameters;
    std::cout << "Total Weights (excluding rolling means/variances): "
              << total << "\n";
    weightsFile.close();
  }

  size_t imgSize;
  size_t classes;
  size_t predictionsPerCell;
  size_t numAttributes;
  std::vector<double> scale;
  Model model;
  std::vector<size_t> layers;
  MatType parameters;
  bool spp;
};

int main(int argc, const char** argv) {
  // Settings
  const size_t numClasses = 80; // coco
  const size_t imgSize = 320;
  const size_t imgChannels = 3;
  const size_t predictionsPerCell = 3;
  const bool spp = true;
  // const size_t numBoxes = 22743; // 608
  // const size_t numBoxes = 10647; // 416
  const size_t numBoxes = 6300; // 320
  const double ignoreProb = 0.5;
  const size_t borderSize = 4;
  const double letterSize = 1.5;
  const std::string lettersDir = "../data/labels";
  const std::string labelsFile = "../data/coco.names";

  std::string weightsFile = "../weights/yolov3-608.weights";
  if (spp)
  {
    weightsFile = "../weights/yolov3-spp.weights";
    std::cout << "Using spatial pyramid pooling\n";
  }


  if (argc != 3)
    throw std::logic_error("usage: ./main <input_image> <output_image>");

  const std::string inputFile = argv[1];
  const std::string outputFile = argv[2];

  const std::unordered_map<char, Image> alphabet = GetAlphabet(lettersDir);
  const std::vector<std::string> labels = GetLabels(labelsFile, numClasses);

  Image image;
  Image input(imgSize, imgSize, imgChannels);
  arma::fmat detections;

  LoadImage(inputFile, image);
  LetterBox(image, input);

  YOLOv3<arma::fmat> model
    (imgSize, numClasses, predictionsPerCell, weightsFile, spp);

  model.Training(false);

  std::cout << "Thinking...\n";
  model.Predict(input.data, detections);

  std::cout << "Model output shape: " << model.OutputDimensions() << "\n";

  DrawBoxes(detections,
            numBoxes,
            ignoreProb,
            borderSize,
            imgSize,
            letterSize,
            labels,
            alphabet,
            image);

  std::cout << "Saving to " << outputFile << ".\n";
  SaveImage(outputFile, image);

  // std::cout << detections.t() << "\n";
  return 0;
}
