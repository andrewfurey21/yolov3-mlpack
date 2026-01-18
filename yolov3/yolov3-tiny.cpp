
#include "yolov3.hpp"
#include <mlpack/core/data/image_layout.hpp>

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

    model = Model();
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
    // Set axis not necessary, since default is concat along channels.

    model.Connect(convolution19, convolution20);
    model.Connect(convolution20, detections21);
    // Again, set axis not necessary, since default is concat along channels.

    // Concatenation order shouldn't matter.
    model.Connect(detections16, concatLayer22);
    model.Connect(detections21, concatLayer22);

    model.Reset();
    LoadWeights(weightsFile);
  }

  ~YOLOv3tiny() {}

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
      maps, kernel, kernel, 1, 1, pad, pad, "none", !batchNorm);

    // set epsilon to zero, couldn't find it used in darknet/ggml versions.
    if (batchNorm)
      block.template Add<mlpack::BatchNorm<MatType>>(2, 2, 0, false);

    block.template Add<mlpack::LeakyReLU<MatType>>(reluSlope);
    size_t layer = model.Add(block);
    layers.push_back(layer);
    return layer;
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
    return model.template Add<mlpack::YOLOv3Layer<MatType>>(imgSize,
      numAttributes, gridSize, predictionsPerCell, anchors);
  }

  using CubeType = typename GetCubeType<MatType>::type;
  // Kind of naive implementation, could probably be improved, maybe use
  // armadillo instead of std::vector for loading.
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

    if (layerPtr->WeightSize() != total)
    {
      std::ostringstream errMessage;
      errMessage << "Layer weight size ( " << layerPtr->WeightSize()
                 <<  " ) is not the same as total weight size ( "
                 << total << " ).";
      throw std::logic_error(errMessage.str());
    }

    layerParams = MatType(totalWeights);
    return total;
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
    parameters.set_size(model.WeightSize(), 1);

    std::ifstream weightsFile(file, std::ios::binary);
    if (!weightsFile)
      throw std::runtime_error("Could not open " + file);

    // Skip header.
    weightsFile.seekg(20, std::ios::cur);

    size_t total = 0;
    total += LoadConvolution(weightsFile, layers[0], 3, 16, 3, total);
    total += LoadConvolution(weightsFile, layers[1], 16, 32, 3, total);
    total += LoadConvolution(weightsFile, layers[2], 32, 64, 3, total);
    total += LoadConvolution(weightsFile, layers[3], 64, 128, 3, total);
    total += LoadConvolution(weightsFile, layers[4], 128, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[5], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[6], 512, 1024, 3, total);
    total += LoadConvolution(weightsFile, layers[7], 1024, 256, 1, total);
    total += LoadConvolution(weightsFile, layers[8], 256, 512, 3, total);
    total += LoadConvolution(weightsFile, layers[9], 512, 255, 1, total, false);
    total += LoadConvolution(weightsFile, layers[10], 256, 128, 1, total);
    total += LoadConvolution(weightsFile, layers[11], 384, 256, 3, total);
    total += LoadConvolution(weightsFile, layers[12], 256, 255, 1, total, false);

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
};

int main(int argc, const char** argv) {
  // Settings
  const size_t numClasses = 80;
  const size_t imgSize = 416;
  const size_t imgChannels = 3;
  const size_t predictionsPerCell = 3;
  const size_t numBoxes = 13 * 13 * 3 + 26 * 26 * 3;
  const double ignoreProb = 0.5;
  const size_t borderSize = 10;
  const double letterSize = 1.5;
  const std::string lettersDir = "./data/labels";
  const std::string labelsFile = "./data/coco.names";
  const std::string weightsFile = "./weights/darknet/yolov3-tiny.weights";
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
  arma::fmat detections;

  LoadImage(inputFile, image);
  LetterBox(image, input);

  YOLOv3tiny<arma::fmat> model
    (imgSize, numClasses, predictionsPerCell, weightsFile);

  model.Training(false);
  model.Predict(input.data, detections);

  std::cout << "Model output shape: " << model.OutputDimensions() << "\n";

  // Image output = Image(mlpack::data::InterleaveChannels(input.data, input.info), input.info);
  Image output = Image(mlpack::data::InterleaveChannels(image.data, image.info), image.info);
  DrawBoxes(detections,
            numBoxes,
            ignoreProb,
            borderSize,
            imgSize,
            letterSize,
            labels,
            alphabet,
            output);

  std::cout << "Saving to " << outputFile << ".\n";
  SaveImage(outputFile, output);
  return 0;
}
