#include "yolov3.hpp"

template <typename MatType = arma::fmat>
class YOLOv3 {
 public:
  YOLOv3(size_t imgSize, size_t classes, size_t predictionsPerCell,
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

    size_t convolution0 = Convolution(32, 3);

    size_t layer4 = Downsample(convolution0, 64, 1);
    size_t layer11 = Downsample(layer4, 128, 2);
    size_t layer36 = Downsample(layer11, 256, 8);
    size_t layer61 = Downsample(layer36, 512, 8);
    size_t layer74 = Downsample(layer61, 1024, 4);

    size_t layer75 = Convolution(512, 1);
    size_t layer76 = Convolution(1024, 3);
    size_t layer77 = Convolution(512, 1);

    size_t sppConvolution = Convolution(512, 1);
    SpatialPyramidPooling(layer77, sppConvolution);

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

    model.Connect(sppConvolution, layer78);

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
    size_t layer1 = MaxPool(13);
    size_t layer2 = MaxPool(9);
    size_t layer3 = MaxPool(5);

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
    return model.template Add<mlpack::YOLOv3Layer<MatType>>(imgSize,
      numAttributes, gridSize, predictionsPerCell, anchors);
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

    assert(layers.size() == 76);

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
};

int main(int argc, const char** argv) {
  // Settings
  const size_t numClasses = 80; // coco
  const size_t imgSize = 320;
  const size_t imgChannels = 3;
  const size_t predictionsPerCell = 3;
  // const size_t numBoxes = 22743; // 608
  // const size_t numBoxes = 10647; // 416
  const size_t numBoxes = 6300; // 320
  const double ignoreProb = 0.5;
  const size_t borderSize = 4;
  const double letterSize = 1.5;
  const std::string lettersDir = "../data/labels";
  const std::string labelsFile = "../data/coco.names";

  const std::string weightsFile = "../weights/yolov3-spp.weights";


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
    (imgSize, numClasses, predictionsPerCell, weightsFile);

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
