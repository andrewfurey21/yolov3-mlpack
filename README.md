# yolov3 in mlpack

<div align="center">
  <img src="./readme.png" width="400" alt="">
</div>

Implementation of the yolov3 family of models in mlpack, including yolov3, yolov3-tiny and yolov3-spp.

## Requirements

mlpack master, armadillo 15

## Example

You can download the weights [here](https://drive.google.com/drive/folders/1BiACM5LxcD1m3wkutQ8GtVesWXfSCdeK). You can compile and run it with:

```bash
g++ example.cpp -O3 -o yolov3-inference -larmadillo -fopenmp

# yolov3 (~60 million parameters)
./yolov3-inference yolov3-320.bin ./data/coco.names dog.jpg output.jpg

# yolov3-tiny (~8 million parameters)
./yolov3-inference yolov3-tiny.bin ./data/coco.names dog.jpg output.jpg

```

Because of how mlpack serializes models, you can run `yolov3` or `yolov3-tiny` without changing anything other than the weights you pass in.
