# yolov3 in mlpack

<div align="center">
  <img src="./output.jpg" width="400" alt="">
</div>

Implementation of the yolov3 family of models in mlpack, including yolov3, yolov3-tiny and yolov3-spp.

## Requirements

mlpack master, armadillo 15

## Example

You can download the weights for running the yolov3 example in `example.cpp` [here](https://drive.google.com/drive/folders/1BiACM5LxcD1m3wkutQ8GtVesWXfSCdeK). You can compile and run it with:

```bash
g++ example.cpp -O3 -o detect -larmadillo -fopenmp

# yolov3 (~60 million parameters)
./detect yolov3-320.bin dog.jpg output.jpg

```
