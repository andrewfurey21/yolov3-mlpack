#ifndef BOUNDINGBOX_HPP
#define BOUNDINGBOX_HPP

#include "image.hpp"
#include "font8x8_basic.h"

void DrawLetter(Image& image, const unsigned char letter, const size_t x, const size_t y, const size_t s);

class BoundingBox
{
 public:
  BoundingBox(const double x1,
              const double y1,
              const double x2,
              const double y2,
              const size_t objectClass,
              const double objectProb,
              const size_t numClasses) :
    x1(x1), y1(y1), x2(x2), y2(y2),
    objectClass(objectClass),
    objectProb(objectProb),
    numClasses(numClasses)
  {
    Color();
  }

  void Color()
  {
    float colors[6][3] =
      { {1,0,1}, {0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,0} };

    float ratio = ((float)objectClass / numClasses) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;

    red = (1 - ratio) * colors[i][0] + ratio * colors[j][0];
    green = (1 - ratio) * colors[i][1] + ratio * colors[j][1];
    blue = (1 - ratio) * colors[i][2] + ratio * colors[j][2];
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

    if (objectProb == 0)
      return;

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
    std::ostringstream newLabel;
    newLabel << label << ": " << std::to_string((int)(roundf(objectProb * 100)))
      << "%";

    double x1 = std::clamp<double>(this->x1, 0, image.info.Width() - 1);
    double y1 = std::clamp<double>(this->y1, 0, image.info.Height() - 1);

    double dx = x1;
    const std::string& actualLabel = newLabel.str();
    const size_t letterSize = 3;
    for (size_t i = 0; i < actualLabel.size(); i++)
    {
      // NOTE: Use this if using ggml's data/labels images.
      // 
      // char letter = actualLabel[i];
      // Image letterImage = alphabet.at(letter);
      // Image resized(letterImage.info.Width() * size, letterImage.info.Height() * size, 3);
      // ResizeImage(letterImage, resized);
      // EmbedImage(resized, image, dx, y1);

      if (dx + letterSize * 8 > image.info.Width())
        break;

      const unsigned char letter = actualLabel[i];
      DrawLetter(image, letter, dx, y1, letterSize);
      dx += letterSize * 8;
    }
  }

  bool operator> (const BoundingBox& other) const
  {
    return objectProb > other.objectProb;
  }

  double x1;
  double y1;
  double x2;
  double y2;
  double objectProb;
  size_t objectClass;

 private:
  double red;
  double green;
  double blue;
  size_t numClasses;
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
               Image& image);

/*
 * Do non max suppression to get rid of extra boxes.
 */

inline void NonMaxSuppression(std::vector<BoundingBox>& bboxes,
                              const double threshold = 0.45);

double Intersection(const BoundingBox& a, const BoundingBox& b);

double Union(const BoundingBox& a, const BoundingBox& b);


#include "boundingbox_impl.hpp"

#endif
