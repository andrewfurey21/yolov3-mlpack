#ifndef LABELS_IMPL_HPP
#define LABELS_IMPL_HPP

#include "labels.hpp"

/*
 *  Get object labels from path.
 */
inline std::vector<std::string> GetLabels(const std::string& path, size_t numClasses)
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

#endif
