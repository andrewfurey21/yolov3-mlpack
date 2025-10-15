#ifndef LABELS_HPP
#define LABELS_HPP

#include "image.hpp"

/*
 *  Get object labels from path.
 */
std::vector<std::string> GetLabels(const std::string& path, size_t numClasses);

std::string AlphabetKey(char letter, size_t size);

// There should be 8 sizes per letter.
// each png should start with letter in ascii decimal
// example d size 7: dir/100_7.png
std::unordered_map<char, Image> GetAlphabet(const std::string& dir);

#include "labels_impl.hpp"

#endif
