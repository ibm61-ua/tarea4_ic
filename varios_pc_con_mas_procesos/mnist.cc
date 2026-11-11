#include "mnist.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <limits>

namespace {
bool readBigEndian(std::istream& input, std::uint32_t& value) {
    unsigned char bytes[4];
    if (!input.read(reinterpret_cast<char*>(bytes), 4)) return false;
    value = (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16)
          | (std::uint32_t(bytes[2]) << 8) | bytes[3];
    return true;
}
}
bool mnist::loadMnist(const char* imageFilename, const char* labelFilename, std::vector<TSample>& dataset) {
    std::ifstream images(imageFilename, std::ios::binary), labels(labelFilename, std::ios::binary);
    std::uint32_t imageMagic, labelMagic, count, labelCount, rows, cols;
    if (!readBigEndian(images, imageMagic) || !readBigEndian(labels, labelMagic)
        || !readBigEndian(images, count) || !readBigEndian(labels, labelCount)
        || !readBigEndian(images, rows) || !readBigEndian(images, cols)
        || imageMagic != 2051 || labelMagic != 2049 || count != labelCount
        || rows < 2 || cols < 2 || rows > std::numeric_limits<int>::max()
        || cols > std::numeric_limits<int>::max()
        || std::uint64_t(rows) * cols > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid or unreadable MNIST headers\n";
        return false;
    }
    std::vector<TSample> loaded;
    std::vector<char> pixels(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t i = 0; i < count; ++i) {
        unsigned char label;
        if (!labels.read(reinterpret_cast<char*>(&label), 1)
            || !images.read(pixels.data(), pixels.size())) {
            std::cerr << "Truncated MNIST data\n";
            return false;
        }
        loaded.emplace_back(pixels.data(), rows, cols, label);
    }
    dataset.insert(dataset.end(), loaded.begin(), loaded.end());
    return true;
}
