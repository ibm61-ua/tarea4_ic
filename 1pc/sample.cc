#include "sample.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

TSample::TSample(const char* pixels, int rows, int cols, int label) : label(label) {
    if (!pixels || rows < 2 || cols < 2)
        throw std::invalid_argument("An image needs at least two rows and columns");
    image.rows = rows / 2;
    image.cols = cols / 2;
    image.pixels.resize(static_cast<std::size_t>(image.rows) * image.cols);
    // Bilinear interpolation at pixel centres, followed by byte rounding.
    // For even dimensions this averages each non-overlapping 2x2 block.
    const auto* bytes = reinterpret_cast<const unsigned char*>(pixels);
    for (int y = 0; y < image.rows; ++y) {
        const double sy = (y + 0.5) * rows / image.rows - 0.5;
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = std::min(y0 + 1, rows - 1);
        const double fy = sy - y0;
        for (int x = 0; x < image.cols; ++x) {
            const double sx = (x + 0.5) * cols / image.cols - 0.5;
            const int x0 = static_cast<int>(std::floor(sx));
            const int x1 = std::min(x0 + 1, cols - 1);
            const double fx = sx - x0;
            const double top = bytes[y0 * cols + x0] * (1-fx) + bytes[y0 * cols + x1] * fx;
            const double bottom = bytes[y1 * cols + x0] * (1-fx) + bytes[y1 * cols + x1] * fx;
            image.pixels[y * image.cols + x] = static_cast<float>(std::floor(top * (1-fy) + bottom * fy + 0.5));
        }
    }
    const auto bounds = std::minmax_element(image.pixels.begin(), image.pixels.end());
    const float minimum = *bounds.first;
    const float range = *bounds.second - minimum;
    for (float& value : image.pixels)
        value = range > 0 ? (value - minimum) / range : 0.0f;
}
