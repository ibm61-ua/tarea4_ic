#ifndef KNN2_SAMPLE_H
#define KNN2_SAMPLE_H
#include <vector>

// Row-major image with owned, normalized floating-point pixels.
struct Image {
    int rows = 0;
    int cols = 0;
    std::vector<float> pixels;
};
struct TSample {
    TSample(const char* pixels, int rows, int cols, int label);
    Image image;
    int label;
};
#endif
