#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <filesystem>
void check(bool condition) { if (!condition) throw std::runtime_error("Test failed"); }
int main() {
    char pixels[16] = {0,0,10,10,0,0,10,10,20,20,30,30,20,20,30,30};
    TSample sample(pixels,4,4,7);
    check(sample.image.rows == 2 && sample.image.cols == 2);
    check(sample.image.pixels[0] == 0 && sample.image.pixels[3] == 1);
    check(std::abs(sample.image.pixels[1]-1.0f/3) < 1e-6);
    char flat[16]; for(char& x : flat) x = static_cast<char>(255);
    TSample constant(flat,4,4,2);
    check(constant.image.pixels[0] == 0);
    char rect[24] = {}; TSample rectangle(rect,4,6,0);
    check(rectangle.image.rows == 2 && rectangle.image.cols == 3);
    knn::KNN classifier({constant,sample},1);
    check(classifier.classify(sample) == 7);
    check(classifier.classify(constant) == 2);
    bool rejected=false;
    try { knn::KNN({sample},2).classify(sample); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    const auto dir = std::filesystem::temp_directory_path()/"knn2-idx-test";
    std::filesystem::create_directories(dir);
    auto write = [](std::ofstream& f, unsigned int n) {
        char b[4] = {char(n>>24),char(n>>16),char(n>>8),char(n)}; f.write(b,4);
    };
    { std::ofstream f(dir/"images",std::ios::binary); write(f,2051); write(f,1); write(f,4); write(f,4); f.write(pixels,16); }
    { std::ofstream f(dir/"labels",std::ios::binary); write(f,2049); write(f,1); f.put(7); }
    std::vector<TSample> loaded;
    check(mnist::loadMnist((dir/"images").string().c_str(),(dir/"labels").string().c_str(),loaded));
    check(loaded.size()==1 && loaded[0].label==7 && loaded[0].image.pixels == sample.image.pixels);
    std::filesystem::resize_file(dir/"images",18);
    check(!mnist::loadMnist((dir/"images").string().c_str(),(dir/"labels").string().c_str(),loaded));
    check(loaded.size()==1);
    std::filesystem::remove_all(dir);
    crossValidation::CrossValidation cv({sample,sample,sample,sample,sample},2);
    check(cv.validate(1) == 1);
}
