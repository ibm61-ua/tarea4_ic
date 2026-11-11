#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"

#include "timer.h"

#include <cassert>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <fstream>

using namespace knn;
using namespace mnist;
using namespace crossValidation;

// =========================================================================
// Configuración de muestras para depuración rápida o prueba final completa.
// - Puede definirse en tiempo de compilación: -DFAST_DEBUG (1000 muestras)
//   o -DMAX_TRAIN_SAMPLES=N.
// - O modificarse aquí como constante predeterminada.
// - También se puede especificar pasando argumentos al ejecutable: ./cv [num_train] [num_test]
// =========================================================================
#ifndef MAX_TRAIN_SAMPLES
#if defined(FAST_DEBUG)
#define MAX_TRAIN_SAMPLES 1000
#else
#define MAX_TRAIN_SAMPLES 20000 // Tamaño por defecto para el fine tuning según el guion
#endif
#endif

#ifndef MAX_TEST_SAMPLES
#if defined(FAST_DEBUG)
#define MAX_TEST_SAMPLES 200
#else
#define MAX_TEST_SAMPLES 0 // 0 = usar todas las muestras disponibles del conjunto de test
#endif
#endif

const std::size_t DEFAULT_TRAIN_LIMIT = MAX_TRAIN_SAMPLES;
const std::size_t DEFAULT_TEST_LIMIT = MAX_TEST_SAMPLES;

/// <summary>
/// Finds the best K parameter of the KNN by performing an N-Fold Cross Validation for the specified range of K.
/// </summary>
/// <param name="minK">The minimum K parameter to be tested.</param>
/// <param name="maxK">The maximum K parameter to be tested.</param>
/// <param name="dataset">The dataset to be used during the cross validation.</param>
/// <param name="folds">The number of folds.</param>
/// <returns>
/// The value of K that better performs in the cross validation.
/// </returns>
int fineTuning(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds)
{
    assert(minK >= 1 && maxK > minK);
    
    CrossValidation crossValidation = CrossValidation::NFold(dataset, folds);
    std::cout<<"Starting search for best K parameter in range ["<<minK<<", "<<maxK<<"] using "<<crossValidation.getName()<<" with "<<dataset.size()<<" samples."<<std::endl;
    
    unsigned int bestK = minK;
    double bestAccuracy = 0.0;
    for(int k = minK; k <= maxK; k++)
    {
        std::cout<<"Testing k "<<k<<std::endl;
        double start = elapsedTime();
        
        double accuracy = crossValidation.validate(k);
        std::cout<<"K: "<<k<<" Accuracy: "<<accuracy * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
        
        if(accuracy > bestAccuracy)
        {
            bestAccuracy = accuracy;
            bestK = k;
        }
    }
    
    return bestK;
}

namespace {
bool fileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}
}

/// <summary>
/// Loads the MNIST dataset.
/// </summary>
/// <param name="train">The output training data.</param>
/// <param name="test">The output test data.</param>
/// <returns>
/// True if the dataset was loaded successfully, false otherwise.
/// </returns>
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    std::string prefix = "";
    if (fileExists("train/train-images-idx3-ubyte")) {
        prefix = "";
    } else if (fileExists("build/train/train-images-idx3-ubyte")) {
        prefix = "build/";
    } else if (fileExists("../train/train-images-idx3-ubyte")) {
        prefix = "../";
    }

    // Load training data.
    std::string img_path = prefix + "train/train-images-idx3-ubyte";
    std::string label_path = prefix + "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cerr<<"The training data could not be read from "<<img_path<<std::endl;
        return false;
    }
    
    // Load testing data.
    img_path = prefix + "test/t10k-images-idx3-ubyte";
    label_path = prefix + "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cerr<<"The test data could not be read from "<<img_path<<std::endl;
        return false;
    }
    
    return true;
}

/// <summary>
/// The entry point.
/// </summary>
/// <param name="argc">The number of arguments.</param>
/// <param name="argv">The arguments.</param>
int main(int argc, char** argv) 
{
    std::size_t trainLimit = DEFAULT_TRAIN_LIMIT;
    std::size_t testLimit = DEFAULT_TEST_LIMIT;

    if (argc > 1) {
        trainLimit = static_cast<std::size_t>(std::atol(argv[1]));
        std::cout << "Configurado limite de muestras de entrenamiento: " << trainLimit << std::endl;
    }
    if (argc > 2) {
        testLimit = static_cast<std::size_t>(std::atol(argv[2]));
        std::cout << "Configurado limite de muestras de test: " << testLimit << std::endl;
    }

    // Load MNIST dataset.
    double totalStart = elapsedTime();
    std::vector<TSample> train;
    std::vector<TSample> test;
    if(!loadDataset(train, test))
    {
        return 1;
    }
    
    // Aplicar límites de muestras si están configurados
    if (train.size() > trainLimit) {
        train.erase(train.begin() + trainLimit, train.end());
    }
    if (testLimit > 0 && test.size() > testLimit) {
        test.erase(test.begin() + testLimit, test.end());
    }

    std::cout << "Dataset cargado: " << train.size() << " muestras de train, "
              << test.size() << " muestras de test." << std::endl;

    // Fine tune the KNN classifier with training data.
    double start = elapsedTime();
    int k = fineTuning(1, 10, train, 5);
    std::cout<<"Best k: "<<k<<" found in "<<elapsedTime() - start<<" seconds"<<std::endl;
    
    // Test the classifier with testing data.
    start = elapsedTime();
    std::vector<int> labels;
    KNN knn(train, k);
    double accuracy = knn.classifyAndEvaluate(test, labels);
    
    std::cout<<"Test accuracy: "<<accuracy<<" completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
    std::cout<<"Total elapsed time "<<elapsedTime() - totalStart<<" seconds"<<std::endl;
    
    return 0;
}
