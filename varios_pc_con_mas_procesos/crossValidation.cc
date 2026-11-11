#include "crossValidation.h"
#include "sample.h"
#include "knn.h"

#include "timer.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>
#include <sstream>

using namespace knn;
using namespace crossValidation;

/// <summary>
/// Performs an N-Fold Cross Validation with the specified KNN parameter.
/// </summary>
/// <param name="k">The K parameter of the KNN classifier.</param>
/// <returns>
/// A value in the range [0.0, 1.0] indicating the averaged accuracy of the KNN in the N folds.
/// </returns>
double CrossValidation::validate(unsigned int k) const
{
    double accuracy = 0.0;
    std::cout<<"Starting "<<this->getName()<<std::endl;
    for(int fold=0;fold<this->folds;fold++)
    {
        std::vector<TSample> train, test;
        this->splitDataset(train, test, fold);
     
        std::cout<<"Testing fold "<<fold<<std::endl;
        double start = elapsedTime();
        KNN knn(train, k);
        std::vector<int> labels;
        double acc = knn.classifyAndEvaluate(test, labels);
        accuracy += acc;
        std::cout<<"Fold: "<<fold<<" Accuracy: "<<acc * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
    }
    
    return accuracy /= this->folds;
}

/// <summary>
/// Splits the dataset into train and validation subsets for an N-Fold Cross Validation.
/// </summary>
/// <param name="train">The output training data.</param>
/// <param name="validation">The output validation data.</param>
/// <param name="fold">The fold to be used as validation data.</param>
void CrossValidation::splitDataset(std::vector<TSample> &train, std::vector<TSample> &validation, unsigned int fold) const
{
    assert(fold < this->folds && "Invalid fold");

    int start = this->dataset.size() * fold / this->folds;
    int end = this->dataset.size() * (fold + 1) / this->folds;

    for(int i=0;i<this->dataset.size();i++)
    {
        if(i<start || i>=end)
            train.push_back(this->dataset[i]);
        else
            validation.push_back(this->dataset[i]);
    }
}

/// <summary>
/// Gets the name of the Cross Validation, either N-Fold or leave-one-out.
/// </summary>
std::string CrossValidation::getName() const
{
    std::stringstream ss("");
    if(this->folds < this->dataset.size())
        ss<<folds<<"-Folds Cross Validation";
    else
        ss<<"leave-one-out Cross Validation";
        
    return ss.str();
}
