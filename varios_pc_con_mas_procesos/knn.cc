#include "knn.h"
#include "mnist.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace knn;

/// <summary>
/// Computes the euclidean distance between two samples.
/// </summary>
/// <param name="a">The first sample.</param>
/// <param name="b">The second sample.</param>
/// <returns>
/// The distance between the two samples.
/// </returns>
float euclidean(const TSample &a, const TSample &b)
{
    if (a.image.rows != b.image.rows || a.image.cols != b.image.cols)
        throw std::invalid_argument("Image dimensions differ");
    float sum = 0.0f;
    const float *imageA = a.image.pixels.data();
    const float *imageB = b.image.pixels.data();

    for(int i=0;i<a.image.rows*a.image.cols;i++)
    {
    	float diff = imageA[i] - imageB[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

/// <summary>
/// Struct that represents the result of comparing an input sample with one of the traininig samples.
/// </summary>
typedef struct KNNResult
{
    /// <summary>
    /// Creates a new instance of the KNNResult.
    /// </summary>
    KNNResult()
    {
        this->label = 0;
        this->distance = 0.0;
    };
    
    /// <summary>
    /// Creates a new instance of the KNNResult.
    /// </summary>
    /// <param name="label">The label of the training sample.</param>
    /// <param name="distance">The distance between the training sample and the test sample.</param>
    KNNResult(int label, float distance)
    {
        this->label = label;
        this->distance = distance;
    };
    
    /// <summary>
    /// Classifies the specified sample.
    /// </summary>
    /// <param name="testSample">The test sample.</param>
    /// <param name="trainingSample">The training sample.</param>
    KNNResult(const TSample &testSample, const TSample &trainingSample)
    {
        this->label = trainingSample.label;
        this->distance = euclidean(testSample, trainingSample);
    };
    
    /// <summary>
    /// Determines whether this instance is closer than another instance.
    /// </summary>
    /// <param name="b">The other result to compare.</param>
    /// <returns>
    /// True if this instance is closer than the other instance.
    /// </returns>
    bool operator<(const KNNResult &b) const
    {
        return this->distance < b.distance;
    }
    
    /// <summary>
    /// The label of the training sample.
    /// </summary>
    int label;
    
    /// <summary>
    /// The distance between the test sample and the training sample.
    /// </summary>
    float distance;
}KNNResult;

/// <summary>
/// Computes an histogram of labels of the specified k results.
/// </summary>
/// <param name="results">The results of the KNN ordered by distance.</param>
/// <returns>
/// A map with the label as key and the frequency as value.
/// </returns>
std::map<int, int> histogram(const std::vector<KNNResult> &results, int k)
{
    std::map<int, int> votes;
    for(int i=0;i<k;i++)
    {
        votes[results[i].label]++;
    }
    
    return votes;
}

/// <summary>
/// Calculates the mode label of the specified histogram.
/// </summary>
/// <param name="histogram">A map representing an histogram of label and their frequency.</param>
/// <returns>
/// The most frequent label.
/// </returns>
int maxVotesLabel(std::map<int, int> histogram)
{
    int maxVotes = 0;
    int maxLabel = 0;
    std::map<int, int>::iterator itr;
    for (itr = histogram.begin(); itr != histogram.end(); ++itr) {
        if(itr->second > maxVotes) {
            maxVotes = itr->second;
            maxLabel = itr->first;
        }
    }
    
    return maxLabel;
}

/// <summary>
/// Classifies the specified sample.
/// </summary>
/// <param name="test">The test sample to be classified.</param>
/// <returns>
/// An integer that represents the most common label of the k nearest neighbors of the test samples.
/// </returns>
int KNN::classify(const TSample &test) const
{
    if (k < 1 || static_cast<std::size_t>(k) > train.size())
        throw std::invalid_argument("k must be between 1 and the training size");
    std::vector<KNNResult> results;
    for(int i=0;i<this->train.size();i++)
    {
        results.push_back(KNNResult(test, this->train[i]));
    }
    
    std::sort(results.begin(), results.end());
    
    return maxVotesLabel(histogram(results, this->k));
}

/// <summary>
/// Classifies and evaluates performance with the specified samples.
/// </summary>
/// <param name="test">The test samples to be classified.</param>
/// <param name="labels">Output parameter with a vector of integers that represents the most common label of the k nearest neighbors for each of the test samples.</param>
/// <returns>
/// The accuracy of the classification.
/// </returns>
double KNN::classifyAndEvaluate(const std::vector<TSample> &test, std::vector<int> &labels) const
{
    if (test.empty()) throw std::invalid_argument("Test set is empty");
    int matches = 0;
    for(int i=0;i<test.size();i++)
    {
        std::cout<<i<<"/"<<test.size()<<"\r"<<std::flush;
        int label = this->classify(test[i]);
        labels.push_back(label);
        
        if(label == test[i].label) matches++;
    }
    
    return matches / (double)test.size();
}

