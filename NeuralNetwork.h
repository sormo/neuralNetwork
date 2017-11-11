#pragma once
#include <memory>
#include <vector>
#include <functional>

namespace NeuralNetwork
{
	struct Network;

	std::shared_ptr<Network> CreateNeuralNetwork(const std::vector<size_t> & topology, 
		std::pair<double, double> weightInitRange, std::pair<double, double> biasInitRange);

	void StochasticGradientDescent(Network & network, 
		const std::vector<std::pair<std::vector<double>, std::vector<double>>> & trainingData,
		size_t numberOfEpochs, size_t batchSize, double learningRate,
		std::function<void()> epochCallback);

	std::vector<double> FeedForward(Network & network, const std::vector<double> & input);
};
