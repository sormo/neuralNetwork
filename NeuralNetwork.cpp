#include "NeuralNetwork.h"
#include "Matrix.h"
#include <vector>
#include <algorithm>

namespace NeuralNetwork
{
	struct Layer
	{
		Matrix weights;
		Matrix values;
		Matrix biases;

		Matrix weightedInput;
	};

	struct Network
	{
		std::vector<Layer> layers;
	};

	struct Error
	{
		Matrix weights;
		Matrix biases;
	};

	struct NetworkError
	{
		std::vector<Error> layers;
	};

	Matrix ActivationFunction(const Matrix & values)
	{
		Matrix ret(values);

		for (size_t i = 0; i < values.GetNumberOfRows(); ++i)
			for (size_t j = 0; j < values.GetNumberOfColumns(); ++j)
				ret(i, j) = 1.0 / (1.0 + exp(-ret(i, j)));

		return ret;
	}

	Matrix ActivationFunctionDerivative(const Matrix & values)
	{
		Matrix ret(ActivationFunction(values));

		for (size_t i = 0; i < values.GetNumberOfRows(); ++i)
			for (size_t j = 0; j < values.GetNumberOfColumns(); ++j)
			{
				double v = ret(i, j);
				ret(i, j) = v * (1.0 - v);
			}


		return ret;
	}

	double Random(double min, double max)
	{
		double f = (double)rand() / RAND_MAX;
		return min + f * (max - min);
	}

	std::shared_ptr<Network> CreateNeuralNetwork(const std::vector<size_t> & topology,
		std::pair<double, double> weightInitRange, std::pair<double, double> biasInitRange)
	{
		std::shared_ptr<Network> ret(new Network);

		for (size_t i = 0; i < topology.size(); ++i)
		{
			size_t currentSize = topology[i];
			size_t previousSize = i == 0 ? 1 : topology[i - 1];

			Layer layer
			{
				Matrix(currentSize, previousSize), // weights
				Matrix(currentSize, 1),            // values 
				Matrix(currentSize, 1),            // biases
				Matrix(currentSize, 1),            // weighted input
			};

			// initialization of weights and biases
			for (size_t i = 0; i < currentSize; ++i)
			{
				for (size_t j = 0; j < previousSize; ++j)
					layer.weights(i, j) = Random(weightInitRange.first, weightInitRange.second);
				layer.biases(i, 0) = Random(biasInitRange.first, biasInitRange.second);
			}

			ret->layers.push_back(std::move(layer));
		}

		return ret;
	}

	NetworkError CreateNetworkError(const Network & network)
	{
		NetworkError ret;
		for (size_t i = 0; i < network.layers.size(); ++i)
		{
			const Layer & layer = network.layers[i];
			ret.layers.push_back({ Matrix(layer.weights.GetShape()), Matrix(layer.biases.GetShape()) });
		}
		return ret;
	}

	void PropagateForward(Network & network, const std::vector<double> & input)
	{
		if (input.size() != network.layers[0].values.GetNumberOfRows())
			throw std::runtime_error("invalid number of input values");

		// set values to input layer
		for (size_t i = 0; i < input.size(); ++i)
		{
			network.layers[0].values(i, 0) = input[i];
		}

		// propagate forward
		for (size_t i = 1; i < network.layers.size(); ++i)
		{
			Layer & currentLayer = network.layers[i];
			Layer & previousLayer = network.layers[i - 1];

			currentLayer.weightedInput = currentLayer.weights * previousLayer.values + currentLayer.biases;
			currentLayer.values = ActivationFunction(currentLayer.weightedInput);
		}
	}

	std::vector<double> FeedForward(Network & network, const std::vector<double> & input)
	{
		PropagateForward(network, input);

		std::vector<double> ret;
		for (size_t i = 0; i < network.layers.back().values.GetNumberOfRows(); ++i)
			ret.push_back(network.layers.back().values(i, 0));

		return ret;
	}

	void PropagateBackward(Network & network, const std::vector<double> & output, NetworkError & error)
	{
		if (output.size() != network.layers.back().values.GetNumberOfRows())
			throw std::runtime_error("invalid number of output values");

		// output layer error
		Layer & outputLayer = network.layers.back();
		Layer & previousLayer = network.layers[network.layers.size() - 2];

		Matrix delta = Matrix(outputLayer.values - Matrix(output)).hadamardProduct(
			ActivationFunctionDerivative(outputLayer.weightedInput));

		error.layers.back().biases += delta;
		error.layers.back().weights += delta * previousLayer.values.transpose();

		// propagate error backward
		// error is accumulating
		for (size_t i = network.layers.size() - 2; i > 0; --i)
		{
			Layer & layer = network.layers[i];
			Layer & nextLayer = network.layers[i + 1];
			Layer & previousLayer = network.layers[i - 1];

			delta = (nextLayer.weights.transpose() * delta).hadamardProduct(
				ActivationFunctionDerivative(layer.weightedInput));

			error.layers[i].biases += delta;
			error.layers[i].weights += delta * previousLayer.values.transpose();
		}
	}

	void UpdateWeights(Network & network, size_t numberOfExamples, double learningRate, NetworkError & error)
	{
		// update weights
		for (size_t i = network.layers.size() - 1; i > 0; --i)
		{
			Layer & layer = network.layers[i];
			Layer & previousLayer = network.layers[i - 1];

			layer.weights -= error.layers[i].weights * learningRate / (double)numberOfExamples;
			layer.biases -= error.layers[i].biases * learningRate / (double)numberOfExamples;

			error.layers[i].weights.Reset(0.0);
			error.layers[i].biases.Reset(0.0);
		}
	}

	void StochasticGradientDescent(Network & network,
		const std::vector<std::pair<std::vector<double>, std::vector<double>>> & trainingData,
		size_t numberOfEpochs, size_t batchSize, double learningRate,
		std::function<void()> epochCallback)
	{
		NetworkError error = CreateNetworkError(network);

		// prepare input data
		std::vector<std::pair<std::vector<double>, std::vector<double>>> data;
		for (const auto & d : trainingData)
			data.push_back({ d.first, d.second });

		for (size_t epoch = 0; epoch < numberOfEpochs; ++epoch)
		{
			// shuffle data for each epoch
			std::random_shuffle(std::begin(data), std::end(data));
			auto dataIt = std::begin(data);

			while (dataIt != std::end(data))
			{
				size_t currentEpochSize = 0;
				for (currentEpochSize; currentEpochSize < batchSize; ++currentEpochSize)
				{
					if (dataIt == std::end(data))
						break;

					PropagateForward(network, dataIt->first);
					PropagateBackward(network, dataIt->second, error);

					dataIt++;
				}

				UpdateWeights(network, currentEpochSize, learningRate, error);
			}

			if (epochCallback)
				epochCallback();
		}
	}
}
