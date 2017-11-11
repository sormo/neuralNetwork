#include "NeuralNetwork.h"
#include <iostream>
#include <iomanip>

#define NUMBER_OF_EPOCHS 50
#define BATCH_SIZE 30
#define LEARNING_RATE 30.0
#define NETWORK_TOPOLOGY { 2, 2, 1 }
#define WEIGHT_RANGE { 0.3, 0.7 }
#define BIAS_RANGE { 0.3, 0.7 }

void TestXor()
{
	std::cout << "Test XOR" << std::endl;

	std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingExamples;
	for (size_t i = 0; i < 1000; ++i)
	{
		int a = rand() % 2;
		int b = rand() % 2;
		int c = a ^ b;

		trainingExamples.push_back({ {(double)a, (double)b}, {(double)c} });
	}

	std::vector<std::pair<std::vector<double>, int>> testExamples;
	for (size_t i = 0; i < 100; ++i)
	{
		int a = rand() % 2;
		int b = rand() % 2;
		int c = a ^ b;

		testExamples.push_back({ { (double)a, (double)b }, c });
	}

	auto network = NeuralNetwork::CreateNeuralNetwork(NETWORK_TOPOLOGY, WEIGHT_RANGE, BIAS_RANGE);

	size_t epochCount = 1;
	NeuralNetwork::StochasticGradientDescent(*network, trainingExamples, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE,
		[&testExamples, &network, &epochCount]
	{
		double error = 0.0;
		size_t errorCount = 0;
		for (const auto & test : testExamples)
		{
			auto data = NeuralNetwork::FeedForward(*network, test.first);
			errorCount += (size_t)std::round(data[0]) == test.second ? 0 : 1;
			error += std::fabs(data[0] - (double)test.second);
		}

		std::cout << "Epoch " << epochCount++ << " " << std::fixed << std::setprecision(2);
		std::cout << 100.0*(1.0 - (double)errorCount / (double)testExamples.size()) << "%";
		std::cout << " (" << testExamples.size() - errorCount << " / " << testExamples.size() << ")";
		std::cout << " error " << std::fixed << std::setprecision(3) << error << std::endl;
	});
}
