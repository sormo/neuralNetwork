#include <fstream>
#include <cassert>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "NeuralNetwork.h"

#define TRAIN_IMAGE_FILE "mnist\\train-images.idx3-ubyte"
#define TRAIN_LABEL_FILE "mnist\\train-labels.idx1-ubyte"
#define TEST_IMAGE_FILE "mnist\\t10k-images.idx3-ubyte"
#define TEST_LABEL_FILE "mnist\\t10k-labels.idx1-ubyte"
#define NUMBER_OF_OUTPUT_DIGITS 10

#define NUMBER_OF_EPOCHS 30
#define BATCH_SIZE 20
#define LEARNING_RATE 5.0
#define NETWORK_TOPOLOGY { 2, 2, 1 }
#define WEIGHT_RANGE { -1.0, 1.0 }
#define BIAS_RANGE { -1.0, 1.0 }

#pragma pack(1)
struct MnistImageHeader
{
	uint32_t magic;
	uint32_t numberOfImages;
	uint32_t numberOfRows;
	uint32_t numberOfColumns;
};

struct MnistLabelHeader
{
	uint32_t magic;
	uint32_t numberOfLabels;
};
#pragma pack()

uint32_t swap32(uint32_t k)
{
	return ((k << 24) |
		((k & 0x0000FF00) << 8) |
		((k & 0x00FF0000) >> 8) |
		(k >> 24)
		);
}

MnistImageHeader ReadImageHeader(std::ifstream & file)
{
	MnistImageHeader header;
	file.read((char*)&header, sizeof(header));

	header.magic = swap32(header.magic);
	header.numberOfColumns = swap32(header.numberOfColumns);
	header.numberOfRows = swap32(header.numberOfRows);
	header.numberOfImages = swap32(header.numberOfImages);

	return header;
}

MnistImageHeader ReadImageHeader(const char * fileName)
{
	std::ifstream file(fileName, std::ios_base::binary);
	assert(file.is_open());

	return ReadImageHeader(file);
}

MnistLabelHeader ReadLabelsHeader(std::ifstream & file)
{
	MnistLabelHeader header;
	file.read((char*)&header, sizeof(header));

	header.magic = swap32(header.magic);
	header.numberOfLabels = swap32(header.numberOfLabels);

	return header;
}

MnistLabelHeader ReadLabelsHeader(const char * fileName)
{
	std::ifstream file(fileName, std::ios_base::binary);
	assert(file.is_open());

	return ReadLabelsHeader(file);
}

uint32_t GetNumberOfPixels(const MnistImageHeader & header)
{
	return header.numberOfColumns*header.numberOfRows;
}

std::vector<double> ConvertPixelsToData(const std::vector<uint8_t> & pixels)
{
	std::vector<double> ret;

	for (size_t i = 0; i < pixels.size(); ++i)
		ret.push_back(((double)pixels[i])/255.0);

	return ret;
}

std::vector<double> ConvertLabelToData(uint8_t label)
{
	std::vector<double> ret;

	for (uint8_t i = 0; i < NUMBER_OF_OUTPUT_DIGITS; ++i)
		ret.push_back( i == label ? 1.0 : 0.0);

	return ret;
}

void ProcessMnistData(const char * imageFileName, const char * labelFileName, 
	std::function<void(const std::vector<uint8_t> & pixels, uint8_t label)> callback)
{
	std::ifstream imageFile(imageFileName, std::ios_base::binary);
	MnistImageHeader imageHeader = ReadImageHeader(imageFile);

	std::ifstream labelFile(labelFileName);
	MnistLabelHeader labelHeader = ReadLabelsHeader(labelFile);

	uint32_t numberOfPixels = GetNumberOfPixels(imageHeader);

	std::vector<uint8_t> imagePixels(numberOfPixels, 0);
	uint8_t label;

	assert(imageHeader.numberOfImages == labelHeader.numberOfLabels);
	for (size_t i = 0; i < imageHeader.numberOfImages; ++i)
	{
		imageFile.read((char*)imagePixels.data(), imagePixels.size());
		labelFile.read((char*)&label, 1);

		callback(imagePixels, label);
	}
}

void TestMnist()
{
	std::cout << "Test MNIST" << std::endl;

	std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingExamples;

	ProcessMnistData(TRAIN_IMAGE_FILE, TRAIN_LABEL_FILE,
		[&trainingExamples]
	(const std::vector<uint8_t> & pixels, uint8_t label)
	{
		trainingExamples.push_back({ ConvertPixelsToData(pixels), ConvertLabelToData(label) });
	});

	std::vector<std::pair<std::vector<double>, uint8_t>> testExamples;

	ProcessMnistData(TEST_IMAGE_FILE, TEST_LABEL_FILE,
		[&testExamples]
	(const std::vector<uint8_t> & pixels, uint8_t label)
	{
		testExamples.push_back({ ConvertPixelsToData(pixels), label });
	});

	uint32_t numberOfPixels = GetNumberOfPixels(ReadImageHeader(TRAIN_IMAGE_FILE));

	auto network = NeuralNetwork::CreateNeuralNetwork({ numberOfPixels, 32, 32, NUMBER_OF_OUTPUT_DIGITS }, 
		WEIGHT_RANGE, BIAS_RANGE);

	size_t epochCount = 1;
	NeuralNetwork::StochasticGradientDescent(*network, trainingExamples, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE,
		[&testExamples, &network, &epochCount]
	{
		size_t errorCount = 0;
		double error = 0.0;
		for (const auto & test : testExamples)
		{
			auto data = NeuralNetwork::FeedForward(*network, test.first);
			size_t maxIndex = std::distance(std::begin(data), std::max_element(std::begin(data), std::end(data)));
			errorCount += maxIndex == test.second ? 0 : 1;

			for (size_t i = 0; i < data.size(); ++i)
				error += i == test.second ? fabs(1.0 - data[i]) : fabs(data[i]);
		}

		std::cout << "Epoch " << epochCount++ << " " << std::fixed << std::setprecision(2);
		std::cout << 100.0*(1.0 - (double)errorCount / (double)testExamples.size()) << "%";
		std::cout << " (" << testExamples.size() - errorCount << " / " << testExamples.size() << ")";
		std::cout << " error " << std::fixed << std::setprecision(3) << error << std::endl;
	});
}
