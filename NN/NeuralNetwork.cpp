//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Based on Bobby Anguelov's code
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "..\precomp.h"

namespace Tmpl8 {

	Network::Network()
	{
		// initialize neural net
		InitializeNetwork();
		InitializeWeights();
	}

	void Network::InitializeNetwork()
	{
		wih8 = (__m256*) _aligned_malloc(477280, 32);
		owih8 = (__m256*) _aligned_malloc(477280, 32);
		who8 = (__m256*) _aligned_malloc(9664, 32);
		owho8 = (__m256*) _aligned_malloc(9664, 32);

		// set input bias node to 1
		ni8[NUMNI8 - 1].m256_f32[0] = -1.f;
		// set hidden bias node to 1
		nh8[NUMNH8 - 1].m256_f32[0] = -1.f;
	}

	void Network::InitializeWeights()
	{
		random_device rd;
		mt19937 generator(rd());
		const float distributionRangeHalfWidth = 2.4f / INPUTSIZE;
		const float standardDeviation = distributionRangeHalfWidth * 2 / 6;
		normal_distribution<> normalDistribution(0, standardDeviation);

		for (int i = 0; i < NUMWIH8; i++) {
			wih8[i] = _mm256_set_ps(
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator)
			);
		}

		memcpy(owih8, wih8, 14915 * 8 * 4);

		for (int i = 0; i < NUMWHO8; i++) {
			who8[i] = _mm256_set_ps(
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator),
				(float)normalDistribution(generator)
			);
		}

		memcpy(owho8, who8, 302 * 8 * 4);
	}

	void Network::LoadWeights(const float* weights)
	{
		const int numInputHiddenWeights = (INPUTSIZE + 1) * (NUMHIDDEN + 1);
		const int numHiddenOutputWeights = (NUMHIDDEN + 1) * NUMOUTPUT;
		int weightIdx = 0;
		for (int i = 0; i < numInputHiddenWeights; i++) weightsInputHidden[i] = weights[weightIdx++];
		for (int i = 0; i < numHiddenOutputWeights; i++) weightsHiddenOutput[i] = weights[weightIdx++];
	}

	void Network::SaveWeights(float* weights)
	{
		const int numInputHiddenWeights = (INPUTSIZE + 1) * (NUMHIDDEN + 1);
		const int numHiddenOutputWeights = (NUMHIDDEN + 1) * NUMOUTPUT;
		int weightIdx = 0;
		for (int i = 0; i < numInputHiddenWeights; i++) weights[weightIdx++] = weightsInputHidden[i];
		for (int i = 0; i < numHiddenOutputWeights; i++) weights[weightIdx++] = weightsHiddenOutput[i];
	}

	float Network::GetHiddenErrorGradient(int hiddenIdx) const
	{
		// get sum of hidden->output weights * output error gradients
		float weightedSum = 0;
		for (int i = 0; i < NUMOUTPUT; i++)
		{
			const int weightIdx = GetHiddenOutputWeightIndex(hiddenIdx, i);
			weightedSum += weightsHiddenOutput[weightIdx] * errorGradientsOutput[i];
		}
		// return error gradient
		return hiddenNeurons[hiddenIdx] * (1.0f - hiddenNeurons[hiddenIdx]) * weightedSum;
	}

	void Network::Train(const TrainingData& trainingData)
	{
		// reset training state
		currentEpoch = 0;
		trainingSetAccuracy = validationSetAccuracy = generalizationSetAccuracy = 0;
		trainingSetMSE = validationSetMSE = generalizationSetMSE = 0;
		// print header
		printf(" Neural Network Training Starting: \n");
		printf("==========================================================================\n");
		printf(" LR: %f, momentum: %f, max epochs: %i\n", LEARNINGRATE, MOMENTUM, MAXEPOCHS);
		printf(" %i input neurons, %i hidden neurons, %i output neurons\n", INPUTSIZE, NUMHIDDEN, NUMOUTPUT);
		printf("==========================================================================\n");
		// train network using training dataset for training and generalization dataset for testing
		while ((trainingSetAccuracy < TARGETACCURACY || generalizationSetAccuracy < TARGETACCURACY) && currentEpoch < MAXEPOCHS)
		{
			// use training set to train network
			timer t;
			t.reset();
			RunEpoch(trainingData.trainingSet);
			float epochTime = t.elapsed();
			// get generalization set accuracy and MSE
			GetSetAccuracyAndMSE(trainingData.generalizationSet, generalizationSetAccuracy, generalizationSetMSE);
			printf("Epoch: %03i - TS accuracy: %4.1f, MSE: %4.4f GS accuracy: %4.1f, in %06.1fms\n", currentEpoch, trainingSetAccuracy, trainingSetMSE, generalizationSetAccuracy, epochTime);
			currentEpoch++;
		}
		// get validation set accuracy and MSE
		GetSetAccuracyAndMSE(trainingData.validationSet, validationSetAccuracy, validationSetMSE);
		// print validation accuracy and MSE
		printf("\nTraining complete. Epochs: %i\n", currentEpoch);
		printf(" Validation set accuracy: %f\n Validation set MSE: %f\n", validationSetAccuracy, validationSetMSE);
	}

	void Network::RunEpoch(const TrainingSet& set)
	{
		float incorrectEntries = 0, MSE = 0;
		for (int i = 0; i < set.size; i++)
		{
			const TrainingEntry& entry = set.entry[i];
			// feed inputs through network and back propagate errors
			Evaluate(entry.inputs);
			Backpropagate(entry.expected);
			// check all outputs from neural network against desired values
			bool resultCorrect = true;
			for (int j = 0; j < NUMOUTPUT; j++)
			{
				if (clampedOutputs[j] != entry.expected[j]) resultCorrect = false;
				const float delta = outputNeurons[j] - entry.expected[j];
				MSE += delta * delta;
			}
			if (!resultCorrect) incorrectEntries++;
		}
		// update training accuracy and MSE
		trainingSetAccuracy = 100.0f - (incorrectEntries / set.size * 100.0f);
		trainingSetMSE = MSE / (NUMOUTPUT * set.size);
	}

	void Network::Backpropagate(const int* expectedOutputs)
	{
		// store expected outputs in f8
		no8[0] = GetOutputErrorGradient8(_mm256_set_ps(
			expectedOutputs[7],
			expectedOutputs[6],
			expectedOutputs[5],
			expectedOutputs[4],
			expectedOutputs[3],
			expectedOutputs[2],
			expectedOutputs[1],
			expectedOutputs[0]
		), no8[0]);

		no8[1] = GetOutputErrorGradient8(_mm256_set_ps(
			0,0,0,0,0,0,
			expectedOutputs[9],
			expectedOutputs[8]
		), no8[1]);


		//swap weights and old weights
		auto tmp = who8;
		who8 = owho8;
		owho8 = tmp;

		//setup variables for loop
		auto n = (float*)nh8;
		uint k = 0;
		__m256* prevWeight = owho8;

		//update weights - hidden - output
		for (__m256* weight = who8; weight < who8 + 302; weight++, prevWeight++, k++) {
			*weight =
				_mm256_add_ps(
					*prevWeight,
					_mm256_add_ps(
						_mm256_mul_ps(
							_mm256_set1_ps(LEARNINGRATE * n[k / 2]),
							no8[k % 2]
						),
						_mm256_mul_ps(
							momentum8,
							_mm256_sub_ps(
								*prevWeight,
								*weight
							)
						)
					)
				);
		}

		// get hidden error gradients
		for (int i = 0; i < NUMNH8; i++) {
			nh8[i] = GetHiddenErrorGradient8(i);
		}

		// multiply nodes with learning rate
		for (int i = 0; i < NUMNI8; i++) {
			ni8[i] = _mm256_mul_ps(
				ni8[i],
				learningRate8
			);
		}
		
		// swap weights with old weights
		tmp = wih8;
		wih8 = owih8;
		owih8 = tmp;


		n = (float*)ni8;
		//update weights - input - hidden
		for (int i = 0; i < 785; i++) {
			for (int h = 0; h < 19; h++) {
				auto weightIdx = i * 19 + h;
				wih8[weightIdx] =
					_mm256_add_ps(
						owih8[weightIdx],
						_mm256_add_ps(
							_mm256_mul_ps(
								_mm256_set1_ps(n[i]),
								nh8[h]
							),
							_mm256_mul_ps(
								momentum8,
								_mm256_sub_ps(
									owih8[weightIdx],
									wih8[weightIdx]
								)
							)
						)
					);
			}
		}
	}

	const int* Network::Evaluate(const float* input)
	{
		// set input neuron values
		for (int i = 0; i < NUMNI8 - 1; i++) {
			ni8[i] = _mm256_set_ps(
				input[i * 8 + 7],
				input[i * 8 + 6],
				input[i * 8 + 5],
				input[i * 8 + 4],
				input[i * 8 + 3],
				input[i * 8 + 2],
				input[i * 8 + 1],
				input[i * 8 + 0]
			);
		}

		// set hidden neurons to zero
		for (int i = 0; i < NUMNH8; i++) {
			nh8[i] = _mm256_setzero_ps();
		}

		// update hidden neuron values with input values
		auto n = (float*)ni8;
		for (int i = 0; i < 785; i++) {
			for (int h = 0; h < 19; h++) {
				nh8[h] =
					_mm256_add_ps(
						nh8[h],
						_mm256_mul_ps(
							wih8[i * 19 + h],
							_mm256_set1_ps(n[i])
						)
					);
			}
		}

		// apply sigmoid function hidden neurons
		for (int i = 0; i < NUMNH8; i++) {
			nh8[i] = SigmoidActivationFunction8(nh8[i]);
		}

		// set hidden bias node to -1 (we messed it up in the previous for loop maybe?)
		nh8[NUMNH8 - 1].m256_f32[6] = -1.f;
		// set dummy bias node to 0 (we messed it up in the previous loop maybe)
		nh8[NUMNH8 - 1].m256_f32[7] = 0.f;

		// set output neurons to zero
		for (int i = 0; i < NUMNO8; i++) {
			no8[i] = _mm256_setzero_ps();
		}

		n = (float*)nh8;
		// update output neuron values with input values
		for (int i = 0; i < NUMWHO8; i++) {
			no8[i % NUMNO8] =
				_mm256_add_ps(
					no8[i % NUMNO8],
					_mm256_mul_ps(
						who8[i],
						_mm256_set1_ps(n[i / 2])
					)
				);
		}

		// apply sigmoid function output neurons
		for (int i = 0; i < NUMNO8; i++) {
			no8[i] = SigmoidActivationFunction8(no8[i]);
		}

		// mask unnessesary results to prevents to smooth out backpropagation
		no8[1] = _mm256_mul_ps(
			no8[1],
			_mm256_set_ps(0, 0, 0, 0, 0, 0, 1, 1)
		);

		// get clamped values
		for (int i = 0; i < 10; i++) {
			outputNeurons[i] = no8[i / 8].m256_f32[i % 8];
			clampedOutputs[i] = ClampOutputValue(no8[i / 8].m256_f32[i % 8]);
		}

		// return clamped values
		return clampedOutputs;
	}

	void Network::GetSetAccuracyAndMSE(const TrainingSet& set, float& accuracy, float& MSE)
	{
		accuracy = 0, MSE = 0;
		float numIncorrectResults = 0;
		for (int i = 0; i < set.size; i++)
		{
			const TrainingEntry& entry = set.entry[i];
			Evaluate(entry.inputs);
			// check if the network outputs match the expected outputs
			int correctResults = 0;
			for (int j = 0; j < NUMOUTPUT; j++)
			{
				correctResults += (clampedOutputs[j] == entry.expected[j]);
				const float delta = outputNeurons[j] - entry.expected[j];
				MSE += delta * delta;
			}
			if (correctResults != NUMOUTPUT) numIncorrectResults++;
		}
		accuracy = 100.0f - (numIncorrectResults / set.size * 100.0f);
		MSE = MSE / (NUMOUTPUT * set.size);
	}

	inline __m256 Network::SigmoidActivationFunction8(__m256 x)
	{
		auto exp8 = _mm256_set_ps(
			expf(-x.m256_f32[7]),
			expf(-x.m256_f32[6]),
			expf(-x.m256_f32[5]),
			expf(-x.m256_f32[4]),
			expf(-x.m256_f32[3]),
			expf(-x.m256_f32[2]),
			expf(-x.m256_f32[1]),
			expf(-x.m256_f32[0])
		);
		
		// this version is less accurate, but faster
		return _mm256_rcp_ps(
			_mm256_add_ps(
				_mm256_set1_ps(1.f),
				exp8
			)
		);
	}

	inline __m256 Network::GetOutputErrorGradient8(__m256 desiredValue8, __m256 outputValue8)
	{
		return _mm256_mul_ps(
			outputValue8,
			_mm256_mul_ps(
				_mm256_sub_ps(
					one8,
					outputValue8
				),
				_mm256_sub_ps(
					desiredValue8,
					outputValue8
				)
			)
		);
	}
	
	__m256 Network::GetHiddenErrorGradient8(int hiddenIdx) const
	{
		__m256 ws8 = _mm256_setzero_ps();

		for (int i = 0; i < 8; i++) {

			auto a = _mm256_mul_ps(
				who8[16 * hiddenIdx + 2 * i],
				no8[0]
			);

			auto b = _mm256_mul_ps(
				who8[16 * hiddenIdx + 2 * i + 1],
				no8[1]
			);

			for (int j = 0; j < 8; j++)
				ws8.m256_f32[i] += a.m256_f32[j] + b.m256_f32[j];
		}

		return
			_mm256_mul_ps(
				_mm256_mul_ps(
					nh8[hiddenIdx],
					_mm256_sub_ps(
						one8,
						nh8[hiddenIdx]
					)
				),
				ws8
			);
	}
}