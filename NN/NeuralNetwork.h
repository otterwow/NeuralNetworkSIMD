//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer

namespace Tmpl8 {

	struct TrainingEntry
	{
		float inputs[INPUTSIZE];
		int expected[NUMOUTPUT];
	};

	struct TrainingSet
	{
		TrainingEntry* entry;
		int size;
	};

	struct TrainingData
	{
		TrainingData(int t, int g, int v)
		{
			trainingSet.entry = new TrainingEntry[t];
			trainingSet.size = t;
			generalizationSet.entry = new TrainingEntry[g];
			generalizationSet.size = g;
			validationSet.entry = new TrainingEntry[v];
			validationSet.size = v;
		}
		TrainingSet trainingSet;
		TrainingSet generalizationSet;
		TrainingSet validationSet;
	};

	class Network
	{
		friend class NetworkTrainer;
		inline static float SigmoidActivationFunction(float x) { return 1.0f / (1.0f + expf(-x)); }
		inline static __m256 SigmoidActivationFunction8(__m256 x);
		inline static int ClampOutputValue(float x) { if (x < 0.1f) return 0; else if (x > 0.9f) return 1; else return -1; }
		inline float GetOutputErrorGradient(float desiredValue, float outputValue) const { return outputValue * (1.0f - outputValue) * (desiredValue - outputValue); }
		inline __m256 GetOutputErrorGradient8(__m256 desiredValue8, __m256 outputValue8);
		int GetInputHiddenWeightIndex(int inputIdx, int hiddenIdx) const { return inputIdx * (NUMHIDDEN + 1) + hiddenIdx; }
		int GetHiddenOutputWeightIndex(int hiddenIdx, int outputIdx) const { return hiddenIdx * NUMOUTPUT + outputIdx; }
	public:
		Network();
		const int* Evaluate(const float* input);
		void Train(const TrainingData& trainingData);
		const float* GetInputHiddenWeights() const { return weightsInputHidden; }
		const float* GetHiddenOutputWeights() const { return weightsHiddenOutput; }
		void LoadWeights(const float* weights);
		void SaveWeights(float* weights);
		void InitializeNetwork();
		void InitializeWeights();
		float GetHiddenErrorGradient(int hiddenIdx) const;
		__m256 GetHiddenErrorGradient8(int hiddenIdx) const;
		void RunEpoch(const TrainingSet& trainingSet);
		void Backpropagate(const int* expectedOutputs);
		void GetSetAccuracyAndMSE(const TrainingSet& trainingSet, float& accuracy, float& mse);
	private:
		// neural net data
		float inputNeurons[INPUTSIZE + 1];
		float hiddenNeurons[NUMHIDDEN + 1];
	public:
		float outputNeurons[NUMOUTPUT];
	private:
		int clampedOutputs[NUMOUTPUT];
		float* weightsInputHidden;
		float* weightsHiddenOutput;
		__m256 ni8[NUMNI8]; // input neurons
		__m256 nh8[NUMNH8]; // hidden neurons
		__m256 no8[NUMNO8]; // output neurons
		__m256* wih8; // weights from input to hidden layer
		__m256* owih8; //
		__m256* who8; // weights from hidden to output layer
		__m256* owho8; //

		//__m256 dih8[NUMWIH8]; // delta weights from input to hidden layer
		__m256 dho8[NUMWHO8]; // delta weights from hidden to output layer
		const __m256 learningRate8 = { LEARNINGRATE, LEARNINGRATE,LEARNINGRATE,LEARNINGRATE,LEARNINGRATE,LEARNINGRATE,LEARNINGRATE,LEARNINGRATE};
		const __m256 momentum8 = { MOMENTUM, MOMENTUM,MOMENTUM,MOMENTUM,MOMENTUM,MOMENTUM,MOMENTUM,MOMENTUM };
		const __m256 one8 = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };

		// training data
		float*   deltaInputHidden;              // delta for input hidden layer
		float*   deltaHiddenOutput;             // delta for hidden output layer
		float*   errorGradientsHidden;          // error gradients for the hidden layer
		float*   errorGradientsOutput;          // error gradients for the outputs
		int      currentEpoch;                  // epoch counter
		float    trainingSetAccuracy;
		float    validationSetAccuracy;
		float    generalizationSetAccuracy;
		float    trainingSetMSE;
		float    validationSetMSE;
		float    generalizationSetMSE;
	};

} // namespace Tmpl8