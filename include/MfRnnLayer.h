/*
 * The C++ Artificial Neural network project.
 * This class handles CNN layer.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef MFRNNLAYER_H
#define MFRNNLAYER_H

#include "MfDoubleMatrix.h"
#include "MfDoubleMatrixArray.h"

class MfRnnLayer
{
public:

    //The empty constructor.
    MfRnnLayer();

    //The second constructor.
    MfRnnLayer(int paraInputSize, int paraHiddenSize, int paraOutputSize, int paraMaxSequenceLength,
        double paraRate);

    //The destructor.
    virtual ~MfRnnLayer();

    //Setter.
    void setLearningRate(double paraRate)
    {
        learningRate = paraRate;
    }

    //Initialize.
    void initialize();

    //Getter.
    MfDoubleMatrixArray* gety()
    {
        return yAt;
    }

    //Getter.
    MfDoubleMatrixArray* getp()
    {
        return pAt;
    }

    //Forward.
    void forward(MfDoubleMatrixArray* paraData);

    //Getter.
    MfDoubleMatrixArray* getdy(MfIntArray* paraIntArray);

    //Get the loss.
    double getLoss(MfIntArray* paraIntArray);

    //Unit test.
    void unitTest();

protected:

    //Input size.
    int inputSize;

    //Hidden size.
    int hiddenSize;

    //Output size.
    int outputSize;

    //Learning rate.
    double learningRate;

    ////// Network state //////

    //Input layer weights.
	MfDoubleMatrix* Wxh;

	//Hidden layer weights.
	MfDoubleMatrix* Whh;

	//Output layer weights.
	MfDoubleMatrix* Why;

	//Hidden bias.
	MfDoubleMatrix* bh;

	//Output bias.
	MfDoubleMatrix* by;

	//Last hidden state.
	MfDoubleMatrix* h;

	////// Training state //////

	//Gradient descent parameters: input layer.
	MfDoubleMatrix* gWxh;

	//Gradient descent parameters: hidden layer.
	MfDoubleMatrix* gWhh;

	//Gradient descent parameters: output layer.
	MfDoubleMatrix* gWhy;

	//Gradient descent parameters: hidden bias.
	MfDoubleMatrix* gbh;

	//Gradient descent parameters: output bias.
	MfDoubleMatrix* gby;

	//Input vectors through time.
	MfDoubleMatrixArray* xAt;

    //Hidden state vectors through time.
	MfDoubleMatrixArray* hAt;

	//Unnormalized output probability vectors through time.
	MfDoubleMatrixArray* yAt;

	//Normalized output probability vectors through time.
	MfDoubleMatrixArray* pAt;

	//Output gradient from a backwards pass.
	MfDoubleMatrixArray* dxAt;

	//Was temporary in the java version.
	MfDoubleMatrixArray* dyAt;

	// The maximal sequence length. For space allocation.
	int maxSequenceLength;

	// Number of steps in the last forward pass (must match the steps for the backward pass).
	int lastSequenceLength;

	//Whether or not initialized.
	bool initialized;

	//The activator.
	Activator* activator;

private:

    //Hidden size row vector. A temporary vector.
    MfDoubleMatrix* hiddenSizeRowVector;

    //Hidden size column vector. A temporary vector.
    MfDoubleMatrix* hiddenSizeColumnVector;

};

#endif // MFRNNLAYER_H
