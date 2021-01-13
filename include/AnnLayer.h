/*
 * The C++ Artificial Neural network project.
 * This class manages one layer of ANN.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef ANNLAYER_H
#define ANNLAYER_H

#include <Matrix.h>

class AnnLayer
{
    public:
        //The default constructor
        AnnLayer();

        //Constructor for input/output size
        AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation);

        //Destructor
        virtual ~AnnLayer();

        //Convert to string for display
        string toString();

        //Set the activation function
        void setActivation(char paraActivation);

        //Forward calculation
        Matrix* forward(Matrix* paraMatrix);

        //Code unit test
        void selfTest();

    protected:
        //The size of the input
        int inputSize;

        //The size of the output
        int outputSize;

        //The activation function
        char activation;

        //The weights for edges
        Matrix* weightMatrix;

        //The offset
        Matrix* offsetMatrix;

    private:
};

#endif // ANNLAYER_H
