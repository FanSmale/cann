/*
 * The C++ Artificial Neural network project.
 * This class constructs a full ANN using AnnLayers.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef FULLANN_H
#define FULLANN_H

#include <Matrix.h>
#include <AnnLayer.h>
#include <IntArray.h>

class FullAnn
{
    public:
        //The default constructor
        FullAnn();

        //The Ann with given sizes and activation function
        FullAnn(IntArray* paraSizes, char paraActivation);

        //Destructor
        virtual ~FullAnn();

        //Convert to string for display
        string toString();

        //Set the activation function for the given layer
        void setActivation(int paraLayer, char paraActivation);

        //Forward layer by layer
        Matrix* forward(Matrix* paraInput);

        //Back propagation
        void backpropagation();

        //Code unit test
        void selfTest();

    protected:

        //Number of layers
        int numLayers;

        //All layers
        AnnLayer** layers;

    private:
};

#endif // FULLANN_H
