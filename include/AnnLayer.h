#ifndef ANNLAYER_H
#define ANNLAYER_H

#include <matrix.h>

class AnnLayer
{
    public:
        //The default constructor
        AnnLayer();

        //Constructor for input/output size
        AnnLayer(int paraInputSize, int paraOutputSize, char paraActivation);

        //Destructor
        virtual ~AnnLayer();

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
