#ifndef FULLANN_H
#define FULLANN_H

#include <Matrix.h>
#include <AnnLayer.h>

class FullAnn
{
    public:
        //The default constructor
        FullAnn();

        //The Ann with given sizes
        FullAnn(Matrix* paraSize);

        //Destructor
        virtual ~FullAnn();

        Matrix* forward(Matrix* paraInput);
        void backward();

    protected:
        AnnLayer** layers;

    private:
};

#endif // FULLANN_H
