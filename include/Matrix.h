#ifndef MATRIX_H
#define MATRIX_H

class Matrix
{
    public:
        //The default constructor
        Matrix();

        //Initialize a matrix with given sizes
        Matrix(int paraRows, int paraColumns);

        //Destructor
        virtual ~Matrix();

        //Show me
        void showMe();

        //Copy a matrix
        Matrix* copy();

        //Add another one with the same size
        Matrix* add(Matrix *paraMatrix);

        //Minus another one with the same size
        Matrix* minus(Matrix *paraMatrix);

        //Multiply another one with the same size
        Matrix* multiply(Matrix *paraMatrix);

        //Dot multiply, return a new matrix
        Matrix* dot(Matrix *paraMatrix);

        //Transpose, return a new matrix
        Matrix* transpose();

        //Code unit test
        void selfTest();

    protected:

        //Number of rows
        int rows;

        //Number of columns
        int columns;

        //The data
        double** data;

    private:
};

#endif // MATRIX_H
