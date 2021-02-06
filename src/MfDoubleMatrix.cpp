/*
 * The C++ Artificial Neural network project.
 * This class manages double matrices.
 * Parallel computing using GPU will be enabled soon.
 * Code available at: github.com/fansmale/cann.
 * Author: Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#include <MfDoubleMatrix.h>

using namespace std;

/**
 * The default constructor.
 */
MfDoubleMatrix::MfDoubleMatrix()
{
    rows = 0;
    columns = 0;
    data = nullptr;
    activator = nullptr;
}//Of the default constructor

//
/**
 * The constructor. Initialize a matrix with given sizes.
 * paraRows: the number of rows.
 * paraColumns: the number of columns.
 */
MfDoubleMatrix::MfDoubleMatrix(int paraRows, int paraColumns)
{
    rows = paraRows;
    columns = paraColumns;

    //Allocate space
    data = new double *[rows];
    for (int i = 0; i < rows; i ++)
    {
        data[i] = new double[columns];
    }//Of for i

    //Some initial values
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = rand() / (double)(RAND_MAX);
        }//Of for j
    }//Of for i

    activator = nullptr;
}//Of the second constructor

//Destructor
MfDoubleMatrix::~MfDoubleMatrix()
{
    for (int i = 0; i < rows; i ++)
    {
        free(data[i]);
    }//Of for i
    free(data);

    free(activator);
}//Of the destructor

//Show me with a string for display
string MfDoubleMatrix::toString()
{
    //string resultString = "I am a matrix with size " + to_string(rows)
    //    + "*" + to_string(columns) + "\r\n";
    string resultString = "";
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            resultString += to_string(data[i][j]) +  ", ";
        }//Of for j
        resultString += "\r\n";
    }//Of for i

    //resultString += "matrix ends. \r\n";

    return resultString;
}//Of toString

/**
 * Getter.
 */
int MfDoubleMatrix::getRows()
{
    return rows;
}//Of getRows

/**
 * Getter.
 */
int MfDoubleMatrix::getColumns()
{
    return columns;
}//Of getColumns

/**
 * Getter. Please use with caution because it is a pointer.
 */
double** MfDoubleMatrix::getData()
{
    return data;
}//Of getData

/**
 * Getter. Get a value at the given position.
 * paraRow: the row.
 * paraColumn: the column.
 */
double MfDoubleMatrix::getValue(int paraRow, int paraColumn)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("MfDoubleMatrix.getValue() out of range.");
        throw "MfDoubleMatrix.getValue() out of range.";
    }//Of if
    return data[paraRow][paraColumn];
}//Of getValue

/**
 * Setter. Set a value at the given position.
 * paraRow: the row.
 * paraColumn: the column.
 * paraValue: the new value.
 */
double MfDoubleMatrix::setValue(int paraRow, int paraColumn, double paraValue)
{
    if ((paraRow >= rows) || (paraColumn >= columns))
    {
        printf("MfDoubleMatrix.setValue() out of range.");
        throw "MfDoubleMatrix.setValue() out of range.";
    }//Of if
    data[paraRow][paraColumn] = paraValue;

    return paraValue;
}//Of setValue

/**
 * Setter. Set the activator, no new space is allocated.
 * paraActivator: the activator.
 */
void MfDoubleMatrix::setActivator(Activator* paraActivator)
{
    activator = paraActivator;
}//Of setActivator


/**
 * Activate each element. Hopefully, the speed will be faster with this embedded operation.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::activate()
{
    for(int i = 0; i < rows; i ++)
    {
        for(int j = 0; j < columns; j ++)
        {
            data[i][j] = activator->activate(data[i][j]);
        }//Of for j
    }//Of for i

    return this;
}//Of activate

/**
 * Copy a matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::clone()
{
    MfDoubleMatrix* resultMfDoubleMatrix = new MfDoubleMatrix(rows, columns);
    double** tempData = resultMfDoubleMatrix->data;

    //Copy
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] = data[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of clone

/**
 * Copy from a matrix. Change my values.
 * paraMatrix: the given matrix
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::cloneFrom(MfDoubleMatrix* paraMatrix)
{
    if (rows != paraMatrix->rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix->columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    //Copy
    double** tempData = paraMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = tempData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of cloneFrom

/**
 * Add another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::add(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = clone();

    if (rows != paraMatrix->rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix->columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    //Maybe the runtime can be saved in this way.
    double** tempData = resultMfDoubleMatrix->data;
    double** tempData2 = paraMatrix->data;

    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] += tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of add

/**
 * The first matrix adds the second one gets me.
 * paraFirstMatrix: the first matrix, often myself.
 * paraSecondMatrix: the second matrix.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::addToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix)
{
    if (paraFirstMatrix->rows != paraSecondMatrix->rows)
    {
        printf("MfDoubleMatrix.addToMe(), rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (paraFirstMatrix->columns != paraSecondMatrix->columns)
    {
        printf("MfDoubleMatrix.addToMe(), columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempFirstData = paraFirstMatrix->data;
    double** tempSecondData = paraSecondMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = tempFirstData[i][j] + tempSecondData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of addToMe

/**
 * Minus another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::subtract(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = clone();

    if (rows != paraMatrix->rows)
    {
        printf("MfDoubleMatrix.subtract(), rows do not match.");
        throw "Rows do not match";
    }//Of if

    if (columns != paraMatrix->columns)
    {
        printf("MfDoubleMatrix.subtract(), columns do not match.");
        throw "Columns do not match";
    }//Of if

    double** tempData = resultMfDoubleMatrix->data;
    double** tempData2 = paraMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] -= tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of subtract

/**
 * The first matrix subtracts the second one gets me.
 * paraFirstMatrix: the first matrix, often myself.
 * paraSecondMatrix: the second matrix.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::subtractToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix)
{
     if (paraFirstMatrix->rows != paraSecondMatrix->rows)
    {
        printf("MfDoubleMatrix.subtractToMe(), rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (paraFirstMatrix->columns != paraSecondMatrix->columns)
    {
        printf("MfDoubleMatrix.subtractToMe(), columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempFirstData = paraFirstMatrix->data;
    double** tempSecondData = paraSecondMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = tempFirstData[i][j] - tempSecondData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of subtractToMe

/**
 * Multiply another one with the same size.
 * paraMatrix: the other matrix.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::cwiseProduct(MfDoubleMatrix* paraMatrix)
{
    MfDoubleMatrix* resultMfDoubleMatrix = clone();

    if (rows != paraMatrix->rows)
    {
        printf("Rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (columns != paraMatrix->columns)
    {
        printf("Columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempData = resultMfDoubleMatrix->data;
    double** tempData2 = paraMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            tempData[i][j] *= tempData2[i][j];
        }//Of for j
    }//Of for i

    return resultMfDoubleMatrix;
}//Of cwiseProduct

/**
 * The first matrix point-to-point products the second one gets me.
 * paraFirstMatrix: the first matrix, often myself.
 * paraSecondMatrix: the second matrix.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::cwiseProductToMe(MfDoubleMatrix* paraFirstMatrix, MfDoubleMatrix* paraSecondMatrix)
{

     if (paraFirstMatrix->rows != paraSecondMatrix->rows)
    {
        printf("MfDoubleMatrix.cwiseProductToMe(), rows do not match.");
        throw "Rows do not match.";
    }//Of if

    if (paraFirstMatrix->columns != paraSecondMatrix->columns)
    {
        printf("MfDoubleMatrix.cwiseProductToMe(), columns do not match.");
        throw "Columns do not match.";
    }//Of if

    double** tempFirstData = paraFirstMatrix->data;
    double** tempSecondData = paraSecondMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = tempFirstData[i][j] * tempSecondData[i][j];
        }//Of for j
    }//Of for i

    return this;
}//Of cwiseProductToMe

/**
 * Times another one.
 * paraMatrix: the other matrix with size n*k.
 * Return: a new matrix with size m*k.
 */
MfDoubleMatrix* MfDoubleMatrix::times(MfDoubleMatrix *paraMatrix)
{
    int tempColumns = paraMatrix->getColumns();

    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns ,tempColumns);
    if (columns != paraMatrix->rows)
    {
        printf("Matrices do not match.");
        throw "Matrices do not match.";
    }//Of if

    //Attention: one exception happened here consistently, but now disappeared.
    //That is, columns changed by the following statement.
    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns,tempColumns);
    MfDoubleMatrix* newMfDoubleMatrixPtr = new MfDoubleMatrix(rows, tempColumns);
    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns, tempColumns);
    double tempValue = 0;

    double** tempData1 = newMfDoubleMatrixPtr->data;
    double** tempData2 = paraMatrix->data;
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < tempColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < columns; k ++)
            {
                tempValue += data[i][k] * tempData2[k][j];
            }//Of for k
            tempData1[i][j] = tempValue;
        }//Of for j
    }//Of for i

    return newMfDoubleMatrixPtr;
}//Of times

/**
 * Two matrices times to fill me.
 * paraFirstMatrix: the first matrix, size m*n.
 * paraSecondMatrix: the second matrix, size n*k.
 * Return: myself, size m*k.
 */
MfDoubleMatrix* MfDoubleMatrix::timesToMe(MfDoubleMatrix *paraFirstMatrix, MfDoubleMatrix *paraSecondMatrix)
{
    int tempFirstRows = paraFirstMatrix->getRows();
    int tempFirstColumns = paraFirstMatrix->getColumns();
    int tempSecondRows = paraSecondMatrix->getRows();
    int tempSecondColumns = paraSecondMatrix->getColumns();

    double** tempFirstData = paraFirstMatrix->data;
    double** tempSecondData = paraSecondMatrix->data;

    //printf("rows = %d, columns = %d, tempColumns = %d.\r\n", rows, columns ,tempColumns);
    if (tempFirstColumns != tempSecondRows)
    {
        printf("Matrices do not match.");
        throw "Matrices do not match.";
    }//Of if

    double tempValue = 0;

    for (int i = 0; i < tempFirstRows; i ++)
    {
        for (int j = 0; j < tempSecondColumns; j ++)
        {
            tempValue = 0;
            for (int k = 0; k < tempFirstColumns; k ++)
            {
                tempValue += tempFirstData[i][k] * tempSecondData[k][j];
            }//Of for k
            data[i][j] = tempValue;
        }//Of for j
    }//Of for i

    return this;
}//Of timesToMe

/**
 * Transpose the matrix to another.
 * Return: the transposed matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::transpose()
{
    MfDoubleMatrix* newMfDoubleMatrixPtr = new MfDoubleMatrix(columns, rows);

    double** tempData = newMfDoubleMatrixPtr->data;
    for (int i = 0; i < columns; i ++)
    {
        for (int j = 0; j < rows; j ++)
        {
            tempData[i][j] = data[j][i];
        }//Of for j
    }//Of for i

    return newMfDoubleMatrixPtr;
}//Of transpose

/**
 * Fill the matrix with the same value.
 * paraValue: the given value.
 */
void MfDoubleMatrix::fill(double paraValue)
{
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < columns; j ++)
        {
            data[i][j] = paraValue;
        }//Of for j
    }//Of for i
}//Of fill

/**
 * Convolution valid elements. The sizes of three matrices should match.
 * paraData: the given data.
 * paraKernel: the given kernel.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::convolutionValidToMe(MfDoubleMatrix *paraData, MfDoubleMatrix *paraKernel)
{
    int tempDataRows = paraData->getRows();
    int tempDataColumns = paraData->getColumns();
    int tempKernelRows = paraKernel->getRows();
    int tempKernelColumns = paraKernel->getColumns();

    //Step 1. Size check.
    if (rows != tempDataRows - tempKernelRows + 1)
    {
        printf("In convolutionValidToMe, rows not match: %d vs. %d.", rows,
               (tempDataRows - tempKernelRows + 1));
        throw "rows not match.";
    }//Of if
    if (columns != tempDataColumns - tempKernelColumns + 1)
    {
        printf("In convolutionValidToMe, columns not match: %d vs. %d.", columns,
               (tempDataColumns - tempKernelColumns + 1));
        throw "columns not match.";
    }//Of if

    //Step 2. Given then new names.
    double** tempData = paraData->data;
    double** tempKernel = paraKernel->data;

    //Step 3. Now compute.
    for(int i = 0; i < rows; i ++)
    {
        for(int j = 0; j < columns; j ++)
        {
            data[i][j] = 0;
            for(int k = 0; k < tempKernelRows; k ++)
            {
                for(int i1 = 0; i1 < tempKernelColumns; i1 ++)
                {
                    data[i][j] += tempData[i + k][j + i1] * tempKernel[k][i1];
                }//Of for i1
            }//Of for k
        }//Of for j
    }//Of for i

    return this;
}//Of convolutionValidToMe

/**
 * Convolution full elements. The sizes of three matrices should match.
 * paraData: the given data.
 * paraKernel: the given kernel.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::convolutionFullToMe(MfDoubleMatrix *paraData, MfDoubleMatrix *paraKernel)
{
    int tempDataRows = paraData->getRows();
    int tempDataColumns = paraData->getColumns();
    int tempKernelRows = paraKernel->getRows();
    int tempKernelColumns = paraKernel->getColumns();

    //Step 1. Size check.
    if (rows != tempDataRows + tempKernelRows - 1)
    {
        printf("In convolutionFullToMe, rows not match: %d vs. %d.\r\n", rows,
               (tempDataRows + tempKernelRows - 1));
        throw "rows not match.";
    }//Of if
    if (columns != tempDataColumns + tempKernelColumns - 1)
    {
        printf("In convolutionFullToMe, columns not match: %d vs. %d.\r\n", columns,
               (tempDataColumns + tempKernelColumns - 1));
        throw "columns not match.";
    }//Of if

    //Step 3. A temp matrix.
    MfDoubleMatrix* tempExtendedMatrix = new MfDoubleMatrix(tempDataRows + 2 * (tempKernelRows - 1),
            tempDataColumns + 2 * (tempKernelColumns - 1));
    tempExtendedMatrix->fill(0);

    //Step 4. Copy.
    double** tempData = paraData->data;
    double** tempExtended = tempExtendedMatrix->data;
    for(int i = 0; i < tempDataRows; i ++)
    {
        for(int j = 0; j < tempDataColumns; j ++)
        {
            tempExtended[i + tempKernelRows - 1][j + tempKernelColumns - 1] = tempData[i][j];
        }//Of for j
    }//Of for i

    //Step 5. Convolution.
    convolutionValidToMe(tempExtendedMatrix, paraKernel);

    //Step 6. Free the space.
    free(tempExtendedMatrix);

    return this;
}//Of convolutionFullToMe

/**
 * Rotate myself by 180 degrees.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::rotate180()
{
    double tempValue;
    //Step 1. Invert columns.
    for(int i = 0; i < rows; i ++)
    {
        for(int j = 0; j < columns / 2; j ++)
        {
            tempValue = data[i][j];
            data[i][j] = data[i][columns - j - 1];
            data[i][columns - j - 1] = tempValue;
        }//Of for j
    }//Of for i

    //Step 2. Invert rows.
    for(int i = 0; i < rows / 2; i ++)
    {
        for(int j = 0; j < columns; j ++)
        {
            tempValue = data[i][j];
            data[i][j] = data[rows - i - 1][j];
            data[rows - i - 1][j] = tempValue;
        }//Of for j
    }//Of for i

    return this;
}//Of rotate180

/**
 * Rotate 180 degrees to me.
 * paraMatrix: the given matrix.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::rotate180ToMe(MfDoubleMatrix* paraMatrix)
{
    double tempValue;
    double** tempData = paraMatrix->data;
    //Step 1. Invert columns.
    for(int i = 0; i < rows; i ++)
    {
        for(int j = 0; j < columns / 2; j ++)
        {
            data[i][j] = tempData[i][columns - j - 1];
            data[i][columns - j - 1] = tempData[i][j];
        }//Of for j
    }//Of for i

    //Step 2. Invert rows.
    for(int i = 0; i < rows / 2; i ++)
    {
        for(int j = 0; j < columns; j ++)
        {
            tempValue = data[i][j];
            data[i][j] = data[rows - i - 1][j];
            data[rows - i - 1][j] = tempValue;
        }//Of for j
    }//Of for i

    return this;
}//Of rotate180ToMe

/**
 * Scale the given matrix with the given size.
 *   For example, matrix of size (4, 12) scaled by size (2, 3) gets a new size (2, 4).
 *   The values of each area is averaged.
 * paraMatrix: the given matrix.
 * paraSize: the given size.
 * Return: a new matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::scale(MfSize* paraSize)
{
    int tempWidth = paraSize->width;
    int tempHeight = paraSize->height;

    int newRows = rows / tempWidth;
    int newColumns = columns / tempHeight;
    int tempArea = tempWidth * tempHeight;
    double tempSum;

    if((newRows * tempWidth != rows) ||(newColumns * tempHeight != columns))
    {
        printf("MfDoubleMatrix::scale, size cannot be divided to int.\r\n");
        throw "MfDoubleMatrix::scale, size cannot be divided to int.";
    }//Of if

    MfDoubleMatrix* resultMatrix = new MfDoubleMatrix(newRows, newColumns);
    double** tempData = resultMatrix->data;

    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newColumns; j++) {
            tempSum = 0.0;
            for (int si = i * tempWidth; si < (i + 1) * tempWidth; si++) {
                for (int sj = j * tempHeight; sj < (j + 1) * tempHeight; sj++) {
                    tempSum += data[si][sj];
                }//Of for sj
            }//Of for si
            tempData[i][j] = tempSum / tempArea;
        }//Of for j
    }//Of for i
    return resultMatrix;
}//Of scale

/**
 * Scale the given matrix with the given size to me. No new space is allocated here.
 *   For example, matrix of size (4, 12) scaled by size (2, 3) gets a new size (2, 4).
 *   The values of each area is averaged.
 * paraMatrix: the given matrix.
 * paraSize: the given size.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::scaleToMe(MfDoubleMatrix* paraMatrix, MfSize* paraSize)
{
    int tempRows = paraMatrix->rows;
    int tempColumns = paraMatrix->columns;
    int tempWidth = paraSize->width;
    int tempHeight = paraSize->height;

    int newRows = tempRows / tempWidth;
    int newColumns = tempColumns / tempHeight;
    int tempArea = tempWidth * tempHeight;
    double tempSum;

    if((newRows != rows) ||(newColumns != columns))
    {
        printf("MfDoubleMatrix::scaleToMe, size not match mine.\r\n");
        throw "MfDoubleMatrix::scaleToMe, size not match mine.";
    }//Of if

    if((newRows * tempWidth != tempRows) ||(newColumns * tempHeight != tempColumns))
    {
        printf("MfDoubleMatrix::scaleToMe, size cannot be divided to int.\r\n");
        throw "MfDoubleMatrix::scaleToMe, size cannot be divided to int.";
    }//Of if

    double** paraData = paraMatrix->data;
    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newColumns; j++) {
            tempSum = 0.0;
            for (int si = i * tempWidth; si < (i + 1) * tempWidth; si++) {
                for (int sj = j * tempHeight; sj < (j + 1) * tempHeight; sj++) {
                    tempSum += paraData[si][sj];
                }//Of for sj
            }//Of for si
            data[i][j] = tempSum / tempArea;
        }//Of for j
    }//Of for i
    return this;
}//Of scaleToMe

/**
 * Kronecker to a bigger matrix.
 * paraMatrix: the given matrix.
 * paraSize: the given size.
 * Return: the bigger matrix.
 */
MfDoubleMatrix* MfDoubleMatrix::kronecker(MfSize* paraSize)
{
    int tempWidth = paraSize->width;
    int tempHeight = paraSize->height;

    MfDoubleMatrix* resultMatrix = new MfDoubleMatrix(rows * tempWidth, columns * tempHeight);
    double** newData = resultMatrix->data;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            for (int ki = i * tempWidth; ki < (i + 1) * tempWidth; ki++) {
                for (int kj = j * tempHeight; kj < (j + 1) * tempHeight; kj++) {
                    newData[ki][kj] = data[i][j];
                }//Of for kj
            }//Of for ki
        }//Of for j
    }//Of for i

    return resultMatrix;
}//Of kronecker

/**
 * Kronecker to a bigger matrix.
 * paraMatrix: the given matrix.
 * paraSize: the given size.
 * Return: myself.
 */
MfDoubleMatrix* MfDoubleMatrix::kroneckerToMe(MfDoubleMatrix* paraMatrix, MfSize* paraSize)
{
    int tempRows = paraMatrix->rows;
    int tempColumns = paraMatrix->columns;
    int tempWidth = paraSize->width;
    int tempHeight = paraSize->height;

    if((tempRows * tempWidth != rows) ||(tempColumns * tempHeight != columns))
    {
        printf("MfDoubleMatrix::kroneckerToMe, size cannot be divided to int.\r\n");
        throw "MfDoubleMatrix::kroneckerToMe, size cannot be divided to int.";
    }//Of if

    double** paraData = paraMatrix->data;

    for (int i = 0; i < tempRows; i++) {
        for (int j = 0; j < tempColumns; j++) {
            for (int ki = i * tempWidth; ki < (i + 1) * tempWidth; ki++) {
                for (int kj = j * tempHeight; kj < (j + 1) * tempHeight; kj++) {
                    data[ki][kj] = paraData[i][j];
                }//Of for kj
            }//Of for ki
        }//Of for j
    }//Of for i

    return this;
}//Of kroneckerToMe

/**
 * Code unit test.
 */
void MfDoubleMatrix::unitTest()
{
    MfDoubleMatrix* tempMfDoubleMatrix = new MfDoubleMatrix(3, 5);
    printf("Original\r\n");

    printf(tempMfDoubleMatrix->toString().data());

    MfDoubleMatrix* tempMfDoubleMatrix2 = tempMfDoubleMatrix->clone();
    printf("Copy\r\n");
    printf(tempMfDoubleMatrix2->toString().data());

    MfDoubleMatrix* tempTransposed = tempMfDoubleMatrix->transpose();
    printf("Transpose\r\n");
    printf(tempTransposed->toString().data());

    MfDoubleMatrix* tempDot = tempMfDoubleMatrix->times(tempTransposed);
    printf("Dot\r\n");
    printf(tempDot->toString().data());

    MfDoubleMatrix* tempAdded = tempMfDoubleMatrix->add(tempMfDoubleMatrix2);
    printf("Add\r\n");
    printf(tempAdded->toString().data());

    MfDoubleMatrix* tempMultiply = tempMfDoubleMatrix->cwiseProduct(tempMfDoubleMatrix2);
    printf("Multiply\r\n");
    printf(tempMultiply->toString().data());

    MfDoubleMatrix* tempMinus = tempMultiply ->subtract(tempMfDoubleMatrix);
    printf("Minus\r\n");
    printf(tempMinus->toString().data());

    MfDoubleMatrix* tempConvolutionValid = new MfDoubleMatrix(2, 3);
    MfDoubleMatrix* tempKernel = new MfDoubleMatrix(2, 3);

    //tempMfDoubleMatrix->fill(1);
    //tempKernel->fill(1);

    tempConvolutionValid->convolutionValidToMe(tempMfDoubleMatrix, tempKernel);
    printf("Data\r\n");
    printf(tempMfDoubleMatrix->toString().data());
    printf("Kernel\r\n");
    printf(tempKernel->toString().data());
    printf("Convolution valid\r\n");
    printf(tempConvolutionValid->toString().data());

    MfDoubleMatrix* tempConvolutionFull = new MfDoubleMatrix(4, 7);
    tempConvolutionFull->convolutionFullToMe(tempMfDoubleMatrix, tempKernel);
    printf("Convolution full\r\n");
    printf(tempConvolutionFull->toString().data());

    tempConvolutionFull->rotate180();
    printf("Rotate 180\r\n");
    printf(tempConvolutionFull->toString().data());

    MfDoubleMatrix* tempRotate = new MfDoubleMatrix(4, 7);
    tempRotate->rotate180ToMe(tempConvolutionFull);
    printf("Rotate 180 to me\r\n");
    printf(tempRotate->toString().data());

    MfDoubleMatrix* tempBigMatrix = new MfDoubleMatrix(4, 12);
    MfSize* tempSize = new MfSize(2, 3);
    //tempBigMatrix->fill(1);
    MfDoubleMatrix* tempScaledMatrix = new MfDoubleMatrix(2, 4);
    tempScaledMatrix->scaleToMe(tempBigMatrix, tempSize);
    printf("tempBigMatrix\r\n");
    printf(tempBigMatrix->toString().data());
    printf("Scale (2, 3)\r\n");
    printf(tempScaledMatrix->toString().data());

    MfDoubleMatrix* tempKronecker = tempScaledMatrix->kronecker(tempSize);
    printf("Kronecker (2, 3)\r\n");
    printf(tempKronecker->toString().data());

    printf("Kronecker (2, 3) to me\r\n");
    MfDoubleMatrix* tempKronecker2 = new MfDoubleMatrix(4, 12);
    tempKronecker2->kroneckerToMe(tempScaledMatrix, tempSize);
    printf(tempKronecker2->toString().data());

    Activator* tempActivator = new Activator('s');
    tempKronecker2 -> setActivator(tempActivator);
}//Of unitTest
