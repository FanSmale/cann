/*
 * The C++ Artificial Neural network project.
 * This class manages all kinds of activators.
 * Code available at: github.com/fansmale/cann.
 * Author: Heng-Ru Zhang and Fan Min
 *   Lab of Machine Learning, School of Computer Science, Southwest Petroleum University, Chengdu, China
 *   www.fansmale.com
 *   minfanphd@163.com, minfan@swpu.edu.cn
 */

#ifndef ACTIVATOR_H
#define ACTIVATOR_H

#define DEFAULT_ACTIVATOR_GAMMA 0.01
#define DEFAULT_ACTIVATOR_BETA 2

#include <math.h>
#include <stdio.h>
#include <string>

using namespace std;

class Activator
{
    public:
        //The default constructor
        Activator();

        //The constructor
        Activator(char paraActivationFunction);

        //The destructor
        virtual ~Activator();

        //Convert to string for display.
        string toString();

        //Set activation function
        void setActivationFunction(char paraFunction)
        {
            activationFunction = paraFunction;
        }

        //Set gamma
        void setGamma(double paraGamma)
        {
            gamma = paraGamma;
        }

        //Set beta
        void setBeta(double paraBeta)
        {
            beta = paraBeta;
        }

        //Activate
        double activate(double paraValue);

        //Derive
        double derive(double paraValue);

        //Unit test.
        void unitTest();

    protected:

        //The activation function
        char activationFunction;

        //The gamma value
        double gamma;

        //The beta value
        double beta;

    private:

        //The sigmoid activation function
        double sigmoid(double paraValue);

        //The sigmoid derive function
        double sigmoidDerive(double paraValue);

        //The tanh activation function
        double tanh(double paraValue);

        //The hard-logistic activation function
        double hardLogistic(double paraValue);

        //The hard-tanh activation function
        double hardTanh(double paraValue);

        //The relu activation function
        double relu(double paraValue);

        //The LeakyReLU activation function
        double leakyRelu(double paraValue);

        //The ELU activation function
        double elu(double paraValue);

        //The Softplus activation function
        double softplus(double paraValue);

        //The Softsign activation function
        double softsign(double paraValue);

        //The Swish activation function
        double swish(double paraValue);

        //The GELU activation function
        double gelu(double paraValue);
};

#endif // ACTIVATOR_H
