#ifndef ACTIVATOR_H
#define ACTIVATOR_H

#define DEFAULT_ACTIVATOR_GAMMA 0.01
#define DEFAULT_ACTIVATOR_BETA 2

class Activator
{
    public:
        //The default constructor
        Activator();

        //The constructor
        Activator(char paraActivationFunction);

        //The destructor
        virtual ~Activator();

        //Set activation function
        void setActivationFunction(char paraFunction);

        //Set gamma
        void setGamma(double paraGamma);

        //Set beta
        void setBeta(double paraBeta);

        //The sigmoid activation function
        double sigmoid(double paraValue);

        //The tanh activation function
        double tanh(double paraValue);

        //The hard-logistic activation function
        double hardLogistic(double paraValue);

        //The hard-tanh activation function
        double hardTanh(double paraValue);

        //The relu activation function
        double relu(double paraValue);

        //The LeakyReLU activation function
        double leakyRelu(double paraValue, double paraGamma);

        //The ELU activation function
        double elu(double paraValue, double paraGamma);

        //The Softplus activation function
        double softplus(double paraValue);

        //The Softsign activation function
        double softsign(double paraValue);

        //The Swish activation function
        double swish(double paraValue, double paraBeta);

        //The GELU activation function
        double gelu(double paraValue);

        //Activate
        double activate(double paraValue);

        void selfTest();

    protected:
        //The activation function
        char activationFunction;

        //The gamma value
        double activationGamma;

        //The beta value
        double activationBeta;
    private:
};

#endif // ACTIVATOR_H
