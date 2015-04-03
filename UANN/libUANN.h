#ifndef _LIB_UANN_H_
#define _LIB_UANN_H_

class Layer{
public:
	/*s-sigmoid t-tansig l-linear g-gaussian r-ReLU*/
	Layer(int Nin,int Nout,char tfunc){
		int rtn;
		rtn=this->LoadLayer(Nin,Nout,tfunc);

		if(rtn<0){
			printf("Malloc Weight Space Failed.\n");
			exit(1);
		}

	}

	~Layer(){

		if(!isfree32(WeightMatrix)){ 
			free(WeightMatrix);
		}

		if(!isfree32(BiasVector)){
			free(BiasVector);
		}

	}

	int LoadLayer(int Nin,int Nout,char tfunc);

	void PrintInfo();

	int Work(double *Input,double *Output);

	int intial(unsigned int seed);

	int GetNumIn(){return(this->Num_Input);}
	
	int GetNumout(){return(this->Num_Output);}

	int SetParameters(double *InWeightMatrix,double *InBiasVector);

private:
	
	int Num_Input;
	int Num_Output;
	char tf;
	double *WeightMatrix;
	double *BiasVector;
	
	int isfree32(void *D_Pointer){
		//Free:Yes Using:NO
		return(D_Pointer==NULL || D_Pointer==(void *)0xcdcdcdcd || D_Pointer==(void *)0xcccccccc);
	}
	
	double autotf(double input){
		switch (this->tf){
		case 's':
			return(1/(1+exp(-input)));//sigmoid
			break;
		case 't':
			double x1,x2;
			x1=exp(input);
			x2=exp(-input);
			return((x1-x2)/(x1+x2));//tansig
			break;
		case 'l':
			return(input);
			break;
		case 'g':
			return(exp(-(input*input)));
			break;
		case 'r':
			return(input>0.0?input:0);
			break;
		default:
			return(1/(1+exp(-input)));//sigmoid
			break;
		}
	}
	
	/*
	//常见传递函数的代码，已集成在上述函数中，故注释
	double sigmoid(double input){
		return(1/(1+exp(-input)));
	}	
	double tansig(double input){
		double x1,x2;
		x1=exp(input);
		x2=exp(-input);
		return((x1-x2)/(x1+x2));
	}	
	double gaussian(double input){
		return(exp(-(input*input)));
	}	
	double ReLU(double input){
		return(input>0.0?input:0);
	}
	*/

	double dot(double *v1,double *v2,int len){
		double sum=0.0;
		int i;

		for (i=0;i<len;i++){
			sum+=v1[i]*v2[i];
		}

		return(sum);
	}

	Layer operator =(Layer in);

};

class ANNet{

public:

	ANNet(int NumOfLayers,int *NodeInLayer,char *TransFunc){
		MaxLayer=0;
		Num_Layer=NumOfLayers;
		Layers=(Layer *)malloc(sizeof(Layer)*NumOfLayers);

		if(Layers==NULL){
			printf("Malloc failed #1.");
			getch();
			exit(1);
		}

		MaxLayer=NodeInLayer[0];

		for (int i=0;i<NumOfLayers;i++){

			if (NodeInLayer[i+1]>MaxLayer){
				MaxLayer=NodeInLayer[i+1];
			}

			Layers[i].LoadLayer(NodeInLayer[i],NodeInLayer[i+1],TransFunc[i]);
			Layers[i].intial(i+NodeInLayer[i]*TransFunc[i]);
		}
	}

	~ANNet(){
		int i=0;
		for(i=0;i<Num_Layer;i++){
			Layers[i].~Layer();
		}
		if(!isfree32(Layers)){
			free(Layers);
		}
	}
	
	void Work(double *Input,double *Output);

	void PrintInfo();

	int SetLayerParameters(double *InWeightMatrix,double *InBiasVector,int LayerID);
private:
	
	Layer *Layers;
	int Num_Layer;
	int MaxLayer;
	//int *Nodes_Per_Layer;
	ANNet operator =(ANNet in);

	int isfree32(void *D_Pointer){
		//Free：Yes Using：NO
		return(D_Pointer==NULL || D_Pointer==(void *)0xcdcdcdcd || D_Pointer==(void *)0xcccccccc);
	}

};

#endif