#ifndef _LIB_UANN_H_
#define _LIB_UANN_H_

class Layer{
public:
	/*s-sigmoid t-tansig l-linear g-gaussian r-ReLU*/
	Layer(int Nin,int Nout,char tfunc);

	~Layer();

	int LoadLayer(int Nin,int Nout,char tfunc);

	void PrintInfo();


	int intial(unsigned int seed);

	int GetNumIn(){return(this->Num_Input);}
	
	int GetNumout(){return(this->Num_Output);}

	int SetParameters(double *InWeightMatrix,double *InBiasVector);

	int Work(double *Input,double *Output){
		int i;
		for(i=0;i<this->Num_Output;i++){
			Output[i]=this->autotf
				(
				BiasVector[i]+
				this->dot(
				Input,
				(this->WeightMatrix+i*this->Num_Input),
				this->Num_Input
				)
				);
		}
		return(0);	
	}

private:
	
	int Num_Input;
	int Num_Output;
	char tf;
	double *WeightMatrix;
	double *BiasVector;
	
	int isfree32(void *D_Pointer);
	
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
	ANNet(int NumOfLayers,int *NodeInLayer,char *TransFunc);

	~ANNet();
	
	void Work(double *Input,double *Output){
		double *Temp1=NULL;
		double *Temp2=NULL;
		int i=0;
		Temp1=(double *)malloc(sizeof(double)*(MaxLayer+1));
		Temp2=(double *)malloc(sizeof(double)*(MaxLayer+1));
		if(isfree32(Temp1)||isfree32(Temp2)){
			printf("Malloc failed!\n");
			exit(1);
		}
		Layers[0].Work(Input,Temp1);
		for(i=1;i<(Num_Layer-1);i++){
			if(i%2==1){
				Layers[i].Work(Temp1,Temp2);
			}else{
				Layers[i].Work(Temp2,Temp1);
			}
		}
		if(Num_Layer%2==1){
			//use Temp2
			Layers[Num_Layer-1].Work(Temp2,Output);
		}else{
			//use Temp1
			Layers[Num_Layer-1].Work(Temp1,Output);
		}
		free(Temp1);
		free(Temp2);
	}

	void PrintInfo();

	int SetLayerParameters(double *InWeightMatrix,double *InBiasVector,int LayerID);

private:
	
	Layer *Layers;
	int Num_Layer;
	int MaxLayer;

	//int *Nodes_Per_Layer;

	ANNet operator =(ANNet in);

	int isfree32(void *D_Pointer);

};

#endif