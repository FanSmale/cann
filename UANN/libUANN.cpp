#include "stdafx.h"

//神经网络单层函数

int Layer::LoadLayer(int Nin,int Nout,char tfunc){
	int WMsize=Nin*Nout;

	this->tf=tfunc;
	this->Num_Input=Nin;
	this->Num_Output=Nout;

	this->WeightMatrix=(double *)malloc(sizeof(double)*(WMsize));
	this->BiasVector=(double *)malloc(sizeof(double)*(Nout));

	if(isfree32(WeightMatrix) || isfree32(BiasVector)){
		return(-1);
	}

	return(0);
}

int Layer::intial(unsigned int seed){
	int WMsize=Num_Input*Num_Output;
	int i;

	srand(seed);

	for (i=0;i<WMsize;i++){
		WeightMatrix[i]=((double)rand())/32768;
	}

	for (i=0;i<Num_Output;i++){
		BiasVector[i]=((double)rand())/32768;
	}

	return(0);
}

void Layer::PrintInfo(){
	int i,j;

	printf("Layer Information:\nTransfer Function:");
	switch (this->tf){
	case 's':
		printf("Sigmoid");
		break;
	case 't':
		printf("Tansig");
		break;
	case 'l':
		printf("Linear");
		break;
	case 'g':
		printf("Gaussian");
		break;
	case 'r':
		printf("ReLU");
		break;
	default:
		printf("Sigmoid(Default)");
		break;
	}
	printf("\n");

	printf("Weight Matrix and bias:\n");
	for (i=0;i<this->Num_Output;i++){

		printf("W%d = (",i);
		for (j=0;j<this->Num_Input;j++){
			if(j==0){
				printf("%lf",this->WeightMatrix[i*Num_Input+j]);
			}else{
				printf(",%lf",this->WeightMatrix[i*Num_Input+j]);
			}
		}

		printf(")    b%d = %lf",i,this->BiasVector[i]);
		printf("\n");

	}

}

int Layer::Work(double *Input,double *Output){
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

int Layer::invWork(double *invInput,double *invOutput){
	int i;
	for(i=0;i<this->Num_Input;i++){
		 
	}
	return(0);
}
//神经网络总体函数

void ANNet::Work(double *Input,double *Output){
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

void ANNet::PrintInfo(){
	int i=0;
	for(i=0;i<Num_Layer;i++){
		printf("\nLayer%d:\n",i);
		Layers[i].PrintInfo();
	}
}