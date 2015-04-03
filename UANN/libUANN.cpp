#include "stdafx.h"

//神经网络单层函数

Layer::Layer(int Nin,int Nout,char tfunc){
	int rtn;
	rtn=this->LoadLayer(Nin,Nout,tfunc);

	if(rtn<0){
		printf("Malloc Weight Space Failed.\n");
		exit(1);
	}

}

Layer::~Layer(){

	if(!isfree32(WeightMatrix)){ 
		free(WeightMatrix);
	}

	if(!isfree32(BiasVector)){
		free(BiasVector);
	}

}

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

int Layer::SetParameters(double *InWeightMatrix,double *InBiasVector){
	int i=0;
	const int WMsize=Num_Input*Num_Output;
	for (i=0; i<WMsize; i++){
		WeightMatrix[i]=InWeightMatrix[i];
	}
	for (i=0; i<Num_Output; i++){
		BiasVector[i]=InBiasVector[i];
	}
	return(0);
}

int Layer::isfree32(void *D_Pointer){
	//Free:Yes Using:NO
	return(D_Pointer==NULL || D_Pointer==(void *)0xcdcdcdcd || D_Pointer==(void *)0xcccccccc);
}
//神经网络总体函数


void ANNet::PrintInfo(){
	int i=0;
	for(i=0;i<Num_Layer;i++){
		printf("\nLayer%d:\n",i);
		Layers[i].PrintInfo();
	}
}

int ANNet::SetLayerParameters(double *InWeightMatrix,double *InBiasVector,int LayerID){
	return(Layers[LayerID].SetParameters(InWeightMatrix,InBiasVector));
}

ANNet::ANNet(int NumOfLayers,int *NodeInLayer,char *TransFunc){
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

ANNet::~ANNet(){
	int i=0;
	for(i=0;i<Num_Layer;i++){
		Layers[i].~Layer();
	}
	if(!isfree32(Layers)){
		free(Layers);
	}
}

int ANNet::isfree32(void *D_Pointer){
	//Free：Yes Using：NO
	return(D_Pointer==NULL || D_Pointer==(void *)0xcdcdcdcd || D_Pointer==(void *)0xcccccccc);
}