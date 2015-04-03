// UANN.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#define NIPT 4
#define NOPT 4

int Layer_UNIT_TEST(){
	double ipt[NIPT]={0.5,-0.6,0.7,-0.9};
	double opt[NOPT]={0};
	int i;
	Layer L1( NIPT , NOPT ,'t');
	printf("Neural Layer Test Strat!\n");
	L1.intial(1);
	L1.Work(ipt,opt);
	L1.PrintInfo();
	printf("Output:");
	for(i=0;i<NOPT;i++){
		printf("%lf ",opt[i]);
	}
	printf("\nLayer Unit Test Finished\n");
	getch();
	return 0;
}

int ANNet_UNIT_TEST(){
	int NodeInLayer[3]={1,3,1};
	double Layer0WM[3]={1,2,3};
	double Layer0BV[3]={0.1,0.2,0.3};
	double Layer1WM[3]={4,5,6};
	double Layer1BV[1]={0.5};
	char tf[2]={'s','l'};
	double input[1]={10.0};
	double output[1]={0.1};

	printf("Neural Network Test Strat!\n");

	ANNet Net0(2,NodeInLayer,tf);
	Net0.SetLayerParameters(Layer0WM,Layer0BV,0);
	Net0.SetLayerParameters(Layer1WM,Layer1BV,1);

	Net0.PrintInfo();

	Net0.Work(input,output);

	printf("Input:%lf\nOutput:%lf",input[0],output[0]);
	getch();
	return(0);
}

int _tmain(int argc, _TCHAR* argv[]){
	Layer_UNIT_TEST();
	ANNet_UNIT_TEST();
	return(0);
}

