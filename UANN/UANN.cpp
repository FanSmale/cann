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

	L1.intial(1);
	L1.Work(ipt,opt);
	//L1.invWork(ipt,opt);
	L1.PrintInfo();
	printf("Output:");
	for(i=0;i<NOPT;i++){
		printf("%lf ",opt[i]);
	}
	printf("\nLayer Unit Test Finished\n");
	getch();
	return 0;
}

int _tmain(int argc, _TCHAR* argv[]){
	int NodeInLayer[3]={1,3,1};
	char tf[2]={'s','l'};
	double input[1]={10.0};
	double output[1]={0.1};

	ANNet Net0(2,NodeInLayer,tf);

	Net0.PrintInfo();
	Net0.Work(input,output);
	printf("Input:%lf\nOutput:%lf",input[0],output[0]);
	getch();
	return(0);
}

