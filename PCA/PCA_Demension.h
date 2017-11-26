//PCA_Demension.h文件  
#pragma once  
#include<iostream>  
#include<vector>  
#include<map>  
#include<Eigen/Dense>  
using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;
//pca降维代码的实现 2017.02.16   
//copyright: PH-SCUT  
class PCA_Demension
{
public:
	PCA_Demension(void);
	~PCA_Demension(void);
	//int covariance(vector<double> x, double x_mean, vector<double> y, double y_mean, double & result);
	int covariance(vector<double> x, vector<double> y, double & result);
	int PCA_demension(vector<vector<double> > Feature_Data, int k, vector<vector<double> > & PCA_Features, vector<vector<double> > & Final_Data);
};