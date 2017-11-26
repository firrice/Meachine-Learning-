#include "PCA_Demension.h"  


PCA_Demension::PCA_Demension(void)
{
}


PCA_Demension::~PCA_Demension(void)
{
}

/*
�������ƣ�covariance
�������ܣ�Э������ȡ
���룺vector<double> x ---- �������x
double x_mean --- x�ľ�ֵ
vector<double> y ---- �������y
double y_mean ---- y�ľ�ֵ
�����double result --- ����������Э����
*/
/*int PCA_Demension::covariance(vector<double> x, double x_mean, vector<double> y, double y_mean, double & result)
{
	double sum_temp = 0;

	for (size_t i = 0; i < x.size(); ++i)
	{
		sum_temp = sum_temp + (x[i] - x_mean)*(y[i] - y_mean);
	}

	result = sum_temp / (x.size() - 1);

	return 1;
}*/
//�����������Ⱦ�����ȥ��ֵ�������Լ���Э�����ʱ����Ҫ��ֵ��
int PCA_Demension::covariance(vector<double> x, vector<double> y, double & result)
{
	double sum_temp = 0;

	for (size_t i = 0; i < x.size(); ++i)
	{
		sum_temp = sum_temp + x[i] * y[i];
	}

	result = sum_temp / x.size() ;

	return 1;
}

/*
�������ƣ�PCA_demension
�������ܣ�PCA��ά
���룺vector<vector<double> > Feature_Data ---- ��Ҫ��ά�����ݣ�ÿһ����һ��������ÿһ�д���һ������
int k --- ����Kά
�����vector<vector<double> > PCA_Features---����Э����������������
vector<vector<double> >  Final_Data -----����������ά�������ֵ
����ֵ: 1 --- ������ȷ ��0 --- �������

*/
int PCA_Demension::PCA_demension(vector<vector<double> > Feature_Data, int k, vector<vector<double> > & PCA_Features, vector<vector<double> > & Final_Data)
{
	//��ÿһ�е�������ȡƽ��ֵ  
	vector<double> Mean_values(Feature_Data[0].size(), 0);//��ʼ��  
	for (size_t j = 0; j < Feature_Data[0].size(); ++j)
	{
		double sum_temp = 0;
		for (size_t i = 0; i < Feature_Data.size(); ++i)
		{
			sum_temp = sum_temp + Feature_Data[i][j];
		}
		Mean_values[j] = sum_temp / Feature_Data.size();
	}

	//����������������ȥ��Ӧ��ƽ��ֵ,ȥ��ֵ����  
	for (size_t j = 0; j<Feature_Data[0].size(); ++j)
	{
		for (size_t i = 0; i<Feature_Data.size(); ++i)
		{
			Feature_Data[i][j] = Feature_Data[i][j] - Mean_values[j];
		}
	}

	//��ȡ����Э�������  
	//����֮ǰ�����Ƚ���ת��  
	//ʵ��ת�ù���  
	vector<vector<double> > trans;
	for (size_t i = 0; i<Feature_Data[0].size(); i++)
	{
		vector<double> temp;
		for (size_t j = 0; j<Feature_Data.size(); j++)
		{
			temp.push_back(Feature_Data[j][i]);
		}
		trans.push_back(temp);
		temp.clear();
	}
	//ʵ����ȡЭ�������  
	vector<vector<double> > covariance_Matrix;
	for (size_t i = 0; i<Feature_Data[0].size(); ++i)
	{
		vector<double> temp;
		for (size_t j = 0; j<Feature_Data[0].size(); ++j)
		{
			double result = 0;
			covariance(trans[i], trans[j], result);

			temp.push_back(result);
		}
		covariance_Matrix.push_back(temp);
		temp.clear();
	}

	//��Э������������ֵ����������  
	/*
	vector<vector<double> > pca_result;
	double dbEps = 0.001;
	int nJt = 1000;
	JacbiCor(covariance_Matrix,k,dbEps,nJt,pca_result);
	*/
	MatrixXd covariance_Matrix_temp(covariance_Matrix.size(), covariance_Matrix.size());
	for (size_t i = 0; i< covariance_Matrix.size(); ++i)
	{
		for (size_t j = 0; j< covariance_Matrix.size(); ++j)
		{
			covariance_Matrix_temp(i, j) = covariance_Matrix[i][j];
		}
	}

	EigenSolver<MatrixXd> es(covariance_Matrix_temp);
	MatrixXd V = es.pseudoEigenvectors();
	cout << "Э������������V �� " << endl << V << endl;

	MatrixXd D = es.pseudoEigenvalueMatrix();
	cout << "Э��������ֵ����D��" << endl << D << endl;
	//��������  
	std::map<double, int> mapEigen;
	int D_row = D.rows();
	int V_row = V.rows();
	double * pdbEigenValues = new double[D_row];
	double * pdbVecs = new double[(D_row)*(D_row)];

	for (size_t i = 0; i < V_row; ++i)
	{
		for (size_t j = 0; j < V_row; ++j)
		{
			pdbVecs[i*(V_row)+j] = V(j, i);
		}
	}
	double *pdbTmpVec = new double[(D_row)*(D_row)];
	for (size_t i = 0; i< D_row; ++i)
	{
		pdbEigenValues[i] = D(i, i);
		mapEigen.insert(make_pair(pdbEigenValues[i], i));
	}

	std::map<double, int>::reverse_iterator iter = mapEigen.rbegin();
	for (int j = 0; iter != mapEigen.rend(), j < D_row; ++iter, ++j)
	{
		for (int i = 0; i < D_row; i++)
		{
			pdbTmpVec[j*D_row + i] = pdbVecs[i + (iter->second)*D_row];
		}

		//����ֵ��������  
		pdbEigenValues[j] = iter->first;
	}

	memcpy(pdbVecs, pdbTmpVec, sizeof(double)*(D_row)*(D_row));
	delete[] pdbTmpVec;

	//ѡ��k����������  
	for (size_t i = 0; i< k; ++i)
	{
		vector<double> temp;
		for (size_t j = 0; j< D_row; ++j)
		{
			temp.push_back(pdbVecs[i*D_row + j]);//һ����һ������  
		}
		PCA_Features.push_back(temp);
		temp.clear();
	}

	delete[] pdbEigenValues;
	delete[] pdbVecs;


	//���о���˷�  
	//�ȶ�PCA_Features����ת�ã��������  
	vector<vector<double> > trans_fature;
	for (size_t i = 0; i<PCA_Features[0].size(); i++)
	{
		vector<double> temp;
		for (size_t j = 0; j<PCA_Features.size(); j++)
		{
			temp.push_back(PCA_Features[j][i]);
		}
		trans_fature.push_back(temp);
		temp.clear();
	}
	//vector<vector<double> > Final_Data;  //Final_data�ǽ�ά�������������
	for (size_t i = 0; i<Feature_Data.size(); ++i)
	{
		vector<double> temp_v;
		for (size_t k = 0; k < trans_fature[0].size(); ++k)
		{
			double temp_m = 0;
			for (size_t j = 0; j<Feature_Data[0].size(); ++j)
			{
				temp_m = temp_m + Feature_Data[i][j] * trans_fature[j][k];
			}
			temp_v.push_back(temp_m);
		}
		Final_Data.push_back(temp_v);
		temp_v.clear();
	}
	/*cout << "��ά֮������� : " << endl;
	for (int m = 0; m<Final_Data.size(); m++)           //b.size()��ȡ�������Ĵ�С
	{
		for (int n = 0; n<Final_Data[m].size(); n++)    //��ȡ�����о���ÿ�������Ĵ�С
			cout << Final_Data[m][n] << " ";
		cout << "\n";
	}*/
	return 1;
}

int main(void)
{
	vector<vector<double>> src_data(3);  //ԭʼ����
	for (int i = 0; i < 3; i++)
		src_data[i].resize(3);

	double src[3][3] = { 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1 };
	int k = 2;  //������ά��
	vector<vector<double> > result_feature; //��ά֮�������
	vector<vector<double> > result_data;  //��ά֮�������

	PCA_Demension op;
	//��ֵ��ԭʼ����
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			src_data[i][j] = src[i][j];
	}
	//����PCA��ά
	op.PCA_demension(src_data, k, result_feature, result_data);
	//�����ά֮�����������
	cout << "��ά֮��������� : " << endl;
	for (int m = 0; m<result_feature.size(); m++)           //b.size()��ȡ�������Ĵ�С
	{
		for (int n = 0; n<result_feature[m].size(); n++)    //��ȡ�����о���ÿ�������Ĵ�С
			cout << result_feature[m][n] << " ";
		cout << "\n";
	}
	cout << "��ά֮������� : " << endl;
	cout << result_data.size() << endl;
	for (int m = 0; m<result_data.size(); m++)           //b.size()��ȡ�������Ĵ�С
	{
		for (int n = 0; n<result_data[m].size(); n++)    //��ȡ�����о���ÿ�������Ĵ�С
			cout << result_data[m][n] << " ";
		cout << "\n";
	}

	while (1);
	return 0;
}