
#pragma once

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include <vector>

using namespace std;
using namespace cv;

class CTexton
{
public:
	CTexton(Mat&, int=400);
	CTexton(Mat& img, Mat& vocabulary, int=400);
	~CTexton(void);
	void buildVocab(Mat&,Mat&,string&);
	void buildVocabEM(Mat&,Mat&);
	void extractTexton(Mat&);
	void extractTextonEM(Mat&,cv::EM&);

private:
	void featureFilters();
	void gauss1d(double, double, double, int, double &);
	void makefilter(double, int, int, Mat&, int, Mat&);
	void gaussianFilter(Mat&, int, double);
	void lapGaussianFilter(Mat& H, int size, double std);
	void makeRFSfilters();

	Mat m_img; //!< source image
	int m_vocabSize;
	vector<Mat> m_F;
	vector<Mat> m_fullFeats;
	vector<Mat> m_rotInvarFeats;
	Mat m_vocabulary;
	Mat m_featMat;
	Mat m_cLabels;
	Mat m_textonMap;
	Mat m_textonColorMap;
};

