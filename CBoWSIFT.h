
#pragma once

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>

using namespace std;
using namespace cv;

class CBoWSIFT
{
public:

	CBoWSIFT(Mat&, int=400);
	CBoWSIFT(Mat& img, Mat& vocabulary, int=400);
	~CBoWSIFT(void);
	void extractSIFT();
	void buildVocab(Mat&,Mat&,string&);
	void extractCodeMap(Mat&);

private:

	Mat m_img; //!< source image
	Mat m_vocabulary;
	int m_vocabSize;
	Mat m_siftMat;
	Mat m_imDesc32;
	Mat m_CodeWordColorMap;

};

