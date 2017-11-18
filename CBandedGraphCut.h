

#ifndef __CBANDEDGRAPHCUT__
#define __CBANDEDGRAPHCUT__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "gcgraph.hpp"
#include "gmmbuilder.h"
#include <iostream>

using namespace std;
using namespace cv;

class CBandedGraphCut
{
public:
	CBandedGraphCut(Mat&, Mat&, int&, const double&, GMMBuilder&);//!< 

	Mat getMask();

	//! Class of the pixel 
	enum { GC_BGD_MASK    = 2,  //!< background
		   GC_FGD_MASK    = 1,  //!< foreground
		 };


private:
	Mat m_img;
	Mat m_dsMask;
	Mat m_mask;
	Mat m_bandMap;
	int m_levels;
	int m_width;
	int m_height;
	GCGraph<double> m_graph;
	vector<cv::Point> m_bandList;
	Mat m_vtxIdxMat;
	Mat m_bgdPb;
	Mat m_fgdPb;
	Mat m_leftW;
	Mat m_upleftW;
	Mat m_upW;
	Mat m_uprightW;
	double m_beta;
	double m_gamma;
	double m_lambda;
	GMMBuilder* m_gmm;//!< GMM colour model

	void calcBandedNWeights();
	void constructBandedGCGraph();
	void estimateBandedSegmentation(Mat&);
	void run();


};



#endif