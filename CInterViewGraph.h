

#ifndef __CINTERVIEWGRAPH__
#define __CINTERVIEWGRAPH__

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "multiViewData.h"
#include "gmmbuilder.h"
#include "gcgraph.hpp"
#include <vector>

using namespace cv;

/**
 Graph of superpixels and pixels
*/



class CInterViewGraph
{
public:

	//! Class of the pixel 
	enum { GC_BGD_MASK    = 2,  //!< background
		   GC_FGD_MASK    = 1,  //!< foreground
		 };

	struct multiViewData* m_mvd;
	
	int m_spEdgeCnt;  //!< number of inter superpixel links
	int m_spIntraEdgeCnt;  //!< number of intra superpixel links
	int m_tolSPCnt;   //!< total number of superpixels
	
	vector<Mat> m_spIntraAdjMatVec; //!< Vector of intra frame superpixel adjacency matrix
	vector<Mat> m_spAdjMatVec; //!< Vector of superpixel adjacency matrix


	// vectors for graph weights
	vector<Mat> m_leftWVec; //!< Vector of pairwise terms to left pixel
	vector<Mat> m_upleftWVec; //!< Vector of pairwise terms to upper-left pixel
	vector<Mat> m_upWVec; //!< Vector of pairwise terms to upper pixel
	vector<Mat> m_uprightWVec; //!< Vector of pairwise terms to upper-right pixel
	vector<double> m_betaVec; //!< Vector of global variance of colour

	CInterViewGraph(struct multiViewData*, double); //!< Constructor
	void getMask(vector<Mat>& maskVec, vector<Mat>&);
	vector<vector<char> >& getSPMatchesMask();
	void run();

private:

	void constructGraph();
	void spAdjMat();
	void spIntraAdjMat();
	void spOptAdjMat();
	void calcPixNWeight();
	void addPixelNodesEdges();
	void addSuperpixelNodes();
	void addSuperpixelEdges();
	void addSppixelEdges();
	void estimateMotionPrior();
	float computeSPHistDist(int, int, int, int);
	float computeSPHistKLDiv(int, int, int, int);
	void motionDiffusion( int, cv::Point&, cv::Point&, Mat&, Mat&, Mat&);
	void gauss1d(double , double, double, int, double & );
	void makefilter(double scale, int phasex, int phasey, Mat& pts, int SUP, Mat& f);
	void gaussianFilter(Mat& H, int size, double std);
	void makeRFSfilters(Mat&, double, double=3);
	void skeletonization(Mat mask, Mat& skel);
	void inference();

	GCGraph<double> m_graph; //!< Graph of pixels and superpixels
	double m_gamma;   //!< Weight on pairwise term
	double m_lambda;  //!< Weight of definite FG/BG
	int m_vtxCount;   //!< Number of vertices
	int m_edgeCount;  //!< Number of edges
	int m_width;      //!< Width of image
	int m_height;     //!< Height of image
	vector<Mat> m_fgdPbVec;      //!< FG likelihood
	vector<Mat> m_bgdPbVec;      //!< BG likelihood
	vector<Mat> m_maskVec;       //!< Labeling map
	vector<Mat> m_spMaskVec;       //!< Superpixel labeling map
	vector<vector<char> > matchesMaskVec;//!< A list of matched keypoints per superpixel pair
	vector<vector<Mat> > m_motionPriorVec;       //!< Motion priors
	int m_kx;
	int m_ky;
	double m_sigma;
	
};



#endif