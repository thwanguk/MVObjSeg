
#ifndef __GMM__
#define __GMM__


#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

//#include "precomp.hpp"
#include <limits>

using namespace cv;

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 15;

    GMM( Mat& _model );
	GMM();

    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};



#endif