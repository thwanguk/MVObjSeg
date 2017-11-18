
#ifndef __UI__
#define __UI__

#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
//#include "graphcutseg.hpp"

#include <iostream>

using namespace std;
using namespace cv;

const Scalar RED = Scalar(0,0,255);
const Scalar BLUE = Scalar(255,0,0);

const int BGD_KEY = CV_EVENT_FLAG_RBUTTON;
const int FGD_KEY = CV_EVENT_FLAG_LBUTTON;

enum { GC_BGD_MASK    = 2,  //!< background
		   GC_FGD_MASK    = 1,  //!< foreground
		 };

void getBinMask( Mat* comMask, Mat& binMask )
{
    if( comMask->empty() || comMask->type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask->rows || binMask.cols!=comMask->cols )
        binMask.create( comMask->size(), CV_8UC1 );
    binMask = *comMask & 1;
}

class GCUI
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 1;
    static const int thickness = -1;//-1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName, Mat& _mask );
    void showImage();
    void mouseClick( int event, int x, int y, int flags, void* param );
    void cut();
	void getMask(Mat& ex_mask);
	void getImage(Mat& ex_image);

private:

    void setLblsInMask( int flags, Point p );

    const string* winName;
    const Mat* image;
	Mat image_temp;
    Mat* mask;
    Mat bgdModel, fgdModel;

    uchar lblsState;
    bool isInitialized;

    vector<Point> fgdPxls, bgdPxls;

};

void GCUI::getMask(Mat& ex_mask)
{
	if( !mask->empty() )
		ex_mask = mask->clone(); 
}

void GCUI::getImage(Mat& ex_image)
{
	ex_image = image_temp.clone(); 
}


void GCUI::reset()
{
    if( !mask->empty() )
        mask->setTo(Scalar::all(0));
    bgdPxls.clear(); fgdPxls.clear();

    isInitialized = false;
    lblsState = NOT_SET;
}

void GCUI::setImageAndWinName( const Mat& _image, const string& _winName, Mat& _mask  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
	mask = &_mask;
    //mask->create( image->size(), CV_8UC1);
    reset();
}

void GCUI::showImage() 
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );

    imshow( *winName, res );
	res.copyTo(image_temp);
}


void GCUI::setLblsInMask( int flags, Point p )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;

    bpxls = &bgdPxls;
    fpxls = &fgdPxls;
    bvalue = GC_BGD_MASK;
    fvalue = GC_FGD_MASK;

    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( *mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( *mask, p, radius, fvalue, thickness );
    }
}

void GCUI::mouseClick( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {

    case CV_EVENT_LBUTTONDOWN: // set GC_FGD labels
        lblsState = IN_PROCESS;
		setLblsInMask(flags, Point(x,y));
        break;

    case CV_EVENT_RBUTTONDOWN: // set GC_BGD(GC_PR_FGD) labels
        lblsState = IN_PROCESS;
		setLblsInMask(flags, Point(x,y));
        break;

    case CV_EVENT_LBUTTONUP:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y));
            lblsState = SET;
            showImage();
        }
        break;

    case CV_EVENT_RBUTTONUP:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y));
            lblsState = SET;
            showImage();
			//namedWindow("mask1",CV_WINDOW_NORMAL);
			//imshow("mask1",(2-*mask)*127);
        }
        break;

    case CV_EVENT_MOUSEMOVE:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y));
            showImage();
        }
        break;
    }
}


#endif