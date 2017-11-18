
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "grabcutapp.h"
#include <iostream>

using namespace std;
using namespace cv;

void help()
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
    		"and then grabcut will attempt to segment it out.\n"
    		"Call:\n"
    		"./grabcut <image_name>\n"
    	"\nSelect a rectangular area around the object you want to segment\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tn - next iteration\n"
        "\n"
        "\tleft mouse button - set rectangle\n"
        "\n"
        "\tCTRL+left mouse button - set GC_BGD pixels\n"
        "\tSHIFT+left mouse button - set CG_FGD pixels\n"
        "\n"
        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
        "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;

void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 1;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
    int getIterCount() const { return iterCount; }
	void getMask(Mat& segmask);


private:
    void setRectInMask();
    void setLblsInMask( int flags, cv::Point p, bool isPr );

    const string* winName;
    const Mat* image;
    Mat mask;
    Mat bgdModel, fgdModel;

    uchar rectState, lblsState, prLblsState;
    bool isInitialized;

    cv::Rect rect;
    vector<cv::Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
};

void GCApplication::getMask(Mat& segmask)
{

	if( !mask.empty() )
		segmask = mask.clone(); 

}

void GCApplication::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
	{
        image->copyTo( res );
		/*
		vector<cv::Point>::const_iterator it;
		for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
			circle( res, *it, radius, BLUE, thickness );
		for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
			circle( res, *it, radius, RED, thickness );
		for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
			circle( res, *it, radius, LIGHTBLUE, thickness );
		for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
			circle( res, *it, radius, PINK, thickness );

		if( rectState == IN_PROCESS || rectState == SET )
			rectangle( res, cv::Point( rect.x, rect.y ), cv::Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
			*/

	}
    else
    {
        getBinMask( mask, binMask );

			// show mask contour
			vector<vector<cv::Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(binMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
			res = image->clone();
			
			//cout << contours.size() << " contours detected" << endl; 
			for(int k=0; k<contours.size(); k++)
			{
				Scalar color(0, 255, 0);//Scalar color(rand()&255, rand()&255, rand()&255);
				drawContours(res, contours, k, color, 2, 8, hierarchy);
			}


        //image->copyTo( res, binMask );
    }

	
    vector<cv::Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, cv::Point( rect.x, rect.y ), cv::Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

    imshow( *winName, res );

}

void GCApplication::setRectInMask()
{
    assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

void GCApplication::setLblsInMask( int flags, cv::Point p, bool isPr )
{
    vector<cv::Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }
    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if( rectState == NOT_SET && !isb && !isf )
            {
                rectState = IN_PROCESS;
                rect = cv::Rect( x, y, 1, 1 );
            }
            if ( (isb || isf) && rectState == SET )
                lblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) && rectState == SET )
                prLblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = cv::Rect( cv::Point(rect.x, rect.y), cv::Point(x,y) );
            rectState = SET;
            setRectInMask();
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = cv::Rect( cv::Point(rect.x, rect.y), cv::Point(x,y) );
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), true);
            showImage();
        }
        break;
    }
}

int GCApplication::nextIter()
{
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        if( rectState != SET )
            return iterCount;

        if( lblsState == SET || prLblsState == SET )
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
        else
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

GCApplication gcapp;

void on_mouse( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}

//int main( int argc, char** argv )
int grabcutapp(string filename, Mat& binFinalMask)
{

    if( filename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << filename << endl;
        return 1;
    }
    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
    	return 1;
    }

    help();

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_NORMAL);//CV_WINDOW_AUTOSIZE );
    cvSetMouseCallback( winName.c_str(), on_mouse, 0 );

    gcapp.setImageAndWinName( image, winName );
    gcapp.showImage();

    for(;;)
    {
        int c = cvWaitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            cout << "Exiting ..." << endl;
            goto exit_main;
        case 'r':
            cout << endl;
            gcapp.reset();
            gcapp.showImage();
            break;
		case 'n':

            int iterCount = gcapp.getIterCount();
            cout << "<" << iterCount << "... ";
			double t = getTickCount();
            int newIterCount = gcapp.nextIter();
			cout << (double) (getTickCount()-t)/getTickFrequency() << " seconds used for this iteration" << endl;
			
			//Mat finalmask, binFinalMask;
			Mat finalmask;
			gcapp.getMask(finalmask);
			getBinMask( finalmask, binFinalMask );
			imwrite(filename+"_seg_gbc.png", binFinalMask);
			Mat fgImg = Mat::zeros(image.size(), CV_8UC1);
			image.copyTo(fgImg, binFinalMask);
			imwrite(filename+"_fgd.png", fgImg);

            if( newIterCount > iterCount )
            {
                gcapp.showImage();
                cout << iterCount << ">" << endl;
            }
            else
                cout << "rect must be determined>" << endl;
            break;
        }
    }

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
