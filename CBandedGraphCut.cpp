
#include "CBandedGraphCut.h"


CBandedGraphCut::CBandedGraphCut( Mat& img, Mat& dsMask, int& levels, const double& gamma, GMMBuilder& gmm )
{
	m_img = img.clone();
	m_dsMask = dsMask.clone();
	m_levels = levels;
	m_gamma = gamma;//50;
    m_lambda = 9*gamma;
	m_width = img.cols;
	m_height = img.rows;
	m_gmm = &gmm;
	run();
}

void CBandedGraphCut::run()
{
	int m = m_levels;
	for(; m>0; m--)
	{
		double t = (double) getTickCount();
				
		vector<vector<cv::Point> > contours;
		vector<Vec4i> hierarchy;
		Mat contourSsMask = m_dsMask.clone();
		findContours(contourSsMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE); 

		if(contours.size()<1)
		{
			resize(m_dsMask, m_mask, cv::Size(m_width, m_height), 0, 0, 0); 
			break;
		}

		int longestContour = 0;
		for(int idx = 0; idx >=0; idx = hierarchy[idx][0]) // next contour on the same hierarchy
		{
			if(contours[idx].size()>contours[longestContour].size())
				longestContour = idx;
		}		

		// build band map from main contour
		vector<cv::Point> mainContour = contours[longestContour];
		Mat dsBandMap = Mat::zeros(m_dsMask.size().height, m_dsMask.size().width, m_dsMask.type());

		for(vector<cv::Point>::iterator itr=mainContour.begin(); itr!=mainContour.end(); itr++)
		{
			dsBandMap.at<uchar>(*itr) = 1;
		}

		if(hierarchy[longestContour][2]>-1) // main contour has holes/children
		{
			int idx = hierarchy[longestContour][2];

			do{
				vector<cv::Point> holeContour = contours[idx];
				for(vector<cv::Point>::iterator itr=holeContour.begin(); itr!=holeContour.end(); itr++)
				{
					dsBandMap.at<uchar>(*itr) = 1;
				}
				idx = hierarchy[idx][0];//next contour at the same level
			}
			while(idx>-1); 
		}

		//Mat usBandMap(cv::Size((double)m_width/pow((double) 2,m-1), (double)m_height/pow((double) 2,m-1)), m_bandMap.type());
		resize(dsBandMap, m_bandMap, cv::Size((double)m_width/pow((double) 2,m-1), (double)m_height/pow((double) 2,m-1)), 0, 0, 0); // nearest neightbour

				
		Mat usMask(cv::Size((double)m_width/pow((double) 2,m-1), (double)m_height/pow((double) 2,m-1)), m_dsMask.type());
		resize(m_dsMask, usMask, cv::Size((double)m_width/pow((double) 2,m-1), (double)m_height/pow((double) 2,m-1)), 0, 0, 0); // nearest neightbour

		if(m!=1)
		{
			dilate(m_bandMap, m_bandMap, Mat());
			//dilate(m_bandMap, m_bandMap, Mat());
			//dilate(m_bandMap, m_bandMap, Mat());
		}
		else
		{
			dilate(m_bandMap, m_bandMap, Mat());
			//dilate(m_bandMap, m_bandMap, Mat());
		}

		//waitKey();

		// up sampling image
		Mat usImage(cv::Size((double)m_width/pow((double) 2,m-1),(double) m_height/pow((double) 2,m-1)), m_img.type());
		m_fgdPb = Mat(usImage.size(), CV_64FC1);
		m_bgdPb = Mat(usImage.size(), CV_64FC1);


		if(m>1)
			resize(m_img, usImage, cv::Size((double)m_width/pow((double) 2,m-1), (double)m_height/pow((double) 2,m-1)), 0, 0, 2); // bicubic
		else
			usImage = m_img.clone();

		cv::Point p;
		for( p.y=0; p.y<usImage.rows; p.y++)
		{
			for( p.x=0; p.x<usImage.cols; p.x++)
			{
				Vec3d color = usImage.at<Vec3d>(p);
				m_fgdPb.at<double>(p) = m_gmm->fgdGMM(color);
				m_bgdPb.at<double>(p) = m_gmm->bgdGMM(color);
			}
		}

		//imshow("usFgdPb",usFgdPb*255);
		//imshow("usBgdPb",usBgdPb*255);
		//waitKey();
		
		// list of points within band
		//vector<cv::Point> bandList;

		for(p.x=0; p.x<m_bandMap.cols; p.x++)
			for(p.y=0; p.y<m_bandMap.rows; p.y++)
				if(m_bandMap.at<uchar>(p))
					m_bandList.push_back(p);

		calcBandedNWeights();

		//GCGraph<double> bandGraph;
		//Mat vtxIdxMat;

		constructBandedGCGraph();

		//imshow("usMask-pre",usMask*255.0);
		estimateBandedSegmentation(usMask);

		t =  ((double)getTickCount()-t)/getTickFrequency();
		cout << " - " << t << " seconds on level #" << m<< " graphcut" << endl;
		//imshow("usBandMap",m_bandMap*255.0);
		//imshow("usMask", usMask*255);
		imshow("dsMask", m_dsMask*255);
		waitKey();

		m_dsMask = usMask.clone();
		m_bandList.clear();

	}
	m_mask = m_dsMask.clone();

}

void CBandedGraphCut::calcBandedNWeights()
{
    const double gammaDivSqrt2 = m_gamma / std::sqrt(2.0f);
    m_leftW.create( m_img.rows, m_img.cols, CV_64FC1 );
    m_upW.create( m_img.rows, m_img.cols, CV_64FC1 );
    m_upleftW.create( m_img.rows, m_img.cols, CV_64FC1 );
    m_uprightW.create( m_img.rows, m_img.cols, CV_64FC1 );

	for(vector<cv::Point>::iterator itr = m_bandList.begin(); itr != m_bandList.end(); itr++ )
    {
		int y = (*itr).y;
		int x = (*itr).x;

		Vec3d color = m_img.at<Vec3d>(y,x);
		if( x>0 ) // left
		{
			Vec3d diff = color - (Vec3d)m_img.at<Vec3d>(y,x-1);
			m_leftW.at<double>(y,x) = m_gamma * exp(-m_beta*diff.dot(diff));
		}

		if( x>0 && y>0 ) // upleft
		{
			Vec3d diff = color - (Vec3d)m_img.at<Vec3d>(y-1,x-1);
			m_upleftW.at<double>(y,x) = m_gamma * exp(-m_beta*diff.dot(diff));
		}

		if( y>0 ) // up
		{
			Vec3d diff = color - (Vec3d)m_img.at<Vec3d>(y-1,x);
			m_upW.at<double>(y,x) = m_gamma * exp(-m_beta*diff.dot(diff));
		}

		if( y>0 && x<m_img.cols-1 ) // upright
		{
			Vec3d diff = color - (Vec3d)m_img.at<Vec3d>(y-1,x+1);
			m_uprightW.at<double>(y,x) = m_gamma * exp(-m_beta*diff.dot(diff));
		}

    }

}


void CBandedGraphCut::constructBandedGCGraph()
{
	vector<vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;

	// extract external and internal contours of band
	Mat contourBandMap = m_bandMap.clone();
	findContours(contourBandMap, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);


	// set new seeds from ex/internal contours of band
	Mat mask = Mat::zeros(m_bandMap.size().height, m_bandMap.size().width, m_bandMap.type());
	///*

	int longestContour = 0;
	for(int idx = 0; idx >=0; idx = hierarchy[idx][0]) // next contour on the same hierarchy
	{
		if(contours[idx].size()>contours[longestContour].size())
			longestContour = idx;
	}	

	imshow("contourBandMap",contourBandMap*255.0);
	//waitKey();


	vector<cv::Point> exContour = contours[longestContour];

	for(vector<cv::Point>::iterator itr=exContour.begin(); itr!=exContour.end(); itr++)
	{
		mask.at<uchar>(*itr) = GC_BGD_MASK;
	}

	// find longest interior contour
	longestContour = -1;
	for(int idx = 0; idx >=0; idx = hierarchy[idx][0])
	{
		if(hierarchy[idx][2]>-1)// start with one which has a child
		{
			for(int hidx = hierarchy[idx][2]; hidx >=0; hidx = hierarchy[hidx][0]) // next contour on the same hierarchy
			{
				if(longestContour == -1)
					longestContour = hidx;
				else if(contours[hidx].size()>contours[longestContour].size())
					longestContour = hidx;
			}	
		}
		
	}
	//cout << "longestContour = " << longestContour << endl;
	if(longestContour>0)
	{
		vector<cv::Point> inContour = contours[longestContour];
		for(vector<cv::Point>::iterator itr=inContour.begin(); itr!=inContour.end(); itr++)
		{
			mask.at<uchar>(*itr) = GC_FGD_MASK;
		}
	}
	//*/

	//imshow("band",(2-mask)/2.0*255.0);
	//waitKey();

	int nNodes = countNonZero(m_bandMap);
	m_vtxIdxMat = Mat::ones(mask.size().height, mask.size().width, CV_32FC1)*(-1);

	// build m_graph within band
    int vtxCount = nNodes;//m_img.cols*m_img.rows,
    int edgeCount = 2*(nNodes*4); //2*(4*m_img.cols*m_img.rows - 3*(m_img.cols + m_img.rows) + 2);

    m_graph.create(vtxCount, edgeCount);

    cv::Point p;

	for(vector<cv::Point>::iterator itr=m_bandList.begin(); itr!=m_bandList.end(); itr++)
	{
		p = (*itr);
		int vtxIdx = m_graph.addVtx();
		m_vtxIdxMat.at<float>(p) = vtxIdx;
	}

	for(vector<cv::Point>::iterator itr=m_bandList.begin(); itr!=m_bandList.end(); itr++)
	{
		p = (*itr);

		// add node
		int vtxIdx = m_vtxIdxMat.at<float>(p);

		//Vec3d color = m_img.at<Vec3d>(p);

		// set t-weights
		double fromSource, toSink;

		if( mask.at<uchar>(p) == GC_BGD_MASK )
		{
			fromSource = 0;
			toSink = m_lambda;
		}
		else if ( mask.at<uchar>(p) == GC_FGD_MASK )
		{
			fromSource = m_lambda;
			toSink = 0;
		}
		else
		{
			fromSource	= -log( m_bgdPb.at<double>(p) +1e-5 );
			toSink		= -log( m_fgdPb.at<double>(p) +1e-5 );
		}

		m_graph.addTermWeights( vtxIdx, fromSource, toSink );

		int vtxid = (int) m_vtxIdxMat.at<float>(p.y, p.x);

		//left
		if( p.x>0 && m_bandMap.at<uchar>(p.y, p.x-1)>0)
		{
			double w = m_leftW.at<double>(p);
			m_graph.addEdges( vtxIdx, (int) m_vtxIdxMat.at<float>(p.y, p.x-1), w, w );
		}
		//upleft
		if( p.y>0 && p.x>0 && m_bandMap.at<uchar>(p.y-1, p.x-1)>0)
		{
			double w = m_upleftW.at<double>(p);
			m_graph.addEdges( vtxIdx, (int)m_vtxIdxMat.at<float>(p.y-1, p.x-1), w, w );
		}
		//up
		if( p.y>0 && m_bandMap.at<uchar>(p.y-1, p.x)>0)
		{
			double w = m_upW.at<double>(p);
			m_graph.addEdges( vtxIdx, (int) m_vtxIdxMat.at<float>(p.y-1, p.x), w, w );
		}
		//upright
		if( p.x<m_bandMap.size().width-1 && p.y>0 && m_bandMap.at<uchar>(p.y-1, p.x+1)>0)
		{
			double w = m_uprightW.at<double>(p);
			m_graph.addEdges( vtxIdx, (int) m_vtxIdxMat.at<float>(p.y-1, p.x+1), w, w );
		}

	}

}


/*
  Estimate segmentation in band using MaxFlow algorithm
*/
void CBandedGraphCut::estimateBandedSegmentation(Mat& mask)
{
    m_graph.maxFlow();

	cv::Point p;

	for(vector<cv::Point>::iterator itr=m_bandList.begin(); itr!=m_bandList.end(); itr++)
	{
		p = *itr;
		// update nodes within band
		if(m_vtxIdxMat.at<float>(p)>-1)
		{
			if( m_graph.inSourceSegment( (int) m_vtxIdxMat.at<float>(p) /*vertex index*/ ) )
				mask.at<uchar>(p) = 1;
			else
				mask.at<uchar>(p) = 0;
		}
	}
}

Mat CBandedGraphCut::getMask()
{
	return m_mask;
}