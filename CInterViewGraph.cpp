
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "gmmbuilder.h"
#include "gcgraph.hpp"
#include "graphcutseg.hpp"
#include "CInterViewGraph.h"
#include <vector>
#include <algorithm>

using namespace cv;

CInterViewGraph::CInterViewGraph(struct multiViewData* mvd, double gamma=50):m_gamma(gamma)
{
	m_mvd = mvd;
	m_lambda = m_gamma*100;//9
	m_kx = 5;
	m_ky = 5;
	m_sigma = 3.0;
	spAdjMat();
	spIntraAdjMat();
	run();
}


//! Member function that drives graph construction and MRF inference
void CInterViewGraph::run()
{
	estimateMotionPrior();
	constructGraph();
	calcPixNWeight();
	addPixelNodesEdges();
	addSuperpixelNodes();
	addSuperpixelEdges();
	addSppixelEdges();
	inference();
}


//! Member function that estimates motion driven labeling prior
void CInterViewGraph::estimateMotionPrior()
{

	
	Mat gausskernel = getGaussianKernel( m_kx, m_sigma, CV_64F );


	//Mat directionalGaussian;
	//makeRFSfilters( directionalGaussian, 0);

	if(!m_mvd->preMaskVec.empty())
	{
		for( int i=0; i<m_mvd->dsImg64Vec.size(); i++)
		{
			Mat fgMotPriMat = Mat::zeros(m_height,m_width,CV_64FC1);
			Mat bgMotPriMat = Mat::zeros(m_height,m_width,CV_64FC1);

			//cout << "m_mvd->preMaskVec[i].depth = " << m_mvd->preMaskVec[i].depth() << endl;
			cv::Point p;
			for( p.y=0; p.y<m_height; p.y++)
			{
				for( p.x=0; p.x<m_width; p.x++)
				{
					Point curPos( p.x+m_mvd->flowVec[i].at<Point2f>(p).x, p.y+m_mvd->flowVec[i].at<Point2f>(p).y);
					if(curPos.x>=0 && curPos.x<m_width && curPos.y>=0 && curPos.y<m_height)
					{
						motionDiffusion( i, p, curPos, fgMotPriMat, bgMotPriMat, gausskernel);
						//motionDiffusion( i, p, curPos, fgMotPriMat, bgMotPriMat, directionalGaussian);
					}
				}
			}
			
			vector<Mat> motPri;
			motPri.push_back(fgMotPriMat.clone());
			motPri.push_back(bgMotPriMat.clone());
			m_motionPriorVec.push_back(motPri);

		}
	}
}

//! Member function that diffuses the motion prior
/*!
    \param curPos the estimated position based on optical flow.
    \param fgMotPriMat the diffused motion prior of FG
    \param fgMotPriMat the diffused motion prior of BG
    \param gausskernel Gaussian kernel
*/
void CInterViewGraph::motionDiffusion( int i, Point& prePos, Point& curPos, Mat& fgMotPriMat, 
									  Mat& bgMotPriMat, Mat& gausskernel)
{
	Vec3d diff = m_mvd->preDsImg64Vec[i].at<Vec3d>(prePos)-m_mvd->dsImg64Vec[i].at<Vec3d>(curPos);
	double clrDistW = exp(-(abs(diff[0])+abs(diff[1])+abs(diff[2]))/50.0);
	//double clrDistW = exp(-(diff.dot(diff))/100.0);

	if(m_mvd->preMaskVec[i].at<uchar>(prePos)==GC_FGD_MASK)
		fgMotPriMat.at<double>(curPos) += clrDistW;
	else if(m_mvd->preMaskVec[i].at<uchar>(prePos)==GC_BGD_MASK) 
		bgMotPriMat.at<double>(curPos) += clrDistW;  

	/*
	for( int dy=std::max((double) 0,curPos.y-(m_ky-1)*0.5); dy<std::min((double) m_height-1,curPos.y+(m_ky-1)*0.5); dy++)
	{
		for( int dx=std::max((double) 0,curPos.x-(m_kx-1)*0.5); dx<std::min((double) m_width-1,curPos.x+(m_kx-1)*0.5); dx++)
		{
			
			Vec3d diff = m_mvd->preDsImg64Vec[i].at<Vec3d>(prePos)-m_mvd->dsImg64Vec[i].at<Vec3d>(curPos);
			double clrDistW = exp(-(abs(diff[0])+abs(diff[1])+abs(diff[2]))/10.0);

			if(m_mvd->preMaskVec[i].at<uchar>(prePos)==GC_FGD_MASK)
				fgMotPriMat.at<double>(dy,dx) += clrDistW*gausskernel.at<double>(dy-(curPos.y-(m_ky-1)*0.5), dx-(curPos.x-(m_kx-1)*0.5));
			else if(m_mvd->preMaskVec[i].at<uchar>(prePos)==GC_BGD_MASK) 
				bgMotPriMat.at<double>(dy,dx) += clrDistW*gausskernel.at<double>(dy-(curPos.y-(m_ky-1)*0.5), dx-(curPos.x-(m_kx-1)*0.5));  

		}
	}//*/
	
}





//! Member function that constructs graph
void CInterViewGraph::constructGraph()
{

	//cout << "ok2"<< endl;
	m_tolSPCnt = 0;
	for( int i=0; i<m_mvd->dsImg64Vec.size(); i++)
		m_tolSPCnt += m_mvd->lCntVec[i];

	m_vtxCount = m_mvd->dsImg64Vec.size()*m_width*m_height + m_tolSPCnt;
	if(m_mvd->dsImg64Vec.size()>2)
		m_edgeCount = (m_mvd->dsImg64Vec.size()*(4*m_width*m_height - 3*(m_width + m_height) + 2) //intra frame pixel links
		+ m_mvd->dsImg64Vec.size()*m_width*m_height //inter pixel-sp links
		+ m_spEdgeCnt //inter sp links
		+ m_spIntraEdgeCnt)*2; //intra sp links
	else
	{
		int circSpEdgeCnt = 0;
		//int k = m_mvd->dsImg64Vec.size()-1;
		//imshow("adj",m_spAdjMatVec[0]*255);
		//waitKey();
		for(int k=0; k<m_mvd->dsImg64Vec.size()-1; k++)
		{
			for(int i=0;i<m_spAdjMatVec[k].rows;i++)
			{
				for(int j=0;j<m_spAdjMatVec[k].cols;j++)
				{
					if(m_spAdjMatVec[k].at<uchar>(i, j)>0) 
						circSpEdgeCnt++;
				}
			}
		}
		m_edgeCount = (m_mvd->dsImg64Vec.size()*(4*m_width*m_height - 3*(m_width + m_height) + 2) + m_mvd->dsImg64Vec.size()*m_width*m_height + m_spEdgeCnt + m_spIntraEdgeCnt-circSpEdgeCnt)*2;
	}
	cout << "m_vtxCount = "<< m_vtxCount << endl;
	cout << "m_edgeCount = "<< m_edgeCount << endl;
	m_graph.create(m_vtxCount, m_edgeCount);
	//m_graph.create(1057547, 10506610);

	cout <<"m_spEdgeCnt=" << m_spEdgeCnt << endl;
}

//! Member function that computes pairwise terms of each pixel
void CInterViewGraph::calcPixNWeight()
{
	//cout << "OK0" << endl;
	for( int i=0; i<m_mvd->dsImg64Vec.size(); i++)
	{
		//cout << "OK1" << endl;
		double beta = calcBeta( m_mvd->dsImg64Vec[i] );
		//cout << "OK2" << endl;
		Mat leftW, upleftW, upW, uprightW;
		calcNWeights( m_mvd->dsImg64Vec[i], leftW, upleftW, upW, uprightW, beta, m_gamma );

		m_betaVec.push_back(beta);
		m_leftWVec.push_back(leftW);
		m_upleftWVec.push_back(upleftW);
		m_upWVec.push_back(upW);
		m_uprightWVec.push_back(uprightW);
	}
}

//! Member function that computes superpixel adjacency matrix based on optical flow motion vectors
void CInterViewGraph::spOptAdjMat()
{

}


//! Member function that computes intra-frame superpixel adjacency matrix based on spatial relationship
void CInterViewGraph::spIntraAdjMat()
{
	m_spIntraEdgeCnt = 0;
	for(int i=0; i<m_mvd->dsSpVec.size(); i++)
	{
		Mat intraAdjMap = Mat::zeros(m_mvd->lCntVec[i],m_mvd->lCntVec[i],CV_8UC1);
		Mat spMap = m_mvd->dsSpVec[i];

		cv::Point p;
		for(p.y=0; p.y<spMap.rows; p.y++)
		{
			int preLabel = spMap.at<float>(p.y,0);
			for( p.x=0; p.x<spMap.cols; p.x++)
			{
				int curLabel = spMap.at<float>(p);
				if(preLabel!=curLabel)
				{
					if(intraAdjMap.at<uchar>(curLabel,preLabel)!=1)
					{
						intraAdjMap.at<uchar>(curLabel,preLabel) = 1;
						intraAdjMap.at<uchar>(preLabel,curLabel) = 1;
						m_spIntraEdgeCnt++;
					}
					preLabel = curLabel;
				}
			}
		}


		for( p.x=0; p.x<spMap.cols; p.x++) 
		{
			int preLabel = spMap.at<float>(0,p.x);
			for(p.y=0; p.y<spMap.rows; p.y++)
			{
				int curLabel = spMap.at<float>(p);
				if(preLabel!=curLabel)
				{
					if(intraAdjMap.at<uchar>(curLabel,preLabel)!=1)
					{
						intraAdjMap.at<uchar>(curLabel,preLabel) = 1;
						intraAdjMap.at<uchar>(preLabel,curLabel) = 1;
						m_spIntraEdgeCnt++;
					}
					preLabel = curLabel;
				}
			}
		}
		m_spIntraAdjMatVec.push_back(intraAdjMap.clone());
	}
	//cout << "ok-1"<< endl;
}
//! Member function that computes superpixel adjacency matrix based on sparse feature matching
void CInterViewGraph::spAdjMat()
{
	m_width = m_mvd->dsImg64Vec[0].cols;
	m_height = m_mvd->dsImg64Vec[0].rows;
	//int imgSize = m_height*m_width;

	m_spEdgeCnt = 0;

	// transform feature mapping into superpixel mapping
	for( int j=0; j<m_mvd->dsImg64Vec.size()-1; j++)
	{

		Mat fundamMat = m_mvd->fundamMatVec[j].clone();
		Mat spAdjMat = Mat::zeros(m_mvd->lCntVec[j],m_mvd->lCntVec[j+1], CV_8UC1);
		//Mat spAdjDistMat = Mat::zeros(m_mvd->lCntVec[j],m_mvd->lCntVec[j+1], CV_32FC1);

		//Mat pixAdjMap = Mat::zeros(imgSize,imgSize, CV_32FC1);

		///*
		// larger-index cam lies on the left
		for( int i=0; i<m_mvd->lCntVec[j+1]; i++)
		{
			// estimate upsampled sp centres
			cv::Point2f dsCentSp = m_mvd->centDsSPViewVec[j+1][i];
			cv::Point2f centSp; 
			centSp.x = (float) dsCentSp.x/(float)m_width*(float)m_mvd->img64Vec[j+1].cols;
			centSp.y = (float) dsCentSp.y/(float)m_height*(float)m_mvd->img64Vec[j+1].rows;
			
			std::vector<cv::Point2f> selPointsLeft;
			selPointsLeft.push_back(centSp);

			std::vector<cv::Vec3f> linesRight;
			cv::computeCorrespondEpilines(
				cv::Mat(selPointsLeft), // image points
				1,                      // in left image 
				m_mvd->fundamMatVec[j],      // F matrix
				linesRight);             // vector of epipolar lines

			//vector<int> spEpipolarLineVec;
			cv::Point p;
			double min_dist = 1000000;
			int min_label;
			bool flag = false;
			int pre_label = -1;
			for(p.x=0; p.x<m_mvd->img64Vec[j].cols; p.x++)
			{
				p.y = -(linesRight[0][2]+linesRight[0][0]*p.x)/(float)linesRight[0][1];
				
				if(p.y>=0 && p.y<m_mvd->img64Vec[j+1].rows)
				{			
					//spEpipolarLineVec.push_back(m_mvd->spVec[j].at<float>(p));
					//cout << "p.y = " << (int)p.y << endl;
					//cout << "p.x = " << (int)p.x << endl;
					//int labelRight = m_mvd->spVec[j].at<float>((int)p.y,(int)p.x);
					int labelRight = m_mvd->dsSpVec[j].at<float>((int)((float)p.y/(float)m_mvd->img64Vec[0].rows*(float)m_height),(int)((float)p.x/(float)m_mvd->img64Vec[0].cols*(float)m_width));


					if(pre_label!=labelRight)
					{
						pre_label = labelRight;

						double FGProb1 = m_mvd->SPProbVec[1][j][labelRight]/
							(m_mvd->SPProbVec[1][j][labelRight]+m_mvd->SPProbVec[0][j][labelRight]);
						double FGProb2 = m_mvd->SPProbVec[1][j+1][i]/
							(m_mvd->SPProbVec[1][j+1][i]+m_mvd->SPProbVec[0][j+1][i]);

						//double dist = abs(FGProb1-FGProb2)*computeSPHistDist(j,j+1,labelRight,i);
						double dist = exp(2.0*pow(FGProb1-FGProb2,2.0))*min(computeSPHistKLDiv(j,j+1,labelRight,i), computeSPHistKLDiv(j+1,j,i,labelRight));
						//double dist = abs(FGProb1-FGProb2)*computeSPHistKLDiv(j,j+1,labelRight,i);


						//cout << "dist = " << dist << endl;
						if(min_dist>dist)// && dist<5) 
						{
							min_dist = dist;
							min_label = labelRight;
							flag = true;
						}
					}

				}

			}

			if(flag)
			{
				if(spAdjMat.at<uchar>(min_label, i) > 0)
					continue;
				else
				{
					spAdjMat.at<uchar>(min_label, i) = 1;
					//spAdjDistMat.at<float>(min_label, i) = min_dist;
					m_spEdgeCnt++;
				}
			}

		}//*/

		// find epipolar line on left image
		for( int i=0; i<m_mvd->lCntVec[j]; i++)
		{
			// estimate upsampled sp centres
			cv::Point2f dsCentSp = m_mvd->centDsSPViewVec[j][i];
			cv::Point2f centSp; 
			centSp.x = (float) dsCentSp.x/(float)m_width*(float)m_mvd->img64Vec[j].cols;
			centSp.y = (float) dsCentSp.y/(float)m_height*(float)m_mvd->img64Vec[j].rows;
			
			std::vector<cv::Point2f> selPointsRight;
			selPointsRight.push_back(centSp);

			std::vector<cv::Vec3f> linesLeft;
			cv::computeCorrespondEpilines(
				cv::Mat(selPointsRight), // image points
				2,                      // in right image 
				m_mvd->fundamMatVec[j],      // F matrix
				linesLeft);             // vector of epipolar lines

			//vector<int> spEpipolarLineVec;
			cv::Point p;
			double min_dist = 1000000;
			int min_label;
			bool flag = false;
			int pre_label = -1;
			for(p.x=0; p.x<m_mvd->img64Vec[j+1].cols; p.x++)
			{
				p.y = -(linesLeft[0][2]+linesLeft[0][0]*p.x)/(float)linesLeft[0][1];
				
				if(p.y>=0 && p.y<m_mvd->img64Vec[j+1].rows)
				{			
					//spEpipolarLineVec.push_back(m_mvd->spVec[j].at<float>(p));
					//cout << "p.y = " << (int)p.y << endl;
					//cout << "p.x = " << (int)p.x << endl;
					int labelLeft = m_mvd->dsSpVec[j+1].at<float>((int)((float)p.y/(float)m_mvd->img64Vec[j+1].rows*(float)m_height),(int)((float)p.x/(float)m_mvd->img64Vec[j+1].cols*(float)m_width));

					if(pre_label!=labelLeft)
					{
						pre_label = labelLeft;

						double FGProb1 = m_mvd->SPProbVec[1][j][i]/
							(m_mvd->SPProbVec[1][j][i]+m_mvd->SPProbVec[0][j][i]);
						double FGProb2 = m_mvd->SPProbVec[1][j+1][labelLeft]/
							(m_mvd->SPProbVec[1][j+1][labelLeft]+m_mvd->SPProbVec[0][j+1][labelLeft]);

						//double dist = abs(FGProb1-FGProb2)*computeSPHistDist(j,[j+1],i,labelLeft);
						double dist = exp(2.0*pow(FGProb1-FGProb2,2.0))*min(computeSPHistKLDiv(j,j+1,i,labelLeft),computeSPHistKLDiv(j+1,j,labelLeft,i));
						//double dist = abs(FGProb1-FGProb2)*computeSPHistKLDiv(j,j+1,i,labelLeft);

						//cout << "dist = " << dist << endl;
						if(min_dist>dist)// && dist<5) 
						{
							min_dist = dist;
							min_label = labelLeft;
							flag = true;
						}
					}
				}
			}

			if(flag)
			{
				if(spAdjMat.at<uchar>(i, min_label) > 0)
					continue;
				else
				{
					spAdjMat.at<uchar>(i, min_label) = 1;
					//spAdjDistMat.at<float>(i, min_label) = min_dist;
					m_spEdgeCnt++;
				}
			}

		}

		m_spAdjMatVec.push_back(spAdjMat.clone());
		//m_spAdjDistMatVec.push_back(spAdjDistMat.clone());
		
		/* visualize sp matching
		Mat visSPMatch(m_mvd->dsSpVec[0].rows, m_mvd->dsSpVec[0].cols*2, CV_8UC3);
		m_mvd->dsImgVec[j].convertTo(visSPMatch(Rect(0,0,m_mvd->dsSpVec[0].cols,m_mvd->dsSpVec[0].rows)), CV_8UC3, 1, 0);
		m_mvd->dsImgVec[j+1].convertTo(visSPMatch(Rect(m_mvd->dsSpVec[0].cols,0,m_mvd->dsSpVec[0].cols,m_mvd->dsSpVec[0].rows)), CV_8UC3, 1, 0);

		for(int r=0; r<spAdjMat.rows; r++)
		{
			for(int c=0; c<spAdjMat.cols; c++)
			{
				//imshow("mask",(2-m_mvd->priorMaskVec[j])*255);
				//waitKey();
				if(spAdjMat.at<uchar>(r, c) > 0 && m_mvd->priorMaskVec[j].at<uchar>(m_mvd->centDsSPViewVec[j][r])==GC_FGD_MASK)
					//m_mvd->priorMaskVec[j].at<uchar>(m_mvd->centDsSPViewVec[j][r])==GC_BGD_MASK)	
				{
					cv::Point p1 = m_mvd->centDsSPViewVec[j][r];
					cv::Point p2 = m_mvd->centDsSPViewVec[j+1][c];

					p2.x += m_mvd->dsSpVec[0].cols;
					RNG& rng = theRNG();
					
					line(visSPMatch, p1, p2, Scalar(rng(256), rng(256), rng(256)));
					imshow("visSPMatch",visSPMatch);
					waitKey();
				}
			}
		}

		//imshow("visSPMatch",visSPMatch);
		//waitKey();
		//*/
	}

	//cout << "spAdjMat.at<uchar>(label_1, label_2) = " << (double)spAdjMatVec[0].at<uchar>(0, 0) << endl;

	if(m_mvd->dsImg64Vec.size()>2)
	{
		int j = m_mvd->dsImg64Vec.size()-1;
		Mat fundamMat = m_mvd->fundamMatVec[j].clone();
		Mat spAdjMat = Mat::zeros(m_mvd->lCntVec[j],m_mvd->lCntVec[0], CV_8UC1);
		//Mat spAdjDistMat = Mat::zeros(m_mvd->lCntVec[j],m_mvd->lCntVec[0], CV_32FC1);

		///*
		// #0 cam lies on the left
		for( int i=0; i<m_mvd->lCntVec[0]; i++)
		{
			// estimate upsampled sp centres
			cv::Point2f dsCentSp = m_mvd->centDsSPViewVec[0][i];
			cv::Point2f centSp; 
			centSp.x = dsCentSp.x/m_width*m_mvd->img64Vec[0].cols;
			centSp.y = dsCentSp.y/m_height*m_mvd->img64Vec[0].rows;
			
			std::vector<cv::Point2f> selPointsLeft;
			selPointsLeft.push_back(centSp);

			std::vector<cv::Vec3f> linesRight;
			cv::computeCorrespondEpilines(
				cv::Mat(selPointsLeft), // image points
				1,                      // in left image 
				m_mvd->fundamMatVec[j],      // F matrix
				linesRight);             // vector of epipolar lines

			//vector<int> spEpipolarLineVec;
			cv::Point p;
			double min_dist = 1000000;
			int min_label;
			int pre_label = -1;
			bool flag = false;

			for(p.x=0; p.x<m_mvd->img64Vec[0].cols; p.x++)
			{
				p.y = -(linesRight[0][2]+linesRight[0][0]*p.x)/linesRight[0][1];
				if(p.y>=0 && p.y<m_mvd->img64Vec[0].rows)
				{
					//spEpipolarLineVec.push_back(m_mvd->spVec[j].at<float>(p));

					//int labelRight = m_mvd->spVec[j].at<float>((int)p.y,(int)p.x);
					int labelRight = m_mvd->dsSpVec[j].at<float>((int)((float)p.y/(float)m_mvd->img64Vec[0].rows*(float)m_height),(int)((float)p.x/(float)m_mvd->img64Vec[0].cols*(float)m_width));


					if(pre_label==labelRight) continue;
					pre_label = labelRight;

					double FGProb1 = m_mvd->SPProbVec[1][j][labelRight]/
						(m_mvd->SPProbVec[1][j][labelRight]+m_mvd->SPProbVec[0][j][labelRight]);
					double FGProb2 = m_mvd->SPProbVec[1][0][i]/
						(m_mvd->SPProbVec[1][0][i]+m_mvd->SPProbVec[0][0][i]);

					//double dist = abs(FGProb1-FGProb2)*computeSPHistDist(j,0,labelRight,i);
					double dist = exp(2.0*pow(FGProb1-FGProb2,2.0))*min(computeSPHistKLDiv(j,0,labelRight,i),computeSPHistKLDiv(0,j,i,labelRight));
					if(min_dist>dist && dist<2) 
					{
						min_dist = dist;
						min_label = labelRight;
						flag = true;
					}
				}

			}

			if(flag)
			{
				if(spAdjMat.at<uchar>(min_label, i) > 0)
					continue;
				else
				{
					spAdjMat.at<uchar>(min_label, i) = 1;
					//spAdjDistMat.at<float>(min_label, i) = min_dist;
					m_spEdgeCnt++;

				}
			}
		}//*/


		// find epipolar line on left image
		for( int i=0; i<m_mvd->lCntVec[j]; i++)
		{
			// estimate upsampled sp centres
			cv::Point2f dsCentSp = m_mvd->centDsSPViewVec[j][i];
			cv::Point2f centSp; 
			centSp.x = (float) dsCentSp.x/(float)m_width*(float)m_mvd->img64Vec[j].cols;
			centSp.y = (float) dsCentSp.y/(float)m_height*(float)m_mvd->img64Vec[j].rows;
			
			std::vector<cv::Point2f> selPointsRight;
			selPointsRight.push_back(centSp);

			std::vector<cv::Vec3f> linesLeft;
			cv::computeCorrespondEpilines(
				cv::Mat(selPointsRight), // image points
				2,                      // in right image 
				m_mvd->fundamMatVec[j],      // F matrix
				linesLeft);             // vector of epipolar lines

			//vector<int> spEpipolarLineVec;
			cv::Point p;
			double min_dist = 1000000;
			int min_label;
			
			int pre_label = -1;
			bool flag = false;
			for(p.x=0; p.x<m_mvd->img64Vec[0].cols; p.x++)
			{
				p.y = -(linesLeft[0][2]+linesLeft[0][0]*p.x)/(float)linesLeft[0][1];
				
				if(p.y>=0 && p.y<m_mvd->img64Vec[0].rows)
				{			
					//spEpipolarLineVec.push_back(m_mvd->spVec[j].at<float>(p));
					//cout << "p.y = " << (int)p.y << endl;
					//cout << "p.x = " << (int)p.x << endl;
					//int labelLeft = m_mvd->spVec[0].at<float>((int)p.y,(int)p.x);
					int labelLeft = m_mvd->dsSpVec[0].at<float>((int)((float)p.y/(float)m_mvd->img64Vec[j+1].rows*(float)m_height),(int)((float)p.x/(float)m_mvd->img64Vec[j+1].cols*(float)m_width));


					if(pre_label!=labelLeft)
					{
						pre_label = labelLeft;

						double FGProb1 = m_mvd->SPProbVec[1][j][i]/
							(m_mvd->SPProbVec[1][j][i]+m_mvd->SPProbVec[0][j][i]);
						double FGProb2 = m_mvd->SPProbVec[1][0][labelLeft]/
							(m_mvd->SPProbVec[1][0][labelLeft]+m_mvd->SPProbVec[0][0][labelLeft]);

						//double dist = abs(FGProb1-FGProb2)*computeSPHistDist(j,0,i,labelLeft);
						double dist = exp(2.0*pow(FGProb1-FGProb2,2.0))*min(computeSPHistKLDiv(j,0,i,labelLeft),computeSPHistKLDiv(0,j,labelLeft,i));

						//cout << "dist = " << dist << endl;
						if(min_dist>dist && dist<2) 
						{
							min_dist = dist;
							min_label = labelLeft;
							flag = true;
						}
					}

				}

			}

			if(flag)
			{
				if(spAdjMat.at<uchar>(i, min_label) > 0)
					continue;
				else
				{
					spAdjMat.at<uchar>(i, min_label) = 1;
					//spAdjDistMat.at<float>(i, min_label) = min_dist;
					m_spEdgeCnt++;
				}
			}

		}

	    /*
		Mat visSPMatch(m_mvd->dsSpVec[0].rows, m_mvd->dsSpVec[0].cols*2, CV_8UC3);
		m_mvd->dsImgVec[j].convertTo(visSPMatch(Rect(0,0,m_mvd->dsSpVec[0].cols,m_mvd->dsSpVec[0].rows)), CV_8UC3, 1, 0);
		m_mvd->dsImgVec[0].convertTo(visSPMatch(Rect(m_mvd->dsSpVec[0].cols,0,m_mvd->dsSpVec[0].cols,m_mvd->dsSpVec[0].rows)), CV_8UC3, 1, 0);

		for(int r=0; r<spAdjMat.rows; r++)
		{
			for(int c=0; c<spAdjMat.cols; c++)
			{
				if(spAdjMat.at<uchar>(r, c) > 0 && 
					m_mvd->priorMaskVec[0].at<uchar>(m_mvd->centDsSPViewVec[0][c])==GC_FGD_MASK)
				{
					cv::Point p1 = m_mvd->centDsSPViewVec[j][r];
					cv::Point p2 = m_mvd->centDsSPViewVec[0][c];

					p2.x += m_mvd->dsSpVec[0].cols;
					RNG& rng = theRNG();
					
					line(visSPMatch, p1, p2, Scalar(rng(256), rng(256), rng(256)));
					imshow("visSPMatch",visSPMatch);
					waitKey();
				}
			}
		}//*/

		m_spAdjMatVec.push_back(spAdjMat.clone());
		//m_spAdjDistMatVec.push_back(spAdjDistMat.clone());
	}

}

//! Member function that returns a list of matched features per superpixel pair
/*!
    \return a list of matched features per superpixel pair for visualisation.
*/
vector<vector<char> >& CInterViewGraph::getSPMatchesMask()
{
	return matchesMaskVec;
}

//! Member function that adds pixels as graph nodes and edges between pixels
void CInterViewGraph::addPixelNodesEdges()
{

	cv::Point p;
	for( int i=0; i<m_mvd->dsImg64Vec.size(); i++)
	{
		Mat fgdPb(m_mvd->dsImg64Vec[0].size(), CV_64FC1);
		Mat bgdPb(m_mvd->dsImg64Vec[0].size(), CV_64FC1);
		for( p.y=0; p.y<m_mvd->dsImg64Vec[0].rows; p.y++)
		{
			for( p.x=0; p.x<m_mvd->dsImg64Vec[0].cols; p.x++)
			{
				Vec3d color = m_mvd->dsImg64Vec[i].at<Vec3d>(p);
				//fgdPb.at<double>(p) = m_mvd->gmmVec[i].fgdGMM(color);
				//bgdPb.at<double>(p) = m_mvd->gmmVec[i].bgdGMM(color);
				double fgP = 0, bgP = 0;
				for( int k=0; k<m_mvd->dsImg64Vec.size(); k++)
				{
					fgP += m_mvd->gmmVec[k].fgdGMM(color);
					bgP += m_mvd->gmmVec[k].bgdGMM(color);
				}
				fgdPb.at<double>(p) = fgP/(double)m_mvd->dsImg64Vec.size();
				bgdPb.at<double>(p) = bgP/(double)m_mvd->dsImg64Vec.size();
			}
		}
		string frmid;
		stringstream sscvt;
		sscvt << i;
		frmid = sscvt.str();

		//imshow("Color fgProbMap "+frmid, fgdPb*255);
		//imshow("Color bgProbMap "+frmid, bgdPb*255);	
		m_bgdPbVec.push_back(bgdPb);
		m_fgdPbVec.push_back(fgdPb);
	}

	double clrWeight = 0.7;
	double dispWeight = 1.0;
	double dispStd = 10.0;
	double wnWeight = 1.0; //neighbourhood consistency weight
	int winSize = 5;//15

	for(int i=0; i<m_mvd->dsImg64Vec.size(); i++)
	{
		//imshow("vis", m_mvd->textonLikelihoodVec[0][i]*255.0);
		//imshow("prior", 255*(2-m_mvd->priorMaskVec[i]));
		//waitKey();
		stringstream cvt;
		string camID;
		cvt << i;
		camID = cvt.str();
		imshow("FG motion prior "+camID, 255*m_motionPriorVec[i][0]);
		//imshow("BG motion prior "+camID, 255*m_motionPriorVec[i][1]);
		//imshow("FG motion prior "+camID, 255*m_motionPriorVec[i][0]/(m_motionPriorVec[i][0]+m_motionPriorVec[i][1]));
		//imshow("BG motion prior "+camID, 255*m_motionPriorVec[i][1]/(m_motionPriorVec[i][0]+m_motionPriorVec[i][1]));
		//waitKey();

		for( p.y = 0; p.y < m_height; p.y++ )
		{
			for( p.x = 0; p.x < m_width; p.x++)
			{
				// add node
				int vtxIdx = m_graph.addVtx();
				//Vec3f color = dsImgVec[i].at<Vec3f>(p);

				// set t-weights
				double fromSource, toSink;

				if(((int)m_mvd->priorMaskVec[i].at<uchar>(p))==GC_FGD_MASK)
				{
					fromSource = m_lambda;
					toSink = 0;
				}
				else if(((int)m_mvd->priorMaskVec[i].at<uchar>(p))==GC_BGD_MASK)
				{
					fromSource = 0;
					toSink = m_lambda;
				}
				else
				{
					/*
					fromSource	= -clrWeight*log( m_bgdPbVec[i].at<double>(p) + 1e-10);
					toSink		= -clrWeight*log( m_fgdPbVec[i].at<double>(p) + 1e-10 );

					fromSource += -(1.0-clrWeight)*log(m_mvd->textonLikelihoodVec[0][i].at<float>(p) + 1e-10);
					toSink     += -(1.0-clrWeight)*log(m_mvd->textonLikelihoodVec[1][i].at<float>(p) + 1e-10);//*/

					fromSource = 0;
					toSink     = 0;

					cv::Point pw;
					double wnFromSource = 0;
					double wnToSink = 0;
					double wnCnt = 0;
					for(pw.y=max(0,(int)(p.y-(winSize-1)*0.5)); pw.y<min(m_height,(int)(p.y+(winSize-1)*0.5)); pw.y++)
					{
						for(pw.x=max(0,(int)(p.x-(winSize-1)*0.5)); pw.x<min(m_width,(int)(p.x+(winSize-1)*0.5)); pw.x++)
						{
							Vec3d diff = m_mvd->dsImg64Vec[i].at<Vec3d>(p) - m_mvd->dsImg64Vec[i].at<Vec3d>(pw);

							//double clrDistW = exp(-m_betaVec[i]*diff.dot(diff));
							//double clrDistW = exp(-abs(sum(diff).val[0])/10.0);
							double clrDistW = exp(-(abs(diff[0])+abs(diff[1])+abs(diff[2]))/50.0);

							wnFromSource += -clrWeight*log( m_bgdPbVec[i].at<double>(pw) + 1e-10) * clrDistW;
							wnToSink	 += -clrWeight*log( m_fgdPbVec[i].at<double>(pw) + 1e-10) * clrDistW;

							wnFromSource += -(1.0-clrWeight)*log(m_mvd->textonLikelihoodVec[0][i].at<float>(pw) + 1e-10) * clrDistW;
							wnToSink     += -(1.0-clrWeight)*log(m_mvd->textonLikelihoodVec[1][i].at<float>(pw) + 1e-10) * clrDistW;
						
							wnCnt += clrDistW;
						}
					}

					fromSource	+= wnWeight*wnFromSource/wnCnt;
					toSink		+= wnWeight*wnToSink/wnCnt;

					///*


					if(!m_mvd->preMaskVec.empty())
					{
						double mtFGPrior = min(1.0, m_motionPriorVec[i][0].at<double>(p));
						double mtBGPrior = min(1.0, m_motionPriorVec[i][1].at<double>(p));
						//double mtFGPrior = m_motionPriorVec[i][0].at<double>(p)/(m_motionPriorVec[i][0].at<double>(p)+m_motionPriorVec[i][1].at<double>(p));
						//double mtBGPrior = m_motionPriorVec[i][1].at<double>(p)/(m_motionPriorVec[i][0].at<double>(p)+m_motionPriorVec[i][1].at<double>(p));
						toSink     += -0.5*log( mtFGPrior + 1e-10 );
						fromSource += -0.5*log( mtBGPrior + 1e-10 );

					}//*/
				}

				m_graph.addTermWeights( vtxIdx, fromSource, toSink );


				int dispIndx = i>0 ? i-1 : m_mvd->dsImg64Vec.size()-1;

				Mat dispMap;
				int dp1;
				if(m_mvd->dsImg64Vec.size()>2 || i>0)
				{
					dispMap = m_mvd->disparityMapVec[dispIndx];
					dp1 = dispMap.at<uchar>(p);
				}

				// set n-weights
				if( p.x>0 )
				{
					// n-weight from colour distance
					double w = m_leftWVec[i].at<double>(p);

					if(m_mvd->dsImg64Vec.size()>2 || i>0)
					{
						// n-weight from disparity
						int dp2 = dispMap.at<uchar>(p.y,p.x-1);
						//double dpw = dispWeight*exp(0.05*(double)abs(dp1-dp2));
						double dpw = dispWeight*exp(-(double)pow(dp1-dp2,2.0)/(2.0*pow(dispStd,2.0)));
						w += dpw;
					}

					m_graph.addEdges( vtxIdx, vtxIdx-1, w, w );
				}
				if( p.x>0 && p.y>0 )
				{
					double w = m_upleftWVec[i].at<double>(p);

					if(m_mvd->dsImg64Vec.size()>2 || i>0)
					{
						int dp2 = dispMap.at<uchar>(p.y-1,p.x-1);
						double dpw = dispWeight*exp(-(double)pow(dp1-dp2,2.0)/(2.0*pow(dispStd,2.0)));
						w += dpw;
					}

					m_graph.addEdges( vtxIdx, vtxIdx-m_width-1, w, w );
				}
				if( p.y>0 )
				{
					double w = m_upWVec[i].at<double>(p);

					if(m_mvd->dsImg64Vec.size()>2 || i>0)
					{
						int dp2 = dispMap.at<uchar>(p.y-1,p.x);
						double dpw = dispWeight*exp(-(double)pow(dp1-dp2,2.0)/(2.0*pow(dispStd,2.0)));
						w += dpw;
					}

					m_graph.addEdges( vtxIdx, vtxIdx-m_width, w, w );
				}
				if( p.x<m_width-1 && p.y>0 )
				{
					double w = m_uprightWVec[i].at<double>(p);

					if(m_mvd->dsImg64Vec.size()>2 || i>0)
					{
						int dp2 = dispMap.at<uchar>(p.y-1,p.x+1);
						double dpw = dispWeight*exp(-(double)pow(dp1-dp2,2.0)/(2.0*pow(dispStd,2.0)));
						w += dpw;
					}

					m_graph.addEdges( vtxIdx, vtxIdx-m_width+1, w, w );
				}
			}
		}
	}

	//imshow("FG likelihood", m_fgdPb*255);
	//imshow("BG likelihood", m_bgdPb*255);

}

//! Member function that adds superpixels as graph nodes
void CInterViewGraph::addSuperpixelNodes()
{
	double clrWeight = 0.1;
	for(int j=0; j<m_mvd->dsImg64Vec.size(); j++)
	{
		for( int i=0; i<m_mvd->lCntVec[j]; i++)
		{
			double unaryW = 10.0;
			int vtxIdx = m_graph.addVtx();
			// take the average colour of current sp
			Vec3d color = m_mvd->spClVec[j].at<Vec3d>(0,i);

			// set t-weights
			double fromSource, toSink;
			fromSource	= -clrWeight*log( m_mvd->gmmVec[j].bgdGMM(color) +1e-10 );
			toSink		= -clrWeight*log( m_mvd->gmmVec[j].fgdGMM(color) +1e-10 );

			fromSource	+= -(1.0-clrWeight)*log( m_mvd->SPProbVec[0][j][i] +1e-10 );
			toSink		+= -(1.0-clrWeight)*log( m_mvd->SPProbVec[1][j][i] +1e-10 );
			
			// unary term
			m_graph.addTermWeights( vtxIdx, unaryW*fromSource, unaryW*toSink );



		}
	}
}

//! Member function that adds edges between pixels and superpixels
void CInterViewGraph::addSppixelEdges()
{
	cv::Point p;

	for(int i = 0; i<m_mvd->dsImg64Vec.size(); i++)
	{
		Mat img8;
		m_mvd->dsImg64Vec[i].convertTo(img8, CV_8UC3);
		Mat imgLab;
		cvtColor(img8, imgLab, CV_BGR2Lab);

		// compute variance of similarity measure between sp and pix
		vector<float> histSimVec;
		for( p.y = 0; p.y < m_height; p.y++ )
		{
			for( p.x = 0; p.x < m_width; p.x++)
			{

				int lb = m_mvd->dsSpVec[i].at<float>( p.y,p.x );
				//if(lb<0 || lb>lCntVec[i]-1) continue;

				float texton = m_mvd->textonMapVec[i].at<float>(p);
				float textonProb = m_mvd->histVec[0][i][lb].at<float>(texton);

				int perHistLen = m_mvd->histVec[2][i][lb].rows/3.0;
				Vec3b labvec = imgLab.at<Vec3b>(p);
				float lprob = m_mvd->histVec[2][i][lb].at<float>(floor((double)labvec[0]/(256.0/(double)perHistLen)));
				float aprob = m_mvd->histVec[2][i][lb].at<float>(perHistLen+floor((double)labvec[1]/(256.0/(double)perHistLen)));
				float bprob = m_mvd->histVec[2][i][lb].at<float>(2*perHistLen+floor((double)labvec[2]/(256.0/(double)perHistLen)));
				float labProb = (lprob+aprob+bprob)/3.0;

				histSimVec.push_back(1.0-(textonProb+labProb)*0.5);
			}
		}

		float* histSimArr = &histSimVec[0];
		Mat histSimMat(1,histSimVec.size(),CV_32FC1,histSimArr);
		Scalar simMean = mean(histSimMat);
		
		float simSigma = 2.0*pow(simMean.val[0], 2.0);
		//cout << "simSigma = " << simSigma << endl;

		int c = 0;
		for( p.y = 0; p.y < m_height; p.y++ )
		{
			for( p.x = 0; p.x < m_width; p.x++)
			{

				int lb = m_mvd->dsSpVec[i].at<float>( p.y,p.x );
				float textonLabProb = histSimVec[c++];

				Vec3d color_pix = m_mvd->dsImg64Vec[i].at<Vec3d>( p.y,p.x );
				Vec3d color_sp = m_mvd->spClVec[i].at<Vec3d>(0,lb);

				Vec3d diff;

				diff[0] = color_pix[0] - color_sp[0];
				diff[1] = color_pix[1] - color_sp[1];
				diff[2] = color_pix[2] - color_sp[2];

				//cout << "textonLabProb = " << textonLabProb << endl;
                double w = 1.0* m_gamma * exp(-pow(1.0-textonLabProb,2.0)/simSigma); //2
				//cout << "exp(-pow(1.0-textonLabProb,2.0)/simSigma) = " << exp(-pow(1.0-textonLabProb,2.0)/simSigma) << endl;
				//double w = gammaDiv;

				// total number of sps before current sp map
				int cnt = 0;
				for(int v=0; v<i; v++)
					cnt += m_mvd->lCntVec[v];

				int vtxIdx_pix = p.x+p.y*m_width+i*m_width*m_height;
				int vtxIdx_sp = m_mvd->dsImg64Vec.size()*m_width*m_height+lb+cnt;

				//w = m_gamma;
				m_graph.addEdges( vtxIdx_pix, vtxIdx_sp, w, w );

			}
		}
	}
}

float CInterViewGraph::computeSPHistKLDiv(int view1, int view2, int sp1, int sp2)
{
	Mat textonHist1 = m_mvd->histVec[0][view1][sp1];
	Mat textonHist2 = m_mvd->histVec[0][view2][sp2];
	Mat divHist;
	cv::divide(textonHist1,textonHist2+1e-10,divHist);
	Mat logHist;
	cv::log(divHist+1e-10,logHist);
	Mat tempMat;
	cv::multiply(textonHist1,logHist,tempMat);
	float textonHistDist = sum(tempMat).val[0];

	Mat siftHist1 = m_mvd->histVec[1][view1][sp1];
	Mat siftHist2 = m_mvd->histVec[1][view2][sp2];
	//Mat divHist;
	cv::divide(siftHist1,siftHist2+1e-10,divHist);
	//Mat logHist;
	cv::log(divHist+1e-10,logHist);
	//Mat tempMat;
	cv::multiply(siftHist1,logHist,tempMat);
	float siftHistDist = sum(tempMat).val[0];

	Mat labHist1 = m_mvd->histVec[2][view1][sp1];
	Mat labHist2 = m_mvd->histVec[2][view2][sp2];
	//Mat divHist;
	cv::divide(labHist1,labHist2+1e-10,divHist);
	//Mat logHist;
	cv::log(divHist+1e-10,logHist);
	//Mat tempMat;
	cv::multiply(labHist1,logHist,tempMat);
	float labHistDist = sum(tempMat).val[0];

	float histDist = 0.4*textonHistDist+0.3*siftHistDist+0.3*labHistDist;
	//float histDist = 0.0*textonHistDist+0.0*siftHistDist+1.0*labHistDist;
	return abs(histDist);
}

float CInterViewGraph::computeSPHistDist(int view1, int view2, int sp1, int sp2)
{
	Mat textonHist1 = m_mvd->histVec[0][view1][sp1];
	Mat textonHist2 = m_mvd->histVec[0][view2][sp2];
	Mat tDist = textonHist1-textonHist2;
	Mat powTDist;
	Mat tempMat;
	cv::pow(tDist, 2.0, powTDist);
	cv::divide(powTDist,textonHist1+textonHist2,tempMat);
	float textonHistDist = 0.5*sum(tempMat).val[0];

	Mat siftHist1 = m_mvd->histVec[1][view1][sp1];
	Mat siftHist2 = m_mvd->histVec[1][view2][sp2];
	Mat sDist = siftHist1-siftHist2;
	Mat powSDist;
	cv::pow(sDist, 2.0, powSDist);
	cv::divide(powSDist,siftHist1+siftHist2,tempMat);
	float siftHistDist = 0.5*sum(tempMat).val[0];

	Mat labHist1 = m_mvd->histVec[2][view1][sp1];
	Mat labHist2 = m_mvd->histVec[2][view2][sp2];
	Mat lDist = labHist1-labHist2;
	Mat powLDist;
	cv::pow(lDist, 2.0, powLDist);
	cv::divide(powLDist,labHist1+labHist2,tempMat);
	float labHistDist = 0.5*sum(tempMat).val[0];

	float histDist = 0.4*textonHistDist+0.3*siftHistDist+0.3*labHistDist;
	return histDist;
}

//! Member function that adds edges between superpixels
void CInterViewGraph::addSuperpixelEdges()
{
	// inter view
	for(int k = 0; k<m_mvd->dsImg64Vec.size()-1; k++)
	{
		// compute the variance of histogram distance between all inter-frame sp 
		vector<float> histDistVec;
		for( int i=0; i<m_mvd->lCntVec[k]; i++)
			for( int j=0; j<m_mvd->lCntVec[k+1]; j++)
				if(m_spAdjMatVec[k].at<uchar>(i, j)>0)
					histDistVec.push_back(min(computeSPHistKLDiv(k,k+1,i,j),computeSPHistKLDiv(k+1,k,j,i)));
					//histDistVec.push_back(computeSPHistDist(k,k+1,i,j));

		float* histDistArr = &histDistVec[0];
		Mat histDistMat(1,histDistVec.size(),CV_32FC1,histDistArr);
		Scalar distMean = mean(histDistMat);
		
		double distSigma = 2.0*pow(distMean.val[0], 2.0);
		//cout << "distSigma=" << distSigma << endl;

		//cout << "OK1" << endl;
		int c = 0;
		for( int i=0; i<m_mvd->lCntVec[k]; i++)
		{

			//cout << "OK2" << endl;
			Vec3d color_sp1 = m_mvd->spClVec[k].at<Vec3d>(0,i);

			// total number of sps before current sp map
			
			int cnt = 0;
			for(int v=0; v<k; v++)
				cnt += m_mvd->lCntVec[v];

			int vtxIdx_sp1 = m_mvd->dsImg64Vec.size()*m_width*m_height + cnt + i;
			
			for( int j=0; j<m_mvd->lCntVec[k+1]; j++)
			{
				if(m_spAdjMatVec[k].at<uchar>(i, j)>0)
				{
					//cout << "OK5" << endl;
					int vtxIdx_sp2 = m_mvd->dsImg64Vec.size()*m_width*m_height + cnt + m_mvd->lCntVec[k] + j;

					Vec3d color_sp2 = m_mvd->spClVec[k+1].at<Vec3d>(0,j);

					float histDist = histDistVec[c++];

					const double gammaDiv = m_gamma * exp(-pow(histDist,(float)2.0)/distSigma);

					Vec3d diff = color_sp1 - color_sp2;

					double w = 10.0*gammaDiv;
					//double w = 2.0*m_gamma;
					m_graph.addEdges( vtxIdx_sp1, vtxIdx_sp2, w, w );

				}
			}
		}
	}
	
	
	if(m_mvd->dsImg64Vec.size()>2)
	{
		cout << "Circle edges between superpixels" << endl;
		int k = m_mvd->dsImg64Vec.size()-1;

		// compute the variance of histogram distance between all inter-frame sp 
		vector<float> histDistVec;
		for( int i=0; i<m_mvd->lCntVec[k]; i++)
			for( int j=0; j<m_mvd->lCntVec[0]; j++)
				if(m_spAdjMatVec[k].at<uchar>(i, j)>0)
					histDistVec.push_back(min(computeSPHistKLDiv(k,0,i,j), computeSPHistKLDiv(0,k,j,i)));
					//histDistVec.push_back(computeSPHistDist(k,0,i,j));

		float* histDistArr = &histDistVec[0];
		Mat histDistMat(1,histDistVec.size(),CV_32FC1,histDistArr);
		Scalar distMean = mean(histDistMat);
		
		double distSigma = 2*pow(distMean.val[0], 2.0)+1e-10;

		int c = 0;
		for( int i=0; i<m_mvd->lCntVec[k]; i++)
		{
			//cout << "OK1" << endl;
			Vec3d color_sp1 = m_mvd->spClVec[k].at<Vec3d>(0,i);

			// total number of sps before current sp map
			int cnt = m_tolSPCnt - m_mvd->lCntVec[k];

			int vtxIdx_sp1 = m_mvd->dsImg64Vec.size()*m_width*m_height + cnt + i;

			//cout << "OK2" << endl;
			for( int j=0; j<m_mvd->lCntVec[0]; j++)
			{
				if(m_spAdjMatVec[k].at<uchar>(i, j)>0)
				{
					//cout << "OK3" << endl;
					//float histDist = computeSPHistDist(k,0,i,j);
					float histDist = histDistVec[c++];
					
					int vtxIdx_sp2 = m_mvd->dsImg64Vec.size()*m_width*m_height + j;
					Vec3d color_sp2 = m_mvd->spClVec[0].at<Vec3d>(0,j);

					//const double gammaDivSqrt2 = m_gamma / std::sqrt(2.0f);
					const double gammaDiv = m_gamma * exp(-pow(histDist,(float)2.0)/distSigma);
					Vec3d diff = color_sp1 - color_sp2;
					//double w = gammaDiv * exp(-(m_betaVec[k]+m_betaVec[0])/2.0*diff.dot(diff));
					double w = 10.0*gammaDiv;
					//double w = 2.0*m_gamma;
					//if(cnt<=524)
						m_graph.addEdges( vtxIdx_sp1, vtxIdx_sp2, w, w );
					//cnt += 2;
					//cout << "cnt=" << cnt << " i=" << i << " j=" << j << endl;
				}
			}
		}
	}

	// intra frame sp edges
	for(int k = 0; k<m_mvd->dsImg64Vec.size(); k++)
	{
		// compute the variance of histogram distance between all inter-frame sp 
		vector<float> histDistVec;
		for( int i=0; i<m_mvd->lCntVec[k]; i++)
			for( int j=0; j<m_mvd->lCntVec[k]; j++)
				if(m_spIntraAdjMatVec[k].at<uchar>(i, j)>0)
					histDistVec.push_back(min(computeSPHistKLDiv(k,k,i,j),computeSPHistKLDiv(k,k,j,i)));
					//histDistVec.push_back(computeSPHistDist(k,k,i,j));

		float* histDistArr = &histDistVec[0];
		Mat histDistMat(1,histDistVec.size(),CV_32FC1,histDistArr);
		Scalar distMean = mean(histDistMat);
		
		double distSigma = 2.0*pow(distMean.val[0], 2.2);
		int c = 0;

		double dispWeight = 0.0; //10
		double dispStd = 10;

		for( int i=0; i<m_mvd->lCntVec[k]; i++)
		{

			//cout << "OK2" << endl;
			Vec3d color_sp1 = m_mvd->spClVec[k].at<Vec3d>(0,i);

			// total number of sps before current sp map
			int cnt = 0;
			for(int v=0; v<k; v++)
				cnt += m_mvd->lCntVec[v];

			int vtxIdx_sp1 = m_mvd->dsImg64Vec.size()*m_width*m_height + cnt + i;


			for( int j=i+1; j<m_mvd->lCntVec[k]; j++)
			{
				if(m_spIntraAdjMatVec[k].at<uchar>(i,j)==1)
				{
					//float histDist = computeSPHistDist(k,k,i,j);
					//cout << "histDistVec.size() = " << histDistVec.size() << endl;
					//cout << "c = " << c << endl;
					float histDist = histDistVec[c++];
					Vec3d color_sp2 = m_mvd->spClVec[k].at<Vec3d>(0,j);
					//const double gammaDivSqrt2 = m_gamma / std::sqrt(2.0f);
					//cout << "distSigma = " << distSigma << endl;
					const double gammaDiv = m_gamma * exp(-pow(histDist,(float)2.0)/distSigma);
					//cout << "distSigma2 = " << distSigma << endl;
					Vec3d diff = color_sp1 - color_sp2;
					//double w = gammaDiv * exp(-m_betaVec[k]*diff.dot(diff));
					double w = 1.0*gammaDiv;

					//int dispIndx = k>0 ? k-1 : m_mvd->dsSpVec.size()-1;
					int dispIndx = m_mvd->dsSpVec.size()>2 ? k : k-1;
					if(m_mvd->dsSpVec.size()>2 || k>0)
					{
						// n-weight from disparity
						//cout << "spDispVec.size() = " << m_mvd->spDispVec.size() << endl;
						int dp1 = m_mvd->spDispVec[dispIndx][i];
						int dp2 = m_mvd->spDispVec[dispIndx][j];
						//double dpw = 1.0*exp(0.05*(double)abs(dp1-dp2));
						double dpw = dispWeight*exp(-(double)pow(dp1-dp2,2.0)/(2.0*pow(dispStd,2.0)));
						w += dpw;
					}

					m_graph.addEdges( vtxIdx_sp1, vtxIdx_sp1+(j-i), w, w );
				}

			}

		}
	}

	
}

//! Member function that performs MRF inference
void CInterViewGraph::inference()
{
	m_graph.maxFlow();
	for(int k = 0; k<m_mvd->dsImg64Vec.size(); k++)
	{
		cv::Point p;
		Mat mask(m_height,m_width,CV_8UC1);
		for( p.y = 0; p.y < m_height; p.y++ )
		{
			for( p.x = 0; p.x < m_width; p.x++ )
			{
				if( m_graph.inSourceSegment( p.y*m_width+p.x + k*m_width*m_height /*vertex index*/ ) )
					mask.at<uchar>(p) = 1;
				else
					mask.at<uchar>(p) = 0;
            
			}
		}
		cv::Rect r(1,1,m_width-2,m_height-2);
		Mat allZeros = Mat::zeros(m_height,m_width,CV_8UC1);
		Mat dsroi = allZeros(r);
		mask(r).convertTo(dsroi, dsroi.type());
		mask = allZeros.clone();

		m_maskVec.push_back(mask);
		/*
		string idstr;          // string which will contain the result
		stringstream convert;   // stream used for the conversion
		convert << k;      // insert the textual representation of 'k' in the characters in the stream
		idstr = convert.str(); // set 'idstr' to the contents of the stream

		string winname = "downsampled mask "+ idstr;
		imshow(winname, mask*255);
		waitKey();
		//imwrite(winname+".png", mask*255);*/
	}


	for(int k = 0; k<m_mvd->dsImg64Vec.size(); k++)
	{
		Mat spMask(1,m_mvd->lCntVec[k],CV_8UC1);
		// total number of sps before current sp map
		int cnt = 0;
		for(int v=0; v<k; v++)
			cnt += m_mvd->lCntVec[v];

		for( int i=0; i<m_mvd->lCntVec[k]; i++)
		{
				if( m_graph.inSourceSegment( m_mvd->dsImg64Vec.size()*m_width*m_height + cnt + i /*vertex index*/ ) )
					spMask.at<uchar>(0,i) = 1;
				else
					spMask.at<uchar>(0,i) = 0;
		}
		m_spMaskVec.push_back(spMask);
	}

}

//! Member function that returns the binary labeling map
/*!
    \return the binary labeling map.
*/
void CInterViewGraph::getMask(vector<Mat>& maskVec, vector<Mat>& spMaskVec)
{
	 maskVec = m_maskVec;
	 spMaskVec = m_spMaskVec;
}


/*!
  per point derivatives
*/
void CInterViewGraph::gauss1d(double sigma, double mean, double x, int ord, double & g)
{
	x = x-mean;
	double num = x*x;
	double variance = sigma*sigma;
	double denom = 2*variance;
	g = (double)exp(-num/denom)/(double)sqrt(CV_PI*denom);
	
	switch(ord)
	{
	      case 1: 
		      g=-(double)g*((double)x/(double)variance);
		      break;
	      case 2: 
		      g=(double)g*((double)(num-variance)/(double)(variance*variance));
			  //cout << "g = " << g << endl;
		      break; 
	}
  
}


void CInterViewGraph::makefilter(double scale, int phasex, int phasey, Mat& pts, int SUP, Mat& f)
{
  	f = Mat(SUP,SUP,CV_64FC1).clone();
	double sum = 0;
	cv::Point p;
	for(p.x=0; p.x<SUP*SUP; p.x++)
	{
		int dx = floor((double)p.x/(double)SUP);
		int dy = p.x-dx*SUP;
		double gx = 0, gy = 0;
		for(p.y=0; p.y<2; p.y++)
		{
			if(p.y==0)
				gauss1d(scale,0,pts.at<double>(p),phasex,gx);
			else
				gauss1d(scale,0,pts.at<double>(p),phasey,gy);
		}
		      
		f.at<double>(dy,dx) = gx*gy;
		sum += gx*gy;
	}

	
	//!< normalise
	double mean = (double)sum/(double)(SUP*SUP);
	//cout << "mean = " << mean << endl;
	f = f-mean;
	double abssum = 0;
	///*
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for(p.x=0; p.x<SUP; p.x++)
	      {
			  abssum += (double)abs(f.at<double>(p));
			  //cout << "f = " << f.at<double>(p) << endl;
		  }
		  //cout << endl;
	}
	
	if(abssum>0)
		f = f/(double)abssum;
	
	
	sum = 0;
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for(p.x=0; p.x<SUP; p.x++)
	      {
			  //cout << "f = " << f.at<double>(p) << endl;
			  sum += (double)f.at<double>(p);
		  }
		  //cout << endl;
	}//*/
	
}

void CInterViewGraph::gaussianFilter(Mat& H, int size, double std)
{
  
	H = Mat(size,size,CV_64FC1).clone();
  	int hsup = (size-1)/2;
	//int disp = hsup+1;
	
	cv::Point p;
	double sum = 0;
	//Mat meshgridMat(size,size,CV_64FC2);
	for(p.y=0; p.y<size; p.y++)
	{
	      for(p.x=0; p.x<size; p.x++)
	      {
		      double x = p.x-hsup;
		      double y = p.y-hsup;
		      double h = exp(-(x*x + y*y)/(2*std*std));
		      H.at<double>(p) = h;
		      sum += h;     
	      }
	}
	
	if(sum!=0)
	      H = H/sum;

	//!< normalise
	Scalar sum_scalar = cv::sum(H);
	sum = sum_scalar.val[0];
	double mean = sum/(double)(size*size);
	H = H-mean;
	
	Scalar abssum_scalar = cv::sum(cv::abs(H));
	double abssum = abssum_scalar.val[0];
	
	if(abssum>0)	
		H = H/abssum;
}

void CInterViewGraph::makeRFSfilters(Mat& F, double rot, double scale)
{
	int SUP = 11; 		//!< Support of the largest filter (must be odd)
	int hsup = (SUP-1)/2;
	//int disp = hsup+1;
	
	cv::Point p;
	Mat meshgridMat(SUP,SUP,CV_64FC2);
	for(p.y=0; p.y<SUP; p.y++)
	{
	      for( p.x=0; p.x<SUP; p.x++)
	      {
		      meshgridMat.at<Vec2d>(p)[0] = p.x-hsup;
			  meshgridMat.at<Vec2d>(p)[1] = p.y-hsup;
		      //meshgridMat.at<Vec2d>(p)[1] = hsup-p.y;
	      }
	}
	
	Mat orgPts(2,SUP*SUP,CV_64FC1);
	for(p.y=0; p.y<2; p.y++)
	{
	      for(p.x=0; p.x<SUP*SUP; p.x++)
	      {
		      int dx = floor((float)p.x/SUP);
		      int dy = p.x-dx*SUP;
		      orgPts.at<double>(p) = meshgridMat.at<Vec2d>(dy,dx)[p.y];
	      }
	}  
	
	double c = cos(rot);
	double s = sin(rot);
	Mat rotMat = (Mat_<double>(2,2) << c, -s, s, c);
	Mat rotPts = rotMat*orgPts;

	gaussianFilter(F, SUP, 5);
	//makefilter(scale,0,0,rotPts,SUP,F);

}

void CInterViewGraph::skeletonization(Mat mask, Mat& skel)
{
	
	mask = 255*(mask);
	skel = Mat::zeros(mask.rows, mask.cols, CV_8UC1);
	Mat temp(mask.rows, mask.cols, CV_8UC1);
	Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));

	//imshow("mask", mask);

	bool done;

	do
	{
		cv::morphologyEx(mask, temp, cv::MORPH_OPEN, element);
		cv::bitwise_not(temp, temp);
		cv::bitwise_and(mask, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		cv::erode(mask, mask, element);

		double max;
		cv::minMaxLoc(mask, 0, &max);
		done = (max==0);
	} 
	while(!done);
	imshow("skeleton", skel);
	
	//waitKey(0);
}