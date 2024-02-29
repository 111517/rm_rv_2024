#include"yolo.h"
#include"kalman_filter.h"
#include"kalman_filter.hpp"
#include<string>
#include<sstream>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <math.h>
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Eigen;
use_kalman_filter KF;

Mat camera_matrix = (Mat_<double>(3, 3) <<4.52592997e+03 , 0.00000000e+00 , 3.02083849e+02,
 0.00000000e+00 , 4.21147920e+03 ,-6.99078517e+01,
 0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00);
	// 相机畸变系数
Mat dist_coeffs = (Mat_<double>(5, 1) << -6.72576871e+00 ,-3.32306997e+00,  4.84083522e-01 ,-1.50676089e-03,
  -4.71233533e+01);


double left_x,left_y,right_x,right_y;
double x_d,y_d;
double x_new,y_new;



void calAngle(Mat cam,Mat dis,int x,int y)
{
    double fx=cam.at<double>(0,0);
    double fy=cam.at<double>(1,1);
    double cx=cam.at<double>(0,2);
    double cy=cam.at<double>(1,2);
    Point2f pnt;
    vector<cv::Point2f> in;
    vector<cv::Point2f> out;
    in.push_back(Point2f(x,y));
    //对像素点去畸变
    undistortPoints(in,out,cam,dis,noArray(),cam);
    pnt=out.front();
    //没有去畸变时的比值
    double rx=(x-cx)/fx;
    double ry=(y-cy)/fy;
    //去畸变后的比值
    double rxNew=(pnt.x-cx)/fx;
    double ryNew=(pnt.y-cy)/fy;
    //输出原来点的坐标和去畸变后点的坐标
    //cout<< "x: "<<x<<" xNew:"<<pnt.x<<endl;
    //cout<< "y: "<<y<<" yNew:"<<pnt.y<<endl;
	x_new=pnt.x;
	y_new=pnt.y;
    //输出未去畸变时测得的角度和去畸变后测得的角度
    //cout<< "angx: "<<atan(rx)/CV_PI*180<<" angleNew:"<<atan(rxNew)/CV_PI*180<<endl;
    //cout<< "angy: "<<atan(ry)/CV_PI*180<<" angleNew:"<<atan(ryNew)/CV_PI*180<<endl;
}

void distance(Mat cam,Mat dis,vector<Point2d>pnts)
{
#define HALF_LENGTH 67.5
#define HALF_WIDTH 28.1
//自定义的物体世界坐标，单位为mm
vector<Point3f> obj=vector<Point3f>{
    cv::Point3f(-HALF_LENGTH, -HALF_WIDTH, 0),	//tl
    cv::Point3f(HALF_LENGTH, -HALF_WIDTH ,0),	//tr
    cv::Point3f(HALF_LENGTH, HALF_WIDTH, 0),	//br
    cv::Point3f(-HALF_LENGTH, HALF_WIDTH, 0)	//bl
};
cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);//init rvec
cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);//init tvec
//进行位置解算
solvePnP(obj,pnts,cam,dis,rVec,tVec,false,SOLVEPNP_ITERATIVE);
//输出平移向量
//pnts.clear()
//cout <<"tvec: "<<tVec<<endl;
//cout<<"rvec: "<<rVec<<endl;
cv::Mat rotateX;
cv::Rodrigues(rVec,rotateX);
//cout<<"rotateX: "<<rotateX<<endl;
double value1=rotateX.at<double>(0,0);
double value2=rotateX.at<double>(1,0);
double value3=rotateX.at<double>(2,0);
double value4=rotateX.at<double>(0,1);
double value5=rotateX.at<double>(1,1);
double value6=rotateX.at<double>(2,1);
double value7=rotateX.at<double>(0,2);
double value8=rotateX.at<double>(1,2);
double value9=rotateX.at<double>(2,2);
//cout<<"value1: "<<value1<<" value2: "<<value2<<" value3: "<<value3<<endl;
// float theta_z = np.arctan2(value2, value1) / np.pi * 180;
// float theta_y = np.arctan2(-1 * value3, np.sqrt(value6 * value6 + value9 * value9)) / np.pi * 180;
// float theta_x = np.arctan2(vlaue6, value9) / np.pi * 180;
// std::cout << "Theta_z: " << theta_z << std::endl;
// std::cout << "Theta_y: " << theta_y << std::endl;
// std::cout << "Theta_x: " << theta_x << std::endl;
//x_d=tVec.at<double>(0,0);
//y_d=tVec.at<double>(1,0);
float x, y, z;
float sy = sqrt(rotateX.at<double>(0,0) * rotateX.at<double>(0,0) +  rotateX.at<double>(1,0) * rotateX.at<double>(1,0) );
x = atan2(rotateX.at<double>(2,1) , rotateX.at<double>(2,2));
y = atan2(-rotateX.at<double>(2,0), sy);
z = atan2(rotateX.at<double>(1,0), rotateX.at<double>(0,0));
cout<<"x: "<<x<<" y: "<<y<<" z: "<<z<<endl;
//cout<<"x_d_dis: "<<x_d<<" y_d_dis: "<<y_d<<endl;
x_d=y;
y_d=z;
}


void kalman(double x,double y)
{
	Mat trans;
	// /=================卡尔曼滤波器的初始化===============================
	// 初始化的值有状态量的数值		stateNum
	// 真实测量值的个数				measureNum
	// 系统噪声方差矩阵				Q
	// 测量噪声方差矩阵				R
	// 预测值的协方差初始化			P
	// X的初始化				x_state
	// 转移矩阵（运动方程）			F

	//kalman_filter_initial
	KF.stateNum = 4;
	KF.measureNum = 2;
	KF.init();
	KF.set_Q(0.0001);
	KF.set_R(0.01);
	KF.set_P(1);
	KF.x_state = (Mat_<float>(4, 1) << x,y, 4, 4);

	//预测阶段
	KF.get_F((Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1));	//kf.transitionMatrix
	KF.z = (Mat_<float>(2, 1) << x, y);     						//测量值
	KF.correct();
	cout<<"kf_x:"<<KF.x_state.at<float>(0)<<" kf_y:"<< KF.x_state.at<float>(1)<<endl;		//预测值

};



bool Yolo::readModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNet(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	//cuda
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}
bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < strideSize; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) {	//anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
					if (box_score >= boxThreshold) {
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre;
						if (max_class_socre >= classThreshold) {
							//rect [x,y,w,h]
							float x = pdata[0];  //x
							float y = pdata[1];  //y
							float w = pdata[2];  //w
							float h = pdata[3];  //h
							int left =(x - 0.5 * w) * ratio_w;
							int top = (y - 0.5 * h) * ratio_h;

							left_x = (x - 0.5 * w) * ratio_w;
							left_y = (y - 0.5 * h) * ratio_h;
							right_x = (x + 0.5 * w) * ratio_w;
							right_y = (y + 0.5 * h) * ratio_h;

							vector<Point2d>image_points;
	                        image_points.push_back(Point2d(left_x,left_y));
	                        image_points.push_back(Point2d(left_x,right_y));
	                        image_points.push_back(Point2d(right_x,right_y));
	                        image_points.push_back(Point2d(right_x,left_y));

							//cout<<"left_x:"<<left_x<<" left_y:"<<left_y<<endl;
							//cout<<"right_x:"<<right_x<<" right_y:"<<right_y<<endl;

							distance(camera_matrix,dist_coeffs,image_points);
							//cout<<"x_d_:"<<x_d<<" y_d_:"<<y_d<<endl;

							//calAngle(camera_matrix,dist_coeffs,x_d,y_d);
							//cout<<"x_d_cal:"<<x_new<<" y_d_cal:"<<y_new<<endl;

							kalman(x_d,y_d);

							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//下一行
				}
			}
		}
	}
 
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
	if (output.size())
		return true;
	else
		return false;
}
 
void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		int width, height;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		//rectangle(img, result[i].box, color[result[i].id], 2, 8);
		string label = className[result[i].id] + ":" + to_string(result[i].confidence);
		cout<<"label:"<<label<<endl;
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(img, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		
		//putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 0.5);
		circle(img, Point(right_x,left_y),3 ,(0,0,255),5);
		circle(img, Point(left_x,right_y),3,(255,255,0),5);
		circle(img, Point(left_x,left_y),3,(255,0,0),5);
		circle(img, Point(right_x,right_y),3,(0,255,0),5);
	}
	//imshow("1", img);
	//imwrite("out.bmp", img);
	//waitKey();
	//destroyAllWindows();
}