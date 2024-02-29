//#include "stdafx.h"
#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
 
using namespace std;
using namespace cv;
using namespace dnn;
 
int main()
{
	//string img_path = "C:\\Users\\m\\source\\repos\\yolov5\\wifi_01.jpg";
	//string img_path = "C:\\Users\\m\\source\\repos\\yolov5\\bus.jpg";
	//string model_path = "C:\\Users\\m\\source\\repos\\yolov5\\best.onnx";
	string model_path = "/home/ayi/c++/model/best.onnx";
	//int num_devices = cv::cuda::getCudaEnabledDeviceCount();
	//if (num_devices <= 0) {
		//cerr << "There is no cuda." << endl;
		//return -1;
	//}
	//else {
		//cout << num_devices << endl;
	//}
 
	Yolo test;
	Net net;
	if (test.readModel(net, model_path, false)) {
		cout << "read net ok!" << endl;
	}
	else {
		return -1;
	}
 
	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<Output> result;
    VideoCapture cap;
	cap.open(0,cv::CAP_V4L2);
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // where 0.25 means "manual exposure, manual iris"
    cap.set(cv::CAP_PROP_EXPOSURE, -9.2); //视频检测将0改成视频地址
	
	while (true) 
    {
		Mat fram;
		//fram = imread("/home/ayi/rm_rv/include/3.jpg");
		cap >> fram;
		vector<Output> result;
		if (test.Detect(fram, net, result))
			test.drawPred(fram, result, color);
		imshow("detect output", fram);
		if (waitKey(2)==27) break; //esc退出
	}
	cap.release();
	destroyAllWindows();
	system("pause");
	return 0;
}