#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "puttextzh.h"
#define PI 3.1415926

using namespace std;
using namespace cv;

//全局变量
Mat frame, result;//frame保存每帧原图像，result用作手势分割
vector<Mat> tpls;//手势模板
int hand = 0;//当前识别的模板下标
vector<Point>path;//运动轨迹
int awayCount=0;//轨迹偏离次数
Rect preRect = Rect(Point(-1, -1), Point(-1, -1));//上一帧的矩形框

//函数
Point Match();//模板匹配
void imageblur(Mat& src, Mat& dst, Size size, int threshold);//图片边缘光滑处理
Rect outLine(int thresh);//找出手部外接矩形
Point centerOfRect(Rect tRect);//找出矩形的中心点
float getDistance(Point p1, Point p2);//计算两点距离
void drawLine(Rect tRect);//画出运动轨迹

//图片边缘光滑处理
//size表示取均值的窗口大小，threshold表示对均值图像进行二值化的阈值
void imageblur(Mat& src, Mat& dst, Size size, int threshold)
{
	int height = src.rows;
	int width = src.cols;
	blur(src, dst, size);
	for (int i = 0; i < height; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (p[j] < threshold)
				p[j] = 0;
			else p[j] = 255;
		}
	}
}

//thresh用来根据面积筛选连通域
Rect outLine(int thresh) {
	//GaussianBlur(result, result, cv::Size(5, 5), 3, 3);//高斯滤波
	//提取轮廓
	Mat threshold_output = Mat::zeros(result.rows, result.cols, CV_8UC1);
	vector<vector<Point>> contours;
	vector<Rect> rects;//合适大小的外接矩形
	vector<Vec4i> hierachy;
	//边缘检测
	findContours(result, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> contours_poly(contours.size());

	
	Mat tImage = Mat::zeros(threshold_output.size(), CV_8UC3);
	frame.copyTo(tImage);

	for (unsigned int i = 0; i < contours.size(); i++)
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);

	int maxArea = 0;//选出最大连通域
	Rect matchRect = Rect(Point(-1,-1), Point(-1, -1));
	for (unsigned int i = 0; i < contours.size(); i++) {

		Rect rect = boundingRect(contours[i]);// 去掉小的连通域
		int tArea = rect.area();
		if (tArea < thresh)
			continue;
		rects.push_back(rect);//保存合适大小的连通域矩形框
		//rectangle(frame, maxRect, Scalar(0, 255, 0), 2, LINE_8);
		
		drawContours(tImage, contours_poly, i, Scalar(255, 0, 0), 1, 8, hierachy, 0, Point());
	}
	//imshow("轮廓", tImage);
	Point p = Match();
	p.x += 50;
	p.y += 100;
	for (unsigned int i = 0; i < rects.size(); i++) {//如果模板匹配得到的结果与该矩形框重合
		if (p.x > rects[i].tl().x&&p.x<rects[i].br().x&&p.y>rects[i].tl().y&&p.y < rects[i].br().y)
			matchRect = rects[i];
	}
	//平滑矩形大小，取当前矩形与前一帧矩形的平均尺寸,降低抖动
	if (preRect.x != -1) {
		int k = 0.7;
		matchRect.width = round(matchRect.width*(1-k) + preRect.width*k);
		matchRect.height = round(matchRect.height*(1 - k) + preRect.height*k);
		preRect = matchRect;
	}
	rectangle(frame, matchRect, Scalar(0, 255, 0), 2, LINE_8);//画出外接矩形框
	return matchRect;
}

Point Match() {
	//cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]　　各种匹配算法
	Point temploc;
	double match_max = -1;//匹配程度
	//对每个手势模板进行匹配
	for (int i = 0; i < tpls.size(); i++) {
		Mat match;
		int match_cols = result.cols - tpls[i].cols + 1;
		int match_rows = result.rows - tpls[i].rows + 1;
		match.create(match_cols, match_rows, CV_32FC1);
		matchTemplate(result, tpls[i], match, TM_CCOEFF_NORMED, Mat());
		Point minloc;
		Point maxloc;
		double mymin, mymax;
		minMaxLoc(match, &mymin, &mymax, &minloc, &maxloc, Mat());
		//取最佳匹配
		if (match_max < mymax) {
			match_max = mymax;
			temploc = maxloc;
			hand = i;//保存模板下标
		}
	}
	if (hand != tpls.size()-1)
		//cout << match_max <<"  "<<hand<< endl;
	rectangle(frame, Rect(temploc.x, temploc.y, 10, 10), Scalar(0, 0, 255), 2, LINE_AA);
	return temploc;
}

Point centerOfRect(Rect rect) {
	return Point(rect.x + round(rect.width / 2.0), rect.y + round(rect.height / 2.0));
}

float getDistance(Point p1, Point p2) {
	return sqrtf(powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2));
}


void drawLine(Rect tRect) {
	if (tRect.x != -1) {
		//Point p = centerOfRect(tRect);
		Point p = tRect.tl();
		//两点距离太大视为误差
		if (!path.empty() && getDistance(p,path.back())> 30.0) {
			awayCount++;
			if (awayCount >= 4) {//偏差次数过多代表手部大幅度位移，重新画线
				awayCount = 0;
				path.clear();
			}
		}
		else {
			path.push_back(p);
		}
	}
	polylines(frame, path, false, Scalar(0, 0, 255), 3);//根据点序列画线
}

int* getFeature(int len) {//len表示特征序列长度
	int *features=new int[len];
	int k = floor(path.size() / len);
	if (k == 0)return NULL;//记录时间太短视为无效
	float xSum=0,ySum=0;
	for (int i = 1; i < len; i++) {
		xSum += path[i*k].x;
		ySum += path[i*k].y;
	}
	float xc = xSum / len, yc = ySum / len;
	cout << "features:";
	for (int i = 0; i < len; i++) {
		float angle = fastAtan2(path[i*k].y - yc, path[i*k].x - xc);//fastAtan2返回角度，0-360
		//16方向离散化
		for (int j = 0; j < 16; j++) {
			if (angle >= j*22.5&&angle <= (j + 1)*22.5)
				features[i]=j;
		}
		cout << features[i]<<",";
	}
	cout << endl;
	path.clear();
	return features;
}

int main()
{
	VideoCapture cap(0);//打开摄像头
	//网络摄像头"http://admin:admin@192.168.137.2:8081"
	if (!cap.isOpened())
	{
		return -1;
	}
	
	Mat ycrcb_image;//YCbCr颜色空间
	Mat Y, Cr, Cb;//对应三个通道
	vector<Mat> channels;
	bool stop = false;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));//用于腐蚀和膨胀的参数
	tpls.push_back(imread("./hands/h0.jpg", CV_8UC1));//模板
	tpls.push_back(imread("./hands/h1.jpg", CV_8UC1));//模板
	tpls.push_back(imread("./hands/h5.jpg", CV_8UC1));//模板
	int tSize = tpls.size();
	for (int i = 0; i < tSize; i++) {
		tpls.push_back(Mat());
		resize(tpls[i], tpls[i + tSize], Size(), 2, 2);//每个模板2个尺寸
	}

	while (!stop)
	{
		cap >> frame;                       //读取视频帧
		flip(frame, frame, 1);				//镜像翻转
		//imshow("原始图像",frame);
		GaussianBlur(frame, frame, cv::Size(5, 5), 5, 5);//高斯滤波
		//imshow("高斯滤波",frame);
		//转换颜色空间并分割颜色通道
		cvtColor(frame, ycrcb_image, CV_BGR2YCrCb);
		split(ycrcb_image, channels); //CV_EXPORTS void split(const Mat& m, vector<Mat>& mv );
		Y = channels.at(0);
		Cr = channels.at(1);
		Cb = channels.at(2);

		//一般的图像文件格式使用的是 Unsigned 8bits，CvMat矩阵对应的参数类型就是CV_8UC1，CV_8UC2，CV_8UC3（最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3）
		result.create(frame.rows, frame.cols, CV_8UC1);

		//Otsu阈值二值化分割图像
		threshold(Cr, result, 0, 255, THRESH_BINARY + THRESH_OTSU);
		//imshow("Otsu", result);

		//开操作，先腐蚀后膨胀
		erode(result, result, element, Point(-1,-1), 2);
		dilate(result, result, element, Point(-1, -1), 2);

		//闭操作，先膨胀后腐蚀
		dilate(result, result, element, Point(-1, -1), 2);
		erode(result, result, element, Point(-1, -1), 2);
		//imshow("形态学操作", result);

		//平滑边缘
		imageblur(result, result, Size(5, 5), 240);

		//轮廓
		Rect tRect=outLine(10000);

		string tip;
		if (hand == tpls.size() - 1) {//伸五指代表停止记录
			tip = "停止记录";
			if (hand == tpls.size() - 1) {
				if (!path.empty()) {
					int* features=getFeature(24);//得到特征向量
				}
			}
		}
		else{
			tip = "开始记录";
			//画轨迹
			drawLine(tRect);
		}
		putTextZH(frame, tip.data() , Point(500, 100), Scalar(0, 255, 0), 30, "微软雅黑");
		int fps = cap.get(CAP_PROP_FPS);
		putText(frame, "fps"+to_string(fps), Point(10, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 1);
		imshow("frame", frame);
		imshow("result", result);

		if (waitKey(30) >= 0)
			stop = true;
	}

	cv::waitKey();
	// 释放申请的相关内存
	cv::destroyAllWindows();
	cap.release();
	return 0;
}