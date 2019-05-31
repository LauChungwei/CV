#include<iostream>
#include<random>
#include <opencv2/opencv.hpp>
#include<time.h>
#include<filesystem>


using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

String facefile = "D:/opencv340/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
String lefteyefile = "D:/opencv340/opencv/build/etc/haarcascades/haarcascade_eye.xml";

CascadeClassifier face_detector;
CascadeClassifier leftyeye_detector;

string haar_face_datapath = "D:/opencv340/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";

string path = "D:/桌面/lcw/data_diy/data/";

int main() {
	int fileno;
	cout << "请选择一个文件夹（数字0~9）：";
	cin >> fileno ;
	
	//path1 += to_string(node) + "/";
	path += to_string(fileno) ;
	
	//在开始写入文件之前首先检查文件夹中已经有的文件的数量，在此基础上命名文件夹避免文件覆盖
	int count(0);
	for (auto&fe : fs::directory_iterator(path))	
	{
		count++;
	}
	cout << count << endl;

	//定义随机数生成引擎
	default_random_engine e(time(0));
	default_random_engine ex(time(0));
	default_random_engine ey(time(0));

	uniform_int_distribution<unsigned> u(0, 8);
	uniform_int_distribution<unsigned> ux(0, 427);
	uniform_int_distribution<unsigned> uy(0, 260);

	//定义背景
	Mat screen = Mat(Size(1281, 780), CV_8UC3, Scalar(255, 255, 255));
	namedWindow("SCREEN", WINDOW_AUTOSIZE);
	
	if (!face_detector.load(facefile)) {
		printf("could not load data file...\n");
		return -1;
	}
	if (!leftyeye_detector.load(lefteyefile)) {
		printf("could not load data file...\n");
		return -1;
	}

	Mat frame;
	VideoCapture capture(0);
	namedWindow("camera", CV_WINDOW_AUTOSIZE);

	Mat gray;
	vector<Rect> faces;
	vector<Rect> eyes;




	while (capture.read(frame)) {
		
		flip(frame, frame, 1);		//处理镜像带来的水平反向
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);		//直方图对均衡化强化图像
		face_detector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(100, 100));
		for (size_t t = 0; t < faces.size(); t++) {
			rectangle(frame, faces[t], Scalar(255, 0, 0), 2, 8, 0);

			// 计算 offset ROI，缩小模板匹配的区域，减少计算量（对可能面积缩小的同时，同时考虑了偏移量）
			int offsety = faces[t].height / 4;
			int offsetx = faces[t].width / 8;
			int eyeheight = faces[t].height / 2 - offsety;
			int eyewidth = faces[t].width / 2 - offsetx;

			// 截取左眼区域
			Rect leftRect;
			leftRect.x = faces[t].x + offsetx;
			leftRect.y = faces[t].y + offsety;
			leftRect.width = eyewidth;
			leftRect.height = eyeheight;
			Mat leftRoi = gray(leftRect);

			// 检测左眼
			leftyeye_detector.detectMultiScale(leftRoi, eyes, 1.1, 1, 0, Size(50, 50));
			for (size_t t = 0; t < eyes.size(); t++) {
				eyes[t].x = eyes[t].x + leftRect.x;
				eyes[t].y = eyes[t].y + leftRect.y;
				
			//rectangle(frame, eyes[t], Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("camera", frame);
		string path1(path);
		//int node = fileno;
		int x, y;
		switch (fileno) {
		case 0:
			x = 0, y = 0; break;
		case 1:
			x = 427, y = 0; break;
		case 2:
			x = 854, y = 0; break;
		case 3:
			x = 0, y = 260; break;
		case 4:
			x = 427, y = 260; break;
		case 5:
			x = 854, y = 260; break;
		case 6:
			x = 0, y = 520; break;
		case 7:
			x = 427, y = 520; break;
		case 8:
			x = 854, y = 520; break;
		}

		x += ux(ex);
		y += uy(ey);

		Mat bg = screen.clone();
		circle(bg, Point(x, y), 10, Scalar(0, 0, 0), 3, 8, 0);
		imshow("SCREEN", bg);


		char c = waitKey(2000);
		if (c == 27)
			break;
		else if (c == 32 && eyes.size()>0) {
			Mat dst;
			eyes[0] += Size(eyes[0].width*0.25, 0);
			resize(frame(eyes[0]), dst, Size(125, 100));
			path1 += "/" + format("eye_%07d.jpg",count);
			imwrite(path1, dst);
			count++;
		}
	}
	



	//system("pause");
	return 0;
}