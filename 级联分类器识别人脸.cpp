#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

String facefile = "D:/opencv3.1/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_mini.xml";
CascadeClassifier face_detector;
int main(int argc, char** argv) {
	Mat src = imread("D:/vcprojects/images/cvtest.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	if (!face_detector.load(facefile)) {
		printf("could no load image...\n");
		return -1;
	}
	imshow("input image", src);

	Mat gray;
	vector<Rect> faces;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);
	face_detector.detectMultiScale(gray, faces, 1.1, 5, 0, Size(50, 50));
	for (size_t t = 0; t < faces.size(); t++) {
		rectangle(src, faces[t], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result", src);

	waitKey(0);
	return 0;
}
