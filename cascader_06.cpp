#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

String facefile = "D:/opencv3.1/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
String lefteyefile = "D:/opencv3.1/opencv/build/etc/haarcascades/haarcascade_eye.xml";
String righteyefile = "D:/opencv3.1/opencv/build/etc/haarcascades/haarcascade_eye.xml";
CascadeClassifier face_detector;
CascadeClassifier leftyeye_detector;
CascadeClassifier righteye_detector;

Rect leftEye , rightEye;
void trackEye(Mat&  im, Mat& tpl, Rect& rect) {
	Mat result;
	int result_cols = im.cols - tpl.cols + 1;
	int result_rows = im.rows - tpl.rows + 1;

	// Ä£°åÆ¥Åä
	result.create(result_rows, result_cols, CV_32FC1);
	matchTemplate(im, tpl, result, TM_CCORR_NORMED);

	// Ñ°ÕÒÎ»ÖÃ
	double minval, maxval;
	Point minloc, maxloc;
	minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
	if (maxval > 0.75) {
		rect.x = rect.x + maxloc.x;
		rect.y = rect.y + maxloc.y;
	} else {
		rect.x = rect.y = rect.width = rect.height = 0;
	}
}

int main(int argc, char** argv) {
	if (!face_detector.load(facefile)) {
		printf("could not load data file...\n");
		return -1;
	}
	if (!leftyeye_detector.load(lefteyefile)) {
		printf("could not load data file...\n");
		return -1;
	}
	if (!righteye_detector.load(righteyefile)) {
		printf("could not load data file...\n");
		return -1;
	}

	Mat frame;
	VideoCapture capture(0);
	namedWindow("demo-win", CV_WINDOW_AUTOSIZE);

	Mat gray;
	vector<Rect> faces;
	vector<Rect> eyes;
	Mat lefttpl, righttpl; // Ä£°å
	while (capture.read(frame)) {
		flip(frame, frame, 1);
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		face_detector.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
		for (size_t t = 0; t < faces.size(); t++) {
			rectangle(frame, faces[t], Scalar(255, 0, 0), 2, 8, 0);

			// ¼ÆËã offset ROI
			int offsety = faces[t].height / 4;
			int offsetx = faces[t].width / 8;
			int eyeheight = faces[t].height/2 - offsety;
			int eyewidth = faces[t].width/2 - offsetx;

			// ½ØÈ¡×óÑÛÇøÓò
			Rect leftRect;
			leftRect.x = faces[t].x + offsetx;
			leftRect.y = faces[t].y + offsety;
			leftRect.width = eyewidth;
			leftRect.height = eyeheight;
			Mat leftRoi = gray(leftRect);

			// ¼ì²â×óÑÛ
			leftyeye_detector.detectMultiScale(leftRoi, eyes, 1.1, 1, 0, Size(20, 20));
			if (lefttpl.empty()) {
				if (eyes.size()) {
					leftRect = eyes[0] + Point(leftRect.x, leftRect.y);
					lefttpl = gray(leftRect);
					rectangle(frame, leftRect, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			else {
				// ¸ú×Ù£¬ »ùÓÚÄ£°åÆ¥Åä
				leftEye.x = leftRect.x;
				leftEye.y = leftRect.y;
				trackEye(leftRoi, lefttpl, leftEye);
				if (leftEye.x > 0 && leftEye.y > 0) {
					leftEye.width = lefttpl.cols;
					leftEye.height = lefttpl.rows;
					rectangle(frame, leftEye, Scalar(0, 0, 255), 2, 8, 0);
				}
			}

			// ½ØÈ¡ÓÒÑÛÇøÓò
			Rect rightRect;
			rightRect.x = faces[t].x + faces[t].width/2;
			rightRect.y = faces[t].y + offsety;
			rightRect.width = eyewidth;
			rightRect.height = eyeheight;
			Mat rightRoi = gray(rightRect);

			// ¼ì²âÓÒÑÛ
			righteye_detector.detectMultiScale(rightRoi, eyes, 1.1, 1, 0, Size(20, 20));
			if (righttpl.empty()) {
				if (eyes.size()) {
					rightRect = eyes[0] + Point(rightRect.x, rightRect.y);
					righttpl = gray(rightRect);
					rectangle(frame, rightRect, Scalar(0, 255, 255), 2, 8, 0);
				}
			} else {
				// ¸ú×Ù£¬ »ùÓÚÄ£°åÆ¥Åä
				rightEye.x = rightRect.x;
				rightEye.y = rightRect.y;
				trackEye(rightRoi, righttpl, rightEye);
				if (rightEye.x > 0 && rightEye.y > 0) {
					rightEye.width = righttpl.cols;
					rightEye.height = righttpl.rows;
					rectangle(frame, rightEye, Scalar(0, 255, 255), 2, 8, 0);
				}
			}
		}
		imshow("demo-win", frame);
		char c = waitKey(100);
		if (c == 27) { // ESC
			break;
		}
	}

	// release resource
	capture.release();
	waitKey(0);
	return 0;
}