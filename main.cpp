#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h>

struct ThresholdData {
	cv::Mat image;
	int thresholdValue = 70;
	int maxValue = 255;
};

void createWindow(const char* name, int x, int y) {
	cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(name, x, y);
}

void showImage(cv::Mat img, const char* name, int x, int y) {
	createWindow(name, x, y);
	cv::imshow(name, img);
}

void changeBrightness(cv::Mat src, cv::Mat dst, int bright) {
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			cv::Vec3b pixelColor = src.at<cv::Vec3b>(i, j);

			if (pixelColor[0] + bright > 255) {
				pixelColor[0] = 255;
			}
			else {
				pixelColor[0] += bright;
			}
			if (pixelColor[1] + bright > 255) {
				pixelColor[1] = 255;
			}
			else {
				pixelColor[1] += bright;
			}
			if (pixelColor[2] + bright > 255) {
				pixelColor[2] = 255;
			}
			else {
				pixelColor[2] += bright;
			}

			dst.at<cv::Vec3b>(i, j) = pixelColor;
		}
	}
}

void onThreshold(int pos, void* userdata) {
	ThresholdData* data = static_cast<ThresholdData*>(userdata);
	cv::Mat binarizedImage;
	cv::threshold(data->image, binarizedImage, pos, data->maxValue, cv::THRESH_BINARY);
	cv::imshow("Binarization", binarizedImage);
}

void createHistogram(cv::Mat src, const char* name, int x, int y) {
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	cv::Mat histogram;
	cv::calcHist(&src, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);

	createWindow(name, x, y);

	int histW = 512, histH = 480;
	int binW = cvRound(static_cast<double>(histW) / histSize);

	cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(30, 30, 30));

	cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX);

	for (int i = 1; i < histSize; ++i) {
		cv::line(histImage,
			cv::Point(binW * (i - 1), histH - cvRound(histogram.at<float>(i - 1))),
			cv::Point(binW * (i), histH - cvRound(histogram.at<float>(i))),
			cv::Scalar(255, 100, 0), 2);
	}

	showImage(histImage, name, x, y);
}

int main() {
	cv::Mat repImage = cv::imread("Samples/reptile.jpg");

	if (repImage.empty()) {
		std::cerr << "Unable to read the file" << std::endl;
		std::cin.get();
		return -1;
	}

	showImage(repImage, "Original Image", 0, 0);

	cv::Mat grayImage;
	cv::cvtColor(repImage, grayImage, cv::COLOR_BGR2GRAY);
	showImage(grayImage, "Grayscale Image", 350, 0);

	cv::Mat resizedImage(200, 200, repImage.type());
	cv::resize(repImage, resizedImage, resizedImage.size());
	showImage(resizedImage, "Resized Image", 700, 0);

	cv::Mat dogImage = cv::imread("Samples/dog.jpg");
	if (dogImage.empty()) {
		std::cerr << "Unable to read the file" << std::endl;
		std::cin.get();
		return -1;
	}

	cv::Mat blurImage;
	cv::blur(dogImage, blurImage, cv::Size(13, 13));
	showImage(blurImage, "Blur", 1050, 0);

	cv::Mat cannyImage;
	cv::Canny(dogImage, cannyImage, 100, 100);
	showImage(cannyImage, "Canny", 0, 350);

	cv::Mat laplacianImage, scaledLaplacianImage;
	cv::Laplacian(grayImage, laplacianImage, CV_16S, 3);
	cv::convertScaleAbs(laplacianImage, scaledLaplacianImage);
	showImage(scaledLaplacianImage, "Laplacian Image", 350, 350);

	cv::Mat brightImage;
	dogImage.copyTo(brightImage);
	changeBrightness(dogImage, brightImage, 100);
	showImage(brightImage, "Brightness", 700, 350);

	ThresholdData trackbarData;
	trackbarData.image = dogImage;

	createWindow("Binarization", 1050, 350);
	cv::createTrackbar("Threshold", "Binarization", &trackbarData.thresholdValue, trackbarData.maxValue, onThreshold, &trackbarData);

	createHistogram(grayImage, "Grayscale Histogram", 0, 700);

	cv::Mat equalizedImage;
	equalizeHist(grayImage, equalizedImage);
	showImage(equalizedImage, "Equalized Image", 1200, 700);

	createHistogram(equalizedImage, "Equalized Histoogram ", 600, 700);

	cv::waitKey(0);

	createWindow("Src", 300, 300);
	createWindow("Dst", 900, 300);

	cv::Mat srcFrame;
	cv::Mat resizedFrame;
	cv::Mat dstFrame;
	cv::Size size(640, 480);

	cv::VideoCapture capture("Samples/bird.mp4");
	capture >> srcFrame;
	cv::resize(srcFrame, resizedFrame, size);

	while (cv::waitKey(40) != 27 && !srcFrame.empty()) {
		imshow("Src", resizedFrame);
		Canny(resizedFrame, dstFrame, 100, 100);
		imshow("Dst", dstFrame);

		capture >> srcFrame;
		cv::resize(srcFrame, resizedFrame, size);
	}

	return 0;
}