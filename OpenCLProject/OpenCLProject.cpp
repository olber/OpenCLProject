#include "clu/openCLGLUtilities.hpp"
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

void Cleanup(cl_mem* memObjects, cl_uint memObjCount)
{
	if (memObjects != nullptr)
	{
		for (cl_uint i = 0; i < memObjCount; i++)
		{
			if (memObjects[i] != nullptr)
				clReleaseMemObject(memObjects[i]);
		}
	}
}

Mat GrayScaleImage(Mat img, cl::Context context, cl::Program program, cl::CommandQueue queue)
{
	cl_int errNum;
	cv::Mat transformedImg;
	cv::cvtColor(img, transformedImg, CV_BGR2RGBA);
	cl_image_format inFormat, outFormat;
	inFormat.image_channel_data_type = CL_UNORM_INT8;
	inFormat.image_channel_order = CL_RGBA;

	size_t globalWorkSize = transformedImg.cols * transformedImg.rows;
	size_t arrSize = globalWorkSize * 4;

	cl_mem memoryObjects[2];

	memoryObjects[0] = clCreateImage2D(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &inFormat, transformedImg.cols, transformedImg.rows, transformedImg.cols * 4, transformedImg.data, &errNum);

	auto data = new float[globalWorkSize];
	outFormat.image_channel_data_type = CL_FLOAT;
	outFormat.image_channel_order = CL_R;
	memoryObjects[1] = clCreateImage2D(context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &outFormat, transformedImg.cols, transformedImg.rows, transformedImg.cols * 4, data, &errNum);

	cl_kernel kernel = clCreateKernel(program(), "grayScale", &errNum);

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memoryObjects[0]);
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memoryObjects[1]);

	size_t glWork[2] = { transformedImg.cols, transformedImg.rows };
	errNum = clEnqueueNDRangeKernel(queue(), kernel, 2, 0, glWork, NULL, NULL, NULL, NULL);
	errNum = clFinish(queue());
		
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { transformedImg.cols, transformedImg.rows, 1 };
	errNum = clEnqueueReadImage(queue(), memoryObjects[1], CL_TRUE, origin, region, transformedImg.cols * 4, 0, data, 0, NULL, NULL);

	cv::Size size;
	size.height = transformedImg.rows;
	size.width = transformedImg.cols;

	cv::Mat outputImageGreyscale(size, CV_32FC1, data);
	Cleanup(memoryObjects, 2);
	return outputImageGreyscale;
}



void video(cl::Context context, cl::Program program, cl::CommandQueue queue) {

	namedWindow("Video");
	Mat frame;
	Mat picture;

	VideoCapture cap(0);
	for (;;)
	{
		if (char(waitKey(1)) != 'q' &&cap.isOpened())
		{
			cap.operator >> (frame);
		}
		frame = GrayScaleImage(frame, context, program, queue);
		imshow("Video", frame);
		delete frame.data;
	}
	cap.release();
}

int main(int argc, char *argv[]) {
	cl::Context context = createCLContextFromArguments(argc, argv);
	cl::Program program = buildProgramFromSource(context, "gr_sc.cl");
	VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
	video(context, program, queue);
}

