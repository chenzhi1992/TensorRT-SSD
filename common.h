#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}
using namespace std;

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
    public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
	}
};

std::string locateFile(const std::string& input, const std::vector<std::string> & directories);
void readPGMFile(const std::string& fileName,  uint8_t *buffer, int inH, int inW);
void Forward_DetectionOutputLayer(float* loc_data, float* conf_data, float* prior_data, int  num_priors_, int num_classes_, vector<vector<float> >* detecions);
#endif // _TRT_COMMON_H_
