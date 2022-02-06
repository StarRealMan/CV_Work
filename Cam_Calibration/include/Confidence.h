#ifndef __CONFIDENCE_H__
#define __CONFIDENCE_H__

#include <iostream>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

class Confidence
{
public:
    typedef std::shared_ptr<Confidence> Ptr;

    Confidence(uint8_t layer_num, float start_sigma, float end_sigma, uint8_t width, uint8_t height);
    ~Confidence();
    void ProcessImage(cv::Mat image);
    std::vector<cv::Point2f> GetMean(void);
    std::vector<cv::Point2f> GetVar(void);

private:
    uint8_t _layer_size;
    uint8_t _width;
    uint8_t _height;

    std::vector<float> _sigma_vec;

    std::vector<cv::Point2f> _corner_mean_vec;
    std::vector<cv::Point2f> _corner_var_vec;
    std::vector<std::vector<cv::Point2f>> _corner_image_vec;
};


#endif