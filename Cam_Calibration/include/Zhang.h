#ifndef __ZHANG_H__
#define __ZHANG_H__

#include <iostream>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#include "Confidence.h"

class Zhang
{
public:
    typedef std::shared_ptr<Confidence> Ptr;

    Zhang(uint8_t _sample_size, uint8_t layer_size, uint8_t width, uint8_t height, cv::Size image_size);
    ~Zhang();

    void LoadImageVec(std::vector<cv::Mat> image_vec);
    std::vector<cv::Point2f> GenChessboard();
    void CalConfidence();
    void OptimizeHomo();
    void DecomposeHomo();


private:
    uint8_t _sample_size;
    uint8_t _layer_size;
    uint8_t _width;
    uint8_t _height;

    cv::Size _image_size;

    std::vector<cv::Mat> _image_vec;
    std::vector<cv::Point2f> _chess_board_vec;

    std::vector<std::vector<cv::Point2f>> _image_corner_mean_vec;
    std::vector<std::vector<cv::Point2f>> _image_corner_var_vec;

    std::vector<cv::Mat> _homo_vec;
};

#endif