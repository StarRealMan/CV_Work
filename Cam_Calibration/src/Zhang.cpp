#include "Zhang.h"

Zhang::Zhang(uint8_t sample_size, uint8_t layer_size, uint8_t width, uint8_t height, cv::Size image_size)
{
    _sample_size = sample_size;
    _layer_size = layer_size;

    _width = width;
    _height = height;

    _image_size = image_size;
}

Zhang::~Zhang()
{

}

void Zhang::LoadImageVec(std::vector<cv::Mat> image_vec)
{
    _image_vec = image_vec;
}

std::vector<cv::Point2f> Zhang::GenChessboard()
{

}

void Zhang::CalConfidence()
{
    for(uint8_t image_num = 0; image_num < _sample_size; image_num++)
    {
        cv::Mat image = _image_vec[image_num];
        
        Confidence::Ptr conf = std::make_shared<Confidence>(_layer_size, 0, 5, _width, _height);
        conf->ProcessImage(image);
        _image_corner_mean_vec.push_back(conf->GetMean());
        _image_corner_var_vec.push_back(conf->GetVar());
    }
}

void Zhang::OptimizeHomo()
{
    _chess_board_vec = GenChessboard();
    for(uint8_t image_num = 0; image_num < _sample_size; image_num++)
    {
        cv::Mat Homo = cv::findHomography(_chess_board_vec, _image_corner_mean_vec[image_num]);
        _homo_vec.push_back(Homo);
    }

    // Optimize!
}

void Zhang::DecomposeHomo()
{

}
