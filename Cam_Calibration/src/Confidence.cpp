#include "Confidence.h"

Confidence::Confidence(uint8_t layer_size, float start_sigma, float end_sigma, uint8_t width, uint8_t height)
{
    _layer_size = layer_size;
    float step_sigma = (end_sigma - start_sigma) / layer_size;
    float sigma = start_sigma;

    _width = width;
    _height = height;

    for(uint layer_num = 0; layer_num < layer_size; layer_num++)
    {
        _sigma_vec.push_back(sigma);
        sigma += step_sigma;
    }

    for(uint point_num = 0; point_num < _width * _height; point_num++)
    {
        _corner_mean_vec.push_back(cv::Point2f(0.0, 0.0));
        _corner_var_vec.push_back(cv::Point2f(0.0, 0.0));
        _corner_image_vec.push_back(std::vector<cv::Point2f>());
    }
}

Confidence::~Confidence()
{
    
}

void Confidence::ProcessImage(cv::Mat image)
{
    for(uint layer_num = 0; layer_num < _layer_size; layer_num++)
    {
        cv::Mat res;

        float sigma = _sigma_vec[layer_num];
        cv::GaussianBlur(image, res, cv::Size(5, 5), sigma);

        std::vector<cv::Point2f> this_corner_vec;
        cv::findChessboardCorners(res, cv::Size(_width, _height), this_corner_vec);

        for(uint point_num = 0; point_num < _width * _height; point_num++)
        {
            cv::Point2f this_corner = this_corner_vec[point_num];
            
            _corner_mean_vec[point_num].x += this_corner.x;
            _corner_mean_vec[point_num].y += this_corner.y;

            _corner_image_vec[point_num].push_back(this_corner);
        }
    }

    for(uint point_num = 0; point_num < _width * _height; point_num++)
    {

        cv::Point2f mean_point = _corner_mean_vec[point_num];

        for(uint layer_num = 0; layer_num < _layer_size; layer_num++)
        {
            cv::Point2f point = _corner_image_vec[point_num][layer_num];

            float var_point_x = (point.x - mean_point.x) * (point.x - mean_point.x);
            float var_point_y = (point.y - mean_point.y) * (point.y - mean_point.y);

            _corner_var_vec[point_num].x += var_point_x;
            _corner_var_vec[point_num].y += var_point_y;
        }

        _corner_mean_vec[point_num].x /= _layer_size;
        _corner_mean_vec[point_num].y /= _layer_size;
        _corner_var_vec[point_num].x /= _layer_size;
        _corner_var_vec[point_num].y /= _layer_size;
    }

}

std::vector<cv::Point2f> Confidence::GetMean()
{
    return _corner_mean_vec;
}

std::vector<cv::Point2f> Confidence::GetVar()
{
    return _corner_var_vec;
}