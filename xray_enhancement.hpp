#ifndef XRAY_ENHANCEMENT_HPP
#define XRAY_ENHANCEMENT_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class XRayEnhancer {
private:
    // 高斯滤波器，用于Retinex
    cv::Mat createGaussianKernel(int size, double sigma);
    
    // 多尺度Retinex的单尺度处理
    cv::Mat singleScaleRetinex(const cv::Mat& image, double sigma);
    
    // 对数变换和归一化
    cv::Mat logTransform(const cv::Mat& image);
    cv::Mat normalizeImage(const cv::Mat& image);

public:
    /**
     * CLAHE增强 - 对比度限制自适应直方图均衡化
     * @param image 输入图像 (CV_8UC1 或 CV_32FC1)
     * @param clipLimit 对比度限制 (默认3.0)
     * @param tileGridSize 网格大小 (默认8x8)
     * @return 增强后的图像
     */
    cv::Mat claheEnhancement(const cv::Mat& image, 
                            double clipLimit = 3.0, 
                            cv::Size tileGridSize = cv::Size(8, 8));
    
    /**
     * 多尺度Retinex增强
     * @param image 输入图像 (CV_8UC1 或 CV_32FC1)
     * @param scales 尺度参数数组 (默认{15, 80, 250})
     * @return 增强后的图像
     */
    cv::Mat multiScaleRetinex(const cv::Mat& image, 
                             const std::vector<double>& scales = {15.0, 80.0, 250.0});
    
    /**
     * 组合增强 - CLAHE + Retinex
     * @param image 输入图像
     * @param useClahe 是否使用CLAHE
     * @param useRetinex 是否使用Retinex
     * @param claheClipLimit CLAHE对比度限制
     * @param retinexScales Retinex尺度参数
     * @return 增强后的图像
     */
    cv::Mat combinedEnhancement(const cv::Mat& image,
                               bool useClahe = true,
                               bool useRetinex = true,
                               double claheClipLimit = 3.0,
                               const std::vector<double>& retinexScales = {15.0, 80.0, 250.0});
};

#endif // XRAY_ENHANCEMENT_HPP