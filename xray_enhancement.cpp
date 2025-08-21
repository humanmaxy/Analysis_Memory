#include "xray_enhancement.hpp"
#include <algorithm>
#include <iostream>

cv::Mat XRayEnhancer::createGaussianKernel(int size, double sigma) {
    cv::Mat kernel(size, size, CV_64F);
    int center = size / 2;
    double sum = 0.0;
    
    // 生成高斯核
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - center;
            int y = j - center;
            double value = std::exp(-(x*x + y*y) / (2.0 * sigma * sigma));
            kernel.at<double>(i, j) = value;
            sum += value;
        }
    }
    
    // 归一化
    kernel /= sum;
    return kernel;
}

cv::Mat XRayEnhancer::logTransform(const cv::Mat& image) {
    cv::Mat result;
    cv::Mat floatImage;
    
    // 转换为浮点数
    if (image.type() != CV_32F) {
        image.convertTo(floatImage, CV_32F, 1.0/255.0);
    } else {
        floatImage = image.clone();
    }
    
    // 防止log(0)，添加小的常数
    cv::Mat safeImage = floatImage + 1e-6;
    
    // 对数变换
    cv::log(safeImage, result);
    
    return result;
}

cv::Mat XRayEnhancer::normalizeImage(const cv::Mat& image) {
    cv::Mat result;
    double minVal, maxVal;
    
    // 找到最小值和最大值
    cv::minMaxLoc(image, &minVal, &maxVal);
    
    // 归一化到[0, 1]
    if (maxVal > minVal) {
        result = (image - minVal) / (maxVal - minVal);
    } else {
        result = image.clone();
    }
    
    return result;
}

cv::Mat XRayEnhancer::singleScaleRetinex(const cv::Mat& image, double sigma) {
    cv::Mat floatImage;
    
    // 转换为浮点数
    if (image.type() != CV_32F) {
        image.convertTo(floatImage, CV_32F, 1.0/255.0);
    } else {
        floatImage = image.clone();
    }
    
    // 防止log(0)
    cv::Mat safeImage = floatImage + 1e-6;
    
    // 高斯模糊作为环绕函数
    cv::Mat surround;
    int kernelSize = static_cast<int>(6 * sigma + 1);
    if (kernelSize % 2 == 0) kernelSize++;
    
    cv::GaussianBlur(safeImage, surround, cv::Size(kernelSize, kernelSize), sigma, sigma);
    
    // 防止log(0)
    surround = surround + 1e-6;
    
    // 计算单尺度Retinex
    cv::Mat logImage, logSurround, result;
    cv::log(safeImage, logImage);
    cv::log(surround, logSurround);
    
    result = logImage - logSurround;
    
    return result;
}

cv::Mat XRayEnhancer::claheEnhancement(const cv::Mat& image, double clipLimit, cv::Size tileGridSize) {
    cv::Mat result;
    cv::Mat grayImage;
    
    // 确保输入是单通道图像
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }
    
    // 转换为8位图像（如果不是的话）
    cv::Mat uint8Image;
    if (grayImage.type() != CV_8U) {
        if (grayImage.type() == CV_32F || grayImage.type() == CV_64F) {
            // 浮点图像，假设范围是[0,1]
            grayImage.convertTo(uint8Image, CV_8U, 255.0);
        } else {
            grayImage.convertTo(uint8Image, CV_8U);
        }
    } else {
        uint8Image = grayImage.clone();
    }
    
    // 创建CLAHE对象
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(tileGridSize);
    
    // 应用CLAHE
    clahe->apply(uint8Image, result);
    
    return result;
}

cv::Mat XRayEnhancer::multiScaleRetinex(const cv::Mat& image, const std::vector<double>& scales) {
    std::vector<cv::Mat> retinexResults;
    
    // 对每个尺度计算单尺度Retinex
    for (double scale : scales) {
        cv::Mat singleResult = singleScaleRetinex(image, scale);
        retinexResults.push_back(singleResult);
    }
    
    // 平均所有尺度的结果
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    for (const auto& retinexResult : retinexResults) {
        result += retinexResult;
    }
    result /= static_cast<double>(scales.size());
    
    // 归一化到[0, 1]
    result = normalizeImage(result);
    
    return result;
}

cv::Mat XRayEnhancer::combinedEnhancement(const cv::Mat& image,
                                         bool useClahe,
                                         bool useRetinex,
                                         double claheClipLimit,
                                         const std::vector<double>& retinexScales) {
    cv::Mat result = image.clone();
    cv::Mat workingImage;
    
    // 转换为浮点数进行处理
    if (result.type() != CV_32F) {
        result.convertTo(workingImage, CV_32F, 1.0/255.0);
    } else {
        workingImage = result.clone();
    }
    
    // 步骤1: CLAHE增强
    if (useClahe) {
        std::cout << "应用CLAHE增强..." << std::endl;
        cv::Mat claheResult = claheEnhancement(workingImage, claheClipLimit);
        
        // 转换回浮点数
        claheResult.convertTo(workingImage, CV_32F, 1.0/255.0);
    }
    
    // 步骤2: Retinex增强
    if (useRetinex) {
        std::cout << "应用多尺度Retinex增强..." << std::endl;
        cv::Mat retinexResult = multiScaleRetinex(workingImage, retinexScales);
        
        // 与CLAHE结果融合（如果都启用）
        if (useClahe) {
            // 加权融合：70% Retinex + 30% CLAHE
            workingImage = 0.7 * retinexResult + 0.3 * workingImage;
        } else {
            workingImage = retinexResult;
        }
    }
    
    // 确保结果在有效范围内
    cv::Mat finalResult;
    workingImage = cv::max(workingImage, 0.0);
    workingImage = cv::min(workingImage, 1.0);
    
    // 转换回8位图像
    workingImage.convertTo(finalResult, CV_8U, 255.0);
    
    return finalResult;
}