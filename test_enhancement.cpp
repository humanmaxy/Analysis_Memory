#include "xray_enhancement.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 创建测试图像
cv::Mat createTestImage(int width = 512, int height = 512) {
    cv::Mat testImage = cv::Mat::zeros(height, width, CV_8U);
    
    // 添加大结构
    cv::ellipse(testImage, cv::Point(width/2, height/2), 
                cv::Size(100, 150), 0, 0, 360, cv::Scalar(180), -1);
    
    // 添加细节结构
    for (int i = 0; i < 5; i++) {
        cv::Point center(width/4 + i * width/8, height/3);
        cv::circle(testImage, center, 20, cv::Scalar(120), 2);
    }
    
    // 添加细线
    for (int i = 0; i < 10; i++) {
        int y = height/4 + i * 20;
        cv::line(testImage, cv::Point(width/8, y), 
                cv::Point(7*width/8, y), cv::Scalar(150), 1);
    }
    
    // 添加噪声
    cv::Mat noise;
    cv::randn(noise, 0, 15);
    noise.convertTo(noise, CV_8U);
    testImage += noise;
    
    return testImage;
}

// 计算图像质量指标
void calculateMetrics(const cv::Mat& original, const cv::Mat& enhanced) {
    // 转换为浮点数
    cv::Mat origFloat, enhFloat;
    original.convertTo(origFloat, CV_32F, 1.0/255.0);
    enhanced.convertTo(enhFloat, CV_32F, 1.0/255.0);
    
    // 计算MSE
    cv::Mat diff;
    cv::subtract(origFloat, enhFloat, diff);
    cv::multiply(diff, diff, diff);
    cv::Scalar mse = cv::mean(diff);
    
    // 计算PSNR
    double psnr = 0.0;
    if (mse[0] > 0) {
        psnr = 20 * std::log10(1.0 / std::sqrt(mse[0]));
    }
    
    // 计算对比度
    cv::Scalar origMean, origStddev;
    cv::Scalar enhMean, enhStddev;
    cv::meanStdDev(origFloat, origMean, origStddev);
    cv::meanStdDev(enhFloat, enhMean, enhStddev);
    
    double contrastRatio = enhStddev[0] / (origStddev[0] + 1e-10);
    
    std::cout << "图像质量指标:" << std::endl;
    std::cout << "  MSE: " << mse[0] << std::endl;
    std::cout << "  PSNR: " << psnr << " dB" << std::endl;
    std::cout << "  对比度改善: " << contrastRatio << "x" << std::endl;
    std::cout << "  原始图像标准差: " << origStddev[0] << std::endl;
    std::cout << "  增强图像标准差: " << enhStddev[0] << std::endl;
}

int main() {
    std::cout << "X光图像增强算法测试" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // 创建测试图像
    cv::Mat testImage = createTestImage();
    std::cout << "创建测试图像: " << testImage.cols << "x" << testImage.rows << std::endl;
    
    // 保存测试图像
    cv::imwrite("test_original.png", testImage);
    std::cout << "测试图像已保存: test_original.png" << std::endl;
    
    // 创建增强器
    XRayEnhancer enhancer;
    
    // 测试CLAHE增强
    std::cout << "\n测试CLAHE增强..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat claheResult = enhancer.claheEnhancement(testImage, 3.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CLAHE处理时间: " << duration.count() << " ms" << std::endl;
    cv::imwrite("test_clahe.png", claheResult);
    calculateMetrics(testImage, claheResult);
    
    // 测试Retinex增强
    std::cout << "\n测试Retinex增强..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat retinexResult = enhancer.multiScaleRetinex(testImage);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 转换为8位保存
    cv::Mat retinex8bit;
    retinexResult.convertTo(retinex8bit, CV_8U, 255.0);
    
    std::cout << "Retinex处理时间: " << duration.count() << " ms" << std::endl;
    cv::imwrite("test_retinex.png", retinex8bit);
    calculateMetrics(testImage, retinex8bit);
    
    // 测试组合增强
    std::cout << "\n测试组合增强..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat combinedResult = enhancer.combinedEnhancement(testImage);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "组合增强处理时间: " << duration.count() << " ms" << std::endl;
    cv::imwrite("test_combined.png", combinedResult);
    calculateMetrics(testImage, combinedResult);
    
    // 测试不同参数
    std::cout << "\n测试不同CLAHE参数..." << std::endl;
    std::vector<double> clipLimits = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    for (double clipLimit : clipLimits) {
        cv::Mat result = enhancer.claheEnhancement(testImage, clipLimit);
        std::string filename = "test_clahe_clip_" + std::to_string((int)clipLimit) + ".png";
        cv::imwrite(filename, result);
        
        std::cout << "ClipLimit " << clipLimit << ": ";
        cv::Scalar mean, stddev;
        cv::meanStdDev(result, mean, stddev);
        std::cout << "对比度 = " << stddev[0] << std::endl;
    }
    
    // 测试不同Retinex尺度
    std::cout << "\n测试不同Retinex尺度..." << std::endl;
    std::vector<std::vector<double>> scalesSets = {
        {15.0, 80.0, 250.0},  // 默认
        {10.0, 50.0, 200.0},  // 小尺度
        {20.0, 100.0, 300.0}, // 大尺度
        {15.0, 80.0}          // 双尺度
    };
    
    for (size_t i = 0; i < scalesSets.size(); i++) {
        cv::Mat result = enhancer.multiScaleRetinex(testImage, scalesSets[i]);
        cv::Mat result8bit;
        result.convertTo(result8bit, CV_8U, 255.0);
        
        std::string filename = "test_retinex_scale_" + std::to_string(i) + ".png";
        cv::imwrite(filename, result8bit);
        
        std::cout << "尺度组 " << i << " (";
        for (double scale : scalesSets[i]) {
            std::cout << scale << " ";
        }
        std::cout << "): ";
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(result8bit, mean, stddev);
        std::cout << "对比度 = " << stddev[0] << std::endl;
    }
    
    std::cout << "\n测试完成！所有结果图像已保存。" << std::endl;
    std::cout << "文件列表:" << std::endl;
    std::cout << "- test_original.png: 原始测试图像" << std::endl;
    std::cout << "- test_clahe.png: CLAHE增强结果" << std::endl;
    std::cout << "- test_retinex.png: Retinex增强结果" << std::endl;
    std::cout << "- test_combined.png: 组合增强结果" << std::endl;
    std::cout << "- test_clahe_clip_*.png: 不同ClipLimit的CLAHE结果" << std::endl;
    std::cout << "- test_retinex_scale_*.png: 不同尺度的Retinex结果" << std::endl;
    
    return 0;
}