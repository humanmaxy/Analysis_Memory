#include "xray_enhancement.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "使用方法: " << argv[0] << " <输入图像> <输出图像> [选项]" << std::endl;
        std::cout << "选项:" << std::endl;
        std::cout << "  --clahe-only    仅使用CLAHE增强" << std::endl;
        std::cout << "  --retinex-only  仅使用Retinex增强" << std::endl;
        std::cout << "  --combined      使用组合增强 (默认)" << std::endl;
        std::cout << "  --clip-limit <value>  CLAHE对比度限制 (默认3.0)" << std::endl;
        return -1;
    }
    
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    
    // 解析命令行参数
    bool useClahe = true;
    bool useRetinex = true;
    double clipLimit = 3.0;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--clahe-only") {
            useClahe = true;
            useRetinex = false;
        } else if (arg == "--retinex-only") {
            useClahe = false;
            useRetinex = true;
        } else if (arg == "--combined") {
            useClahe = true;
            useRetinex = true;
        } else if (arg == "--clip-limit" && i + 1 < argc) {
            clipLimit = std::stod(argv[i + 1]);
            i++; // 跳过下一个参数
        }
    }
    
    // 读取输入图像
    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "错误: 无法读取图像 " << inputPath << std::endl;
        return -1;
    }
    
    std::cout << "输入图像尺寸: " << inputImage.cols << "x" << inputImage.rows << std::endl;
    std::cout << "图像类型: " << inputImage.type() << std::endl;
    
    // 创建增强器
    XRayEnhancer enhancer;
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    
    if (useClahe && !useRetinex) {
        std::cout << "执行CLAHE增强..." << std::endl;
        result = enhancer.claheEnhancement(inputImage, clipLimit);
    } else if (!useClahe && useRetinex) {
        std::cout << "执行Retinex增强..." << std::endl;
        cv::Mat retinexResult = enhancer.multiScaleRetinex(inputImage);
        retinexResult.convertTo(result, CV_8U, 255.0);
    } else {
        std::cout << "执行组合增强..." << std::endl;
        result = enhancer.combinedEnhancement(inputImage, useClahe, useRetinex, clipLimit);
    }
    
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "处理完成，用时: " << duration.count() << " ms" << std::endl;
    
    // 保存结果
    bool success = cv::imwrite(outputPath, result);
    if (success) {
        std::cout << "结果已保存到: " << outputPath << std::endl;
    } else {
        std::cerr << "错误: 无法保存图像到 " << outputPath << std::endl;
        return -1;
    }
    
    // 显示图像信息
    std::cout << "输出图像尺寸: " << result.cols << "x" << result.rows << std::endl;
    std::cout << "输出图像类型: " << result.type() << std::endl;
    
    // 可选：显示图像（如果编译时启用了GUI支持）
    #ifdef OPENCV_GUI_ENABLED
    cv::namedWindow("原始图像", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("增强图像", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("原始图像", inputImage);
    cv::imshow("增强图像", result);
    
    std::cout << "按任意键退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    #endif
    
    return 0;
}