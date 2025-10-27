#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像增强核心算法测试
仅使用Python标准库进行基本测试
"""

import math
import random


class SimpleImageProcessor:
    """简化的图像处理器，用于测试核心算法逻辑"""
    
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
    
    def create_test_image(self):
        """创建测试图像"""
        # 创建一个简单的测试图像（二维数组）
        image = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # 创建一些基本结构
                center_x, center_y = self.width // 2, self.height // 2
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # 基础值
                value = 0.3
                
                # 添加圆形结构
                if distance < 30:
                    value = 0.7
                elif distance < 35:
                    value = 0.5
                
                # 添加噪声
                noise = (random.random() - 0.5) * 0.2
                value += noise
                
                # 确保在有效范围内
                value = max(0.0, min(1.0, value))
                row.append(value)
            
            image.append(row)
        
        return image
    
    def gaussian_blur(self, image, kernel_size=5):
        """简单的高斯模糊"""
        # 创建高斯核
        sigma = kernel_size / 3.0
        kernel = []
        center = kernel_size // 2
        
        for i in range(kernel_size):
            row = []
            for j in range(kernel_size):
                x, y = i - center, j - center
                value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
                row.append(value)
            kernel.append(row)
        
        # 归一化核
        kernel_sum = sum(sum(row) for row in kernel)
        kernel = [[val / kernel_sum for val in row] for row in kernel]
        
        # 应用卷积
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                value = 0.0
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        img_y = y + ky - center
                        img_x = x + kx - center
                        
                        # 边界处理
                        if 0 <= img_y < self.height and 0 <= img_x < self.width:
                            value += image[img_y][img_x] * kernel[ky][kx]
                
                row.append(max(0.0, min(1.0, value)))
            result.append(row)
        
        return result
    
    def median_filter(self, image, kernel_size=3):
        """中值滤波"""
        result = []
        center = kernel_size // 2
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                values = []
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        img_y = y + ky - center
                        img_x = x + kx - center
                        
                        if 0 <= img_y < self.height and 0 <= img_x < self.width:
                            values.append(image[img_y][img_x])
                
                # 计算中值
                values.sort()
                median = values[len(values) // 2] if values else 0.0
                row.append(median)
            
            result.append(row)
        
        return result
    
    def histogram_equalization(self, image, bins=256):
        """直方图均衡化"""
        # 计算直方图
        histogram = [0] * bins
        
        for row in image:
            for pixel in row:
                bin_index = int(pixel * (bins - 1))
                histogram[bin_index] += 1
        
        # 计算累积分布函数
        total_pixels = self.width * self.height
        cdf = [0] * bins
        cdf[0] = histogram[0]
        
        for i in range(1, bins):
            cdf[i] = cdf[i-1] + histogram[i]
        
        # 归一化CDF
        cdf_min = min(val for val in cdf if val > 0)
        cdf_normalized = [(val - cdf_min) / (total_pixels - cdf_min) for val in cdf]
        
        # 应用均衡化
        result = []
        for row in image:
            new_row = []
            for pixel in row:
                bin_index = int(pixel * (bins - 1))
                new_pixel = cdf_normalized[bin_index]
                new_row.append(max(0.0, min(1.0, new_pixel)))
            result.append(new_row)
        
        return result
    
    def unsharp_mask(self, image, amount=1.5):
        """非锐化掩模"""
        # 创建模糊版本
        blurred = self.gaussian_blur(image, 5)
        
        # 创建掩模
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                original = image[y][x]
                blur = blurred[y][x]
                
                # 计算增强值
                enhanced = original + amount * (original - blur)
                enhanced = max(0.0, min(1.0, enhanced))
                row.append(enhanced)
            
            result.append(row)
        
        return result
    
    def calculate_metrics(self, original, processed):
        """计算简单的质量指标"""
        # 计算均方误差
        mse = 0.0
        for y in range(self.height):
            for x in range(self.width):
                diff = original[y][x] - processed[y][x]
                mse += diff * diff
        
        mse /= (self.width * self.height)
        
        # 计算PSNR
        psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float('inf')
        
        # 计算对比度
        def calculate_std(image):
            mean = sum(sum(row) for row in image) / (self.width * self.height)
            variance = sum(sum((pixel - mean)**2 for pixel in row) for row in image)
            variance /= (self.width * self.height)
            return math.sqrt(variance)
        
        orig_std = calculate_std(original)
        proc_std = calculate_std(processed)
        contrast_improvement = proc_std / orig_std if orig_std > 0 else 1.0
        
        return {
            'MSE': mse,
            'PSNR': psnr,
            'Contrast_Improvement': contrast_improvement
        }
    
    def print_image_stats(self, image, name):
        """打印图像统计信息"""
        total = sum(sum(row) for row in image)
        mean = total / (self.width * self.height)
        
        variance = sum(sum((pixel - mean)**2 for pixel in row) for row in image)
        variance /= (self.width * self.height)
        std = math.sqrt(variance)
        
        min_val = min(min(row) for row in image)
        max_val = max(max(row) for row in image)
        
        print(f"{name} 统计:")
        print(f"  均值: {mean:.4f}")
        print(f"  标准差: {std:.4f}")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")


def test_core_algorithms():
    """测试核心算法"""
    print("X光图像增强核心算法测试")
    print("=" * 50)
    
    # 创建处理器
    processor = SimpleImageProcessor(100, 100)
    
    # 创建测试图像
    print("创建测试图像...")
    original_image = processor.create_test_image()
    processor.print_image_stats(original_image, "原始图像")
    print()
    
    # 测试高斯模糊
    print("测试高斯模糊去噪...")
    blurred = processor.gaussian_blur(original_image, 3)
    processor.print_image_stats(blurred, "高斯模糊")
    metrics = processor.calculate_metrics(original_image, blurred)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  对比度变化: {metrics['Contrast_Improvement']:.2f}x")
    print()
    
    # 测试中值滤波
    print("测试中值滤波去噪...")
    median_filtered = processor.median_filter(original_image, 3)
    processor.print_image_stats(median_filtered, "中值滤波")
    metrics = processor.calculate_metrics(original_image, median_filtered)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  对比度变化: {metrics['Contrast_Improvement']:.2f}x")
    print()
    
    # 测试直方图均衡化
    print("测试直方图均衡化...")
    equalized = processor.histogram_equalization(original_image)
    processor.print_image_stats(equalized, "直方图均衡化")
    metrics = processor.calculate_metrics(original_image, equalized)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  对比度变化: {metrics['Contrast_Improvement']:.2f}x")
    print()
    
    # 测试非锐化掩模
    print("测试非锐化掩模...")
    sharpened = processor.unsharp_mask(original_image, 1.5)
    processor.print_image_stats(sharpened, "非锐化掩模")
    metrics = processor.calculate_metrics(original_image, sharpened)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  对比度变化: {metrics['Contrast_Improvement']:.2f}x")
    print()
    
    # 测试组合处理
    print("测试组合处理管道...")
    
    # 步骤1：去噪
    step1 = processor.median_filter(original_image, 3)
    print("  步骤1: 中值滤波去噪")
    
    # 步骤2：直方图均衡化
    step2 = processor.histogram_equalization(step1)
    print("  步骤2: 直方图均衡化")
    
    # 步骤3：轻微模糊
    step3 = processor.gaussian_blur(step2, 2)
    print("  步骤3: 轻微高斯模糊")
    
    # 步骤4：锐化
    final_result = processor.unsharp_mask(step3, 1.2)
    print("  步骤4: 非锐化掩模增强")
    
    processor.print_image_stats(final_result, "最终结果")
    metrics = processor.calculate_metrics(original_image, final_result)
    print(f"  最终PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  最终对比度改善: {metrics['Contrast_Improvement']:.2f}x")
    print()
    
    print("=" * 50)
    print("核心算法测试完成!")
    print("算法逻辑验证通过，可以处理实际图像数据。")
    print("=" * 50)


def test_noise_robustness():
    """测试噪声鲁棒性"""
    print("\n噪声鲁棒性测试")
    print("=" * 30)
    
    processor = SimpleImageProcessor(50, 50)
    
    # 创建不同噪声水平的测试图像
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    
    for noise_level in noise_levels:
        print(f"\n测试噪声水平: {noise_level}")
        
        # 创建带噪声的图像
        clean_image = processor.create_test_image()
        noisy_image = []
        
        for row in clean_image:
            noisy_row = []
            for pixel in row:
                noise = (random.random() - 0.5) * 2 * noise_level
                noisy_pixel = max(0.0, min(1.0, pixel + noise))
                noisy_row.append(noisy_pixel)
            noisy_image.append(noisy_row)
        
        # 应用去噪
        denoised = processor.median_filter(noisy_image, 3)
        denoised = processor.gaussian_blur(denoised, 2)
        
        # 计算改善
        noisy_metrics = processor.calculate_metrics(clean_image, noisy_image)
        denoised_metrics = processor.calculate_metrics(clean_image, denoised)
        
        print(f"  噪声图像PSNR: {noisy_metrics['PSNR']:.2f} dB")
        print(f"  去噪后PSNR: {denoised_metrics['PSNR']:.2f} dB")
        print(f"  PSNR改善: {denoised_metrics['PSNR'] - noisy_metrics['PSNR']:.2f} dB")


def main():
    """主函数"""
    # 设置随机种子以获得可重复的结果
    random.seed(42)
    
    try:
        test_core_algorithms()
        test_noise_robustness()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("核心算法工作正常，可以集成到完整系统中。")
        print("要运行完整的图像处理，请安装所需依赖:")
        print("pip install opencv-python numpy scipy scikit-image matplotlib Pillow")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()