#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法对比测试工具
比较不同增强算法的细节保持能力和精度
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from advanced_xray_enhancer import AdvancedXRayEnhancer
from xray_enhancement import XRayEnhancer  # 原始版本
import os


class AlgorithmComparison:
    """算法对比工具"""
    
    def __init__(self):
        self.test_image = None
        self.results = {}
        
    def create_test_image(self, size=(512, 512)):
        """创建测试图像，包含各种细节结构"""
        height, width = size
        image = np.zeros((height, width), dtype=np.float64)
        
        # 添加大尺度结构
        cv2.ellipse(image, (width//2, height//2), (100, 150), 0, 0, 360, 0.7, -1)
        
        # 添加中等尺度细节
        for i in range(5):
            center = (width//4 + i * width//8, height//3)
            cv2.circle(image, center, 20, 0.5, 2)
        
        # 添加细线结构
        for i in range(10):
            y = height//4 + i * 20
            cv2.line(image, (width//8, y), (7*width//8, y), 0.6, 1)
        
        # 添加纹理区域
        texture_region = image[height//2:3*height//4, width//4:3*width//4]
        noise_texture = np.random.random(texture_region.shape) * 0.2
        image[height//2:3*height//4, width//4:3*width//4] = texture_region + noise_texture
        
        # 添加噪声
        noise = np.random.normal(0, 0.05, (height, width))
        image = image + noise
        
        # 归一化
        image = np.clip(image, 0, 1)
        
        self.test_image = image
        return image
    
    def load_test_image(self, image_path):
        """从文件加载测试图像"""
        try:
            # 使用鲁棒加载方法
            with open(image_path, 'rb') as f:
                file_data = f.read()
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                self.test_image = image.astype(np.float64) / 255.0
                return True
            else:
                return False
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def test_algorithm(self, algorithm_name, enhancement_func, **kwargs):
        """测试单个算法"""
        if self.test_image is None:
            raise ValueError("请先创建或加载测试图像")
        
        print(f"测试算法: {algorithm_name}")
        
        start_time = time.time()
        try:
            enhanced_image = enhancement_func(self.test_image, **kwargs)
            processing_time = time.time() - start_time
            
            # 计算指标
            metrics = self.calculate_detailed_metrics(self.test_image, enhanced_image)
            metrics['processing_time'] = processing_time
            metrics['success'] = True
            
            self.results[algorithm_name] = {
                'enhanced_image': enhanced_image,
                'metrics': metrics
            }
            
            print(f"  完成，用时: {processing_time:.2f}秒")
            return enhanced_image, metrics
            
        except Exception as e:
            print(f"  失败: {e}")
            self.results[algorithm_name] = {
                'enhanced_image': None,
                'metrics': {'success': False, 'error': str(e)}
            }
            return None, None
    
    def calculate_detailed_metrics(self, original, enhanced):
        """计算详细的质量指标"""
        metrics = {}
        
        # 1. 均方误差
        mse = np.mean((original - enhanced) ** 2)
        metrics['MSE'] = mse
        
        # 2. 峰值信噪比
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['PSNR'] = psnr
        
        # 3. 结构相似性指标 (简化版SSIM)
        metrics['SSIM'] = self._calculate_ssim(original, enhanced)
        
        # 4. 高频内容保持
        def high_frequency_content(img):
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            return np.mean(np.abs(laplacian))
        
        original_hfc = high_frequency_content(original)
        enhanced_hfc = high_frequency_content(enhanced)
        metrics['High_Freq_Preservation'] = enhanced_hfc / (original_hfc + 1e-10)
        
        # 5. 边缘保持
        def edge_strength(img):
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            return np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        original_edge = edge_strength(original)
        enhanced_edge = edge_strength(enhanced)
        metrics['Edge_Preservation'] = enhanced_edge / (original_edge + 1e-10)
        
        # 6. 对比度改善
        original_contrast = np.std(original)
        enhanced_contrast = np.std(enhanced)
        metrics['Contrast_Enhancement'] = enhanced_contrast / (original_contrast + 1e-10)
        
        # 7. 细节锐度
        def detail_sharpness(img):
            # 使用Tenengrad算子
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.mean(grad_x**2 + grad_y**2)
            return tenengrad
        
        original_sharpness = detail_sharpness(original)
        enhanced_sharpness = detail_sharpness(enhanced)
        metrics['Sharpness_Ratio'] = enhanced_sharpness / (original_sharpness + 1e-10)
        
        # 8. 噪声抑制
        def estimate_noise(img):
            # 使用Laplacian估计噪声
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            return np.sqrt(np.pi/2) * np.mean(np.abs(laplacian))
        
        original_noise = estimate_noise(original)
        enhanced_noise = estimate_noise(enhanced)
        metrics['Noise_Reduction'] = original_noise / (enhanced_noise + 1e-10)
        
        return metrics
    
    def _calculate_ssim(self, img1, img2, window_size=11, k1=0.01, k2=0.03):
        """计算结构相似性指标"""
        from scipy.ndimage import gaussian_filter
        
        mu1 = gaussian_filter(img1, sigma=1.5)
        mu2 = gaussian_filter(img2, sigma=1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
        
        c1 = (k1) ** 2
        c2 = (k2) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return np.mean(ssim_map)
    
    def run_comprehensive_comparison(self, test_image_path=None):
        """运行全面的算法对比"""
        print("=" * 80)
        print("X光图像增强算法对比测试")
        print("=" * 80)
        
        # 准备测试图像
        if test_image_path:
            if not self.load_test_image(test_image_path):
                print("使用合成测试图像...")
                self.create_test_image()
        else:
            print("创建合成测试图像...")
            self.create_test_image()
        
        print(f"测试图像尺寸: {self.test_image.shape}")
        
        # 保存测试图像
        cv2.imwrite('test_image_original.png', (self.test_image * 255).astype(np.uint8))
        print("测试图像已保存: test_image_original.png")
        
        # 测试高精度算法
        print("\n" + "-" * 50)
        print("测试高精度算法")
        print("-" * 50)
        
        advanced_enhancer = AdvancedXRayEnhancer()
        advanced_enhancer.original_image = self.test_image
        
        # 小波去噪
        self.test_algorithm(
            "小波去噪",
            lambda img: advanced_enhancer.wavelet_denoising(img)
        )
        
        # BM3D去噪
        self.test_algorithm(
            "BM3D去噪",
            lambda img: advanced_enhancer.bm3d_denoising(img)
        )
        
        # 自适应Wiener滤波
        self.test_algorithm(
            "自适应Wiener滤波",
            lambda img: advanced_enhancer.adaptive_wiener_filter(img)
        )
        
        # 多尺度细节增强
        self.test_algorithm(
            "多尺度细节增强",
            lambda img: advanced_enhancer.multiscale_detail_enhancement(img)
        )
        
        # 冲击滤波
        self.test_algorithm(
            "冲击滤波",
            lambda img: advanced_enhancer.shock_filter(img)
        )
        
        # 相干增强扩散
        self.test_algorithm(
            "相干增强扩散",
            lambda img: advanced_enhancer.coherence_enhancing_diffusion(img)
        )
        
        # 高精度完整管道
        self.test_algorithm(
            "高精度完整管道",
            lambda img: advanced_enhancer.precision_enhancement_pipeline()
        )
        
        # 测试原始算法作为对比
        print("\n" + "-" * 50)
        print("测试原始算法（对比基准）")
        print("-" * 50)
        
        try:
            original_enhancer = XRayEnhancer()
            original_enhancer.original_image = self.test_image
            
            self.test_algorithm(
                "原始综合增强",
                lambda img: original_enhancer.comprehensive_enhancement()
            )
        except Exception as e:
            print(f"原始算法测试失败: {e}")
        
        # 生成报告
        self.generate_comparison_report()
        
        # 保存结果图像
        self.save_comparison_images()
        
        # 创建可视化对比
        self.create_visual_comparison()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("算法对比报告")
        print("=" * 80)
        
        # 表头
        print(f"{'算法名称':<20} {'PSNR':<8} {'SSIM':<8} {'细节保持':<10} {'边缘保持':<10} {'时间(s)':<8}")
        print("-" * 80)
        
        # 排序算法（按SSIM排序）
        sorted_results = []
        for name, result in self.results.items():
            if result['metrics'].get('success', False):
                sorted_results.append((name, result))
        
        sorted_results.sort(key=lambda x: x[1]['metrics'].get('SSIM', 0), reverse=True)
        
        # 显示结果
        for name, result in sorted_results:
            metrics = result['metrics']
            print(f"{name:<20} "
                  f"{metrics.get('PSNR', 0):<8.2f} "
                  f"{metrics.get('SSIM', 0):<8.4f} "
                  f"{metrics.get('High_Freq_Preservation', 0):<10.4f} "
                  f"{metrics.get('Edge_Preservation', 0):<10.4f} "
                  f"{metrics.get('processing_time', 0):<8.2f}")
        
        print("\n" + "=" * 80)
        print("指标说明:")
        print("PSNR: 峰值信噪比，越高越好")
        print("SSIM: 结构相似性，越接近1越好")
        print("细节保持: 高频内容保持率，越接近1越好")
        print("边缘保持: 边缘强度保持率，越接近1越好")
        print("=" * 80)
    
    def save_comparison_images(self):
        """保存对比图像"""
        print("\n保存对比图像...")
        
        for name, result in self.results.items():
            if result['enhanced_image'] is not None:
                filename = f"comparison_{name.replace(' ', '_').replace('/', '_')}.png"
                cv2.imwrite(filename, (result['enhanced_image'] * 255).astype(np.uint8))
                print(f"已保存: {filename}")
    
    def create_visual_comparison(self):
        """创建可视化对比图"""
        print("\n创建可视化对比图...")
        
        # 选择最好的几个算法
        valid_results = [(name, result) for name, result in self.results.items() 
                        if result['enhanced_image'] is not None]
        
        if len(valid_results) == 0:
            print("没有有效的结果可供可视化")
            return
        
        # 按SSIM排序，选择前6个
        valid_results.sort(key=lambda x: x[1]['metrics'].get('SSIM', 0), reverse=True)
        top_results = valid_results[:6]
        
        # 创建对比图
        n_cols = 3
        n_rows = (len(top_results) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (name, result) in enumerate(top_results):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(result['enhanced_image'], cmap='gray')
            plt.title(f"{name}\nSSIM: {result['metrics'].get('SSIM', 0):.4f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("可视化对比图已保存: algorithm_comparison.png")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='X光图像增强算法对比测试')
    parser.add_argument('--input', help='输入测试图像路径（可选，不提供则使用合成图像）')
    parser.add_argument('--output-dir', default='.', help='输出目录（默认当前目录）')
    
    args = parser.parse_args()
    
    # 切换到输出目录
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # 创建对比工具
    comparison = AlgorithmComparison()
    
    # 运行对比测试
    comparison.run_comprehensive_comparison(args.input)
    
    print("\n对比测试完成！")
    print("结果文件:")
    print("- test_image_original.png: 原始测试图像")
    print("- comparison_*.png: 各算法结果图像")
    print("- algorithm_comparison.png: 可视化对比图")


if __name__ == "__main__":
    main()