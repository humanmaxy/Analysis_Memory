#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像增强算法测试脚本
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from xray_enhancement import XRayEnhancer
import os


def create_test_xray_image():
    """创建测试用的X光图像"""
    # 创建一个模拟的X光图像
    height, width = 512, 512
    
    # 基础背景
    image = np.zeros((height, width), dtype=np.float64)
    
    # 添加一些结构（模拟骨骼）
    # 椭圆形结构
    cv2.ellipse(image, (width//2, height//2), (100, 150), 0, 0, 360, 0.8, -1)
    cv2.ellipse(image, (width//2-50, height//2-100), (30, 80), 0, 0, 360, 0.6, -1)
    cv2.ellipse(image, (width//2+50, height//2-100), (30, 80), 0, 0, 360, 0.6, -1)
    
    # 添加一些细节线条
    cv2.line(image, (width//2-80, height//2-50), (width//2+80, height//2+50), 0.7, 3)
    cv2.line(image, (width//2-80, height//2+50), (width//2+80, height//2-50), 0.7, 3)
    
    # 添加噪声
    noise = np.random.normal(0, 0.1, (height, width))
    image = image + noise
    
    # 添加一些高频噪声
    high_freq_noise = np.random.normal(0, 0.05, (height, width))
    image = image + high_freq_noise
    
    # 模拟低对比度
    image = image * 0.6 + 0.2
    
    # 确保在有效范围内
    image = np.clip(image, 0, 1)
    
    return image


def test_individual_methods():
    """测试各个增强方法"""
    print("=" * 60)
    print("测试各个增强方法")
    print("=" * 60)
    
    # 创建测试图像
    test_image = create_test_xray_image()
    
    # 保存测试图像
    cv2.imwrite('/workspace/test_xray_original.png', (test_image * 255).astype(np.uint8))
    print("测试图像已保存: test_xray_original.png")
    
    enhancer = XRayEnhancer()
    enhancer.original_image = test_image
    
    # 测试去噪方法
    print("\n测试去噪方法:")
    noise_methods = ['multi_scale', 'nl_means', 'tv_chambolle', 'bilateral']
    
    for method in noise_methods:
        print(f"  测试 {method} 去噪...")
        try:
            result = enhancer.adaptive_noise_reduction(test_image, method)
            # 计算PSNR
            mse = np.mean((test_image - result) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            print(f"    PSNR: {psnr:.2f} dB")
            
            # 保存结果
            cv2.imwrite(f'/workspace/test_denoising_{method}.png', (result * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"    错误: {e}")
    
    # 测试直方图均衡方法
    print("\n测试直方图均衡方法:")
    hist_methods = ['clahe', 'adaptive_eq', 'gamma_correction']
    
    for method in hist_methods:
        print(f"  测试 {method} 直方图均衡...")
        try:
            result = enhancer.adaptive_histogram_equalization(test_image, method)
            
            # 计算对比度改善
            orig_contrast = np.std(test_image)
            new_contrast = np.std(result)
            improvement = new_contrast / orig_contrast
            print(f"    对比度改善: {improvement:.2f}x")
            
            # 保存结果
            cv2.imwrite(f'/workspace/test_histogram_{method}.png', (result * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"    错误: {e}")
    
    # 测试边缘保护方法
    print("\n测试边缘保护方法:")
    edge_methods = ['anisotropic', 'guided_filter', 'domain_transform']
    
    for method in edge_methods:
        print(f"  测试 {method} 边缘保护...")
        try:
            result = enhancer.edge_preserving_smoothing(test_image, method)
            
            # 计算边缘保持
            metrics = enhancer.calculate_quality_metrics(test_image, result)
            edge_preservation = metrics.get('Edge_Preservation', 0)
            print(f"    边缘保持: {edge_preservation:.4f}")
            
            # 保存结果
            cv2.imwrite(f'/workspace/test_edge_{method}.png', (result * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"    错误: {e}")


def test_comprehensive_enhancement():
    """测试综合增强"""
    print("\n" + "=" * 60)
    print("测试综合增强")
    print("=" * 60)
    
    # 创建测试图像
    test_image = create_test_xray_image()
    
    enhancer = XRayEnhancer()
    enhancer.original_image = test_image
    
    # 测试不同的参数组合
    test_configs = [
        {
            'name': '基础配置',
            'params': {
                'noise_method': 'multi_scale',
                'hist_method': 'clahe',
                'edge_method': 'anisotropic',
                'use_unsharp': True,
                'use_retinex': False,
                'use_morphology': False,
                'use_frequency': False
            }
        },
        {
            'name': '高质量配置',
            'params': {
                'noise_method': 'nl_means',
                'hist_method': 'clahe',
                'edge_method': 'anisotropic',
                'use_unsharp': True,
                'use_retinex': True,
                'use_morphology': True,
                'use_frequency': False
            }
        },
        {
            'name': '最大增强配置',
            'params': {
                'noise_method': 'multi_scale',
                'hist_method': 'adaptive_eq',
                'edge_method': 'guided_filter',
                'use_unsharp': True,
                'use_retinex': True,
                'use_morphology': True,
                'use_frequency': True
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n测试 {config['name']}:")
        try:
            import time
            start_time = time.time()
            
            result = enhancer.comprehensive_enhancement(**config['params'])
            
            processing_time = time.time() - start_time
            print(f"  处理时间: {processing_time:.2f}秒")
            
            # 计算质量指标
            metrics = enhancer.calculate_quality_metrics(test_image, result)
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  SSIM: {metrics['SSIM']:.4f}")
            print(f"  对比度改善: {metrics['Contrast_Improvement']:.2f}x")
            print(f"  边缘保持: {metrics['Edge_Preservation']:.4f}")
            
            # 保存结果
            filename = f"test_comprehensive_{config['name'].replace(' ', '_')}.png"
            cv2.imwrite(f'/workspace/{filename}', (result * 255).astype(np.uint8))
            print(f"  结果已保存: {filename}")
            
        except Exception as e:
            print(f"  错误: {e}")


def create_comparison_plot():
    """创建对比图"""
    print("\n" + "=" * 60)
    print("创建对比图")
    print("=" * 60)
    
    # 检查文件是否存在
    files_to_compare = [
        ('test_xray_original.png', '原始图像'),
        ('test_comprehensive_基础配置.png', '基础增强'),
        ('test_comprehensive_高质量配置.png', '高质量增强'),
        ('test_comprehensive_最大增强配置.png', '最大增强')
    ]
    
    existing_files = [(f, title) for f, title in files_to_compare if os.path.exists(f'/workspace/{f}')]
    
    if len(existing_files) < 2:
        print("没有足够的图像文件进行对比")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('X光图像增强效果对比', fontsize=16)
        
        for i, (filename, title) in enumerate(existing_files[:4]):
            row, col = i // 2, i % 2
            
            image = cv2.imread(f'/workspace/{filename}', cv2.IMREAD_GRAYSCALE)
            if image is not None:
                axes[row, col].imshow(image, cmap='gray')
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(existing_files), 4):
            row, col = i // 2, i % 2
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('/workspace/xray_enhancement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("对比图已保存: xray_enhancement_comparison.png")
        
    except Exception as e:
        print(f"创建对比图失败: {e}")


def performance_test():
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    # 测试不同尺寸的图像
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    enhancer = XRayEnhancer()
    
    for width, height in sizes:
        print(f"\n测试图像尺寸: {width}x{height}")
        
        # 创建测试图像
        test_image = np.random.rand(height, width).astype(np.float64)
        # 添加一些结构
        cv2.circle(test_image, (width//2, height//2), min(width, height)//4, 0.8, -1)
        # 添加噪声
        test_image += np.random.normal(0, 0.1, (height, width))
        test_image = np.clip(test_image, 0, 1)
        
        enhancer.original_image = test_image
        
        # 测试基础配置的处理时间
        import time
        start_time = time.time()
        
        try:
            result = enhancer.comprehensive_enhancement(
                noise_method='multi_scale',
                hist_method='clahe',
                edge_method='anisotropic',
                use_unsharp=True,
                use_retinex=False,
                use_morphology=False,
                use_frequency=False
            )
            
            processing_time = time.time() - start_time
            pixels = width * height
            
            print(f"  处理时间: {processing_time:.2f}秒")
            print(f"  处理速度: {pixels/processing_time:.0f} 像素/秒")
            print(f"  内存使用: {test_image.nbytes / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            print(f"  错误: {e}")


def main():
    """主函数"""
    print("X光图像增强算法测试")
    print("=" * 60)
    
    try:
        # 测试各个方法
        test_individual_methods()
        
        # 测试综合增强
        test_comprehensive_enhancement()
        
        # 创建对比图
        create_comparison_plot()
        
        # 性能测试
        performance_test()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("测试结果图像已保存到当前目录")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()