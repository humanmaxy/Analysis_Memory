#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像增强命令行工具
专门针对黑色X光图像的高质量增强处理
"""

import argparse
import cv2
import numpy as np
import os
import time
from xray_enhancement import XRayEnhancer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='X光图像增强工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('output', help='输出图像路径')
    parser.add_argument('--noise-method', default='multi_scale', 
                       choices=['multi_scale', 'nl_means', 'tv_chambolle', 'bilateral', 'none'],
                       help='去噪方法 (默认: multi_scale)')
    parser.add_argument('--hist-method', default='clahe',
                       choices=['clahe', 'adaptive_eq', 'gamma_correction', 'none'],
                       help='直方图均衡方法 (默认: clahe)')
    parser.add_argument('--edge-method', default='anisotropic',
                       choices=['anisotropic', 'guided_filter', 'domain_transform', 'none'],
                       help='边缘保护方法 (默认: anisotropic)')
    parser.add_argument('--no-unsharp', action='store_true', help='禁用非锐化掩模')
    parser.add_argument('--no-retinex', action='store_true', help='禁用Retinex增强')
    parser.add_argument('--no-morphology', action='store_true', help='禁用形态学增强')
    parser.add_argument('--use-frequency', action='store_true', help='启用频域增强')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 创建增强器
    enhancer = XRayEnhancer()
    
    # 加载图像
    if args.verbose:
        print(f"加载图像: {args.input}")
    
    if not enhancer.load_image(args.input):
        print("错误: 无法加载图像")
        return 1
    
    if args.verbose:
        print(f"图像尺寸: {enhancer.original_image.shape}")
    
    # 设置参数
    params = {
        'noise_method': args.noise_method,
        'hist_method': args.hist_method,
        'edge_method': args.edge_method,
        'use_unsharp': not args.no_unsharp,
        'use_retinex': not args.no_retinex,
        'use_morphology': not args.no_morphology,
        'use_frequency': args.use_frequency
    }
    
    if args.verbose:
        print("增强参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # 执行增强
    print("开始图像增强处理...")
    start_time = time.time()
    
    try:
        enhanced_image = enhancer.comprehensive_enhancement(**params)
        processing_time = time.time() - start_time
        
        if args.verbose:
            print(f"处理完成，用时: {processing_time:.2f}秒")
        
        # 保存结果
        enhancer.save_enhanced_image(args.output)
        
        # 计算质量指标
        if args.verbose:
            print("计算图像质量指标...")
            metrics = enhancer.calculate_quality_metrics(
                enhancer.original_image, enhanced_image
            )
            
            print("\n图像质量指标:")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  SSIM: {metrics['SSIM']:.4f}")
            print(f"  对比度改善: {metrics['Contrast_Improvement']:.2f}x")
            print(f"  边缘保持: {metrics['Edge_Preservation']:.4f}")
        
        print(f"增强完成! 结果已保存到: {args.output}")
        return 0
        
    except Exception as e:
        print(f"错误: 增强处理失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())