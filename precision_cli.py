#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高精度X光图像增强命令行工具
专注于细节保持和精度最大化
"""

import argparse
import numpy as np
import time
from advanced_xray_enhancer import AdvancedXRayEnhancer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高精度X光图像增强工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('output', help='输出图像路径')
    
    # 算法选择
    parser.add_argument('--wavelet', action='store_true', default=True, help='启用小波去噪')
    parser.add_argument('--no-wavelet', dest='wavelet', action='store_false', help='禁用小波去噪')
    parser.add_argument('--bm3d', action='store_true', help='启用BM3D去噪 (高质量，慢)')
    parser.add_argument('--wiener', action='store_true', default=True, help='启用自适应Wiener滤波')
    parser.add_argument('--no-wiener', dest='wiener', action='store_false', help='禁用Wiener滤波')
    parser.add_argument('--multiscale', action='store_true', default=True, help='启用多尺度细节增强')
    parser.add_argument('--no-multiscale', dest='multiscale', action='store_false', help='禁用多尺度增强')
    parser.add_argument('--shock', action='store_true', default=True, help='启用冲击滤波')
    parser.add_argument('--no-shock', dest='shock', action='store_false', help='禁用冲击滤波')
    parser.add_argument('--coherence', action='store_true', help='启用相干增强扩散 (慢)')
    
    # 预设模式
    parser.add_argument('--preset', choices=['fast', 'balanced', 'highest'], 
                       help='预设模式: fast(快速), balanced(平衡), highest(最高质量)')
    
    # 输出选项
    parser.add_argument('--format', choices=['8bit', '16bit'], default='16bit',
                       help='输出位深: 8bit或16bit (默认16bit)')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    parser.add_argument('--metrics', action='store_true', help='计算并显示精度指标')
    
    args = parser.parse_args()
    
    # 应用预设
    if args.preset:
        if args.preset == 'fast':
            args.wavelet = True
            args.bm3d = False
            args.wiener = True
            args.multiscale = False
            args.shock = False
            args.coherence = False
        elif args.preset == 'balanced':
            args.wavelet = True
            args.bm3d = False
            args.wiener = True
            args.multiscale = True
            args.shock = True
            args.coherence = False
        elif args.preset == 'highest':
            args.wavelet = True
            args.bm3d = True
            args.wiener = True
            args.multiscale = True
            args.shock = True
            args.coherence = True
    
    # 创建增强器
    enhancer = AdvancedXRayEnhancer()
    
    # 加载图像
    if args.verbose:
        print(f"加载图像: {args.input}")
    
    if not enhancer.load_image(args.input):
        print("错误: 无法加载图像")
        return 1
    
    if args.verbose:
        print(f"图像尺寸: {enhancer.original_image.shape}")
        print(f"数据类型: {enhancer.original_image.dtype}")
        print("\n算法配置:")
        print(f"  小波去噪: {args.wavelet}")
        print(f"  BM3D去噪: {args.bm3d}")
        print(f"  Wiener滤波: {args.wiener}")
        print(f"  多尺度增强: {args.multiscale}")
        print(f"  冲击滤波: {args.shock}")
        print(f"  相干扩散: {args.coherence}")
        print(f"  输出格式: {args.format}")
    
    # 执行增强
    print("开始高精度图像增强...")
    start_time = time.time()
    
    try:
        enhanced_image = enhancer.precision_enhancement_pipeline(
            use_wavelet=args.wavelet,
            use_bm3d=args.bm3d,
            use_wiener=args.wiener,
            use_multiscale=args.multiscale,
            use_shock=args.shock,
            use_coherence=args.coherence
        )
        
        processing_time = time.time() - start_time
        
        if args.verbose:
            print(f"处理完成，用时: {processing_time:.2f}秒")
        
        # 保存结果
        if args.format == '16bit':
            # 保存为16位TIFF格式
            if not args.output.lower().endswith(('.tiff', '.tif')):
                args.output = args.output.rsplit('.', 1)[0] + '.tiff'
        
        enhancer.save_enhanced_image(args.output)
        
        # 计算精度指标
        if args.metrics or args.verbose:
            print("\n计算精度指标...")
            metrics = enhancer.calculate_precision_metrics(
                enhancer.original_image, enhanced_image
            )
            
            print("\n精度指标:")
            print(f"  高频保持率: {metrics['High_Freq_Preservation']:.4f}")
            print(f"  边缘锐度比: {metrics['Edge_Sharpness_Ratio']:.4f}")
            print(f"  细节保持度: {metrics['Detail_Preservation']:.4f}")
            print(f"  降噪比率: {metrics['Noise_Reduction_Ratio']:.2f}")
        
        print(f"\n高精度增强完成! 结果已保存到: {args.output}")
        print(f"处理时间: {processing_time:.2f}秒")
        
        return 0
        
    except Exception as e:
        print(f"错误: 增强处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())