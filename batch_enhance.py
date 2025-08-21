#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像批量增强工具
"""

import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from xray_enhancement import XRayEnhancer
import threading

class BatchEnhancer:
    """批量增强器"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(4, os.cpu_count())
        self.lock = threading.Lock()
        self.processed = 0
        self.total = 0
        
    def enhance_single_image(self, input_path, output_path, params):
        """增强单张图像"""
        try:
            enhancer = XRayEnhancer()
            
            if not enhancer.load_image(input_path):
                return False, f"无法加载图像: {input_path}"
            
            enhanced_image = enhancer.comprehensive_enhancement(**params)
            enhancer.save_enhanced_image(output_path)
            
            with self.lock:
                self.processed += 1
                print(f"进度: {self.processed}/{self.total} - 完成: {os.path.basename(input_path)}")
            
            return True, None
            
        except Exception as e:
            return False, f"处理失败 {input_path}: {str(e)}"
    
    def batch_enhance(self, input_dir, output_dir, params, extensions=None):
        """批量增强"""
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm'}
        
        # 获取所有图像文件
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in extensions:
                    input_path = os.path.join(root, file)
                    # 保持目录结构
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, rel_path)
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    image_files.append((input_path, output_path))
        
        if not image_files:
            print(f"在目录 {input_dir} 中没有找到图像文件")
            return
        
        self.total = len(image_files)
        print(f"找到 {self.total} 个图像文件")
        print(f"使用 {self.max_workers} 个线程进行处理")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 批量处理
        start_time = time.time()
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.enhance_single_image, input_path, output_path, params): 
                (input_path, output_path) 
                for input_path, output_path in image_files
            }
            
            # 收集结果
            for future in as_completed(futures):
                input_path, output_path = futures[future]
                success, error = future.result()
                
                if not success:
                    failed_files.append((input_path, error))
                    print(f"失败: {error}")
        
        total_time = time.time() - start_time
        
        # 打印统计信息
        print(f"\n批处理完成!")
        print(f"总时间: {total_time:.2f}秒")
        print(f"成功处理: {self.processed} 个文件")
        print(f"失败: {len(failed_files)} 个文件")
        print(f"平均处理时间: {total_time/self.total:.2f}秒/文件")
        
        if failed_files:
            print(f"\n失败的文件:")
            for file_path, error in failed_files:
                print(f"  {file_path}: {error}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='X光图像批量增强工具')
    parser.add_argument('input_dir', help='输入目录')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--workers', type=int, default=None, help='并行线程数')
    parser.add_argument('--noise-method', default='multi_scale', 
                       choices=['multi_scale', 'nl_means', 'tv_chambolle', 'bilateral', 'none'],
                       help='去噪方法')
    parser.add_argument('--hist-method', default='clahe',
                       choices=['clahe', 'adaptive_eq', 'gamma_correction', 'none'],
                       help='直方图均衡方法')
    parser.add_argument('--edge-method', default='anisotropic',
                       choices=['anisotropic', 'guided_filter', 'domain_transform', 'none'],
                       help='边缘保护方法')
    parser.add_argument('--no-unsharp', action='store_true', help='禁用非锐化掩模')
    parser.add_argument('--no-retinex', action='store_true', help='禁用Retinex增强')
    parser.add_argument('--no-morphology', action='store_true', help='禁用形态学增强')
    parser.add_argument('--use-frequency', action='store_true', help='启用频域增强')
    parser.add_argument('--extensions', nargs='+', default=None,
                       help='图像文件扩展名 (默认: jpg jpeg png bmp tiff tif dcm)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"错误: {args.input_dir} 不是目录")
        return 1
    
    # 处理扩展名
    extensions = None
    if args.extensions:
        extensions = {f'.{ext.lstrip(".")}' for ext in args.extensions}
    
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
    
    print("批量增强参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # 创建批处理器并执行
    batch_enhancer = BatchEnhancer(max_workers=args.workers)
    batch_enhancer.batch_enhance(args.input_dir, args.output_dir, params, extensions)
    
    return 0


if __name__ == "__main__":
    exit(main())