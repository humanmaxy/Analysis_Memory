#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像增强使用示例
演示如何使用不同的增强配置
"""

import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

try:
    from xray_enhancement import XRayEnhancer
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("警告: 缺少依赖包，将显示使用示例而不执行实际处理")


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)
    
    if not HAS_DEPENDENCIES:
        print("""
# 基础使用代码示例:
from xray_enhancement import XRayEnhancer

# 创建增强器
enhancer = XRayEnhancer()

# 加载图像
if enhancer.load_image('input_xray.jpg'):
    print("图像加载成功!")
    
    # 使用默认参数进行增强
    enhanced_image = enhancer.comprehensive_enhancement()
    
    # 保存结果
    enhancer.save_enhanced_image('output_enhanced.jpg')
    
    # 计算质量指标
    metrics = enhancer.calculate_quality_metrics(
        enhancer.original_image, 
        enhanced_image
    )
    
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"对比度改善: {metrics['Contrast_Improvement']:.2f}x")
else:
    print("无法加载图像文件")
        """)
        return
    
    # 实际执行代码（如果有依赖）
    enhancer = XRayEnhancer()
    
    # 这里可以加载实际图像文件
    print("要运行此示例，请提供X光图像文件路径")
    print("示例: enhancer.load_image('your_xray_image.jpg')")


def example_custom_parameters():
    """自定义参数示例"""
    print("\n" + "=" * 60)
    print("自定义参数示例")
    print("=" * 60)
    
    print("""
# 高质量增强配置
enhancer = XRayEnhancer()
enhancer.load_image('input.jpg')

enhanced = enhancer.comprehensive_enhancement(
    noise_method='nl_means',        # 非局部均值去噪（高质量）
    hist_method='adaptive_eq',      # 自适应直方图均衡
    edge_method='anisotropic',      # 各向异性扩散
    use_unsharp=True,               # 启用非锐化掩模
    use_retinex=True,               # 启用Retinex增强
    use_morphology=True,            # 启用形态学增强
    use_frequency=True              # 启用频域增强（耗时）
)

# 快速处理配置
fast_enhanced = enhancer.comprehensive_enhancement(
    noise_method='bilateral',       # 双边滤波（快速）
    hist_method='clahe',           # CLAHE
    edge_method='guided_filter',   # 引导滤波（快速）
    use_unsharp=True,              
    use_retinex=False,             # 关闭耗时的增强
    use_morphology=False,
    use_frequency=False
)
    """)


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("批量处理示例")
    print("=" * 60)
    
    print("""
# 方法1: 使用批量处理工具
import subprocess

# 批量处理整个文件夹
subprocess.run([
    'python', 'batch_enhance.py',
    'input_folder',      # 输入文件夹
    'output_folder',     # 输出文件夹
    '--workers', '4',    # 使用4个线程
    '--noise-method', 'multi_scale',
    '--hist-method', 'clahe',
    '--verbose'          # 显示详细信息
])

# 方法2: 编程方式批量处理
import os
from xray_enhancement import XRayEnhancer

def batch_enhance_folder(input_folder, output_folder):
    enhancer = XRayEnhancer()
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for filename in os.listdir(input_folder):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"enhanced_{filename}")
            
            print(f"处理: {filename}")
            
            if enhancer.load_image(input_path):
                enhanced = enhancer.comprehensive_enhancement()
                enhancer.save_enhanced_image(output_path)
                print(f"完成: {filename}")
            else:
                print(f"跳过: {filename} (无法加载)")

# 使用示例
batch_enhance_folder('xray_images', 'enhanced_images')
    """)


def example_quality_assessment():
    """质量评估示例"""
    print("\n" + "=" * 60)
    print("质量评估示例")
    print("=" * 60)
    
    print("""
# 详细的质量评估
enhancer = XRayEnhancer()
enhancer.load_image('input.jpg')

# 测试不同的去噪方法
methods = ['multi_scale', 'nl_means', 'tv_chambolle', 'bilateral']
results = {}

for method in methods:
    enhanced = enhancer.comprehensive_enhancement(
        noise_method=method,
        hist_method='clahe',
        edge_method='anisotropic'
    )
    
    metrics = enhancer.calculate_quality_metrics(
        enhancer.original_image, 
        enhanced
    )
    
    results[method] = metrics
    
    print(f"{method}:")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  对比度改善: {metrics['Contrast_Improvement']:.2f}x")
    print(f"  边缘保持: {metrics['Edge_Preservation']:.4f}")
    print()

# 找出最佳方法
best_method = max(results.keys(), key=lambda k: results[k]['SSIM'])
print(f"最佳方法: {best_method}")
    """)


def example_medical_specific():
    """医学图像特定示例"""
    print("\n" + "=" * 60)
    print("医学图像特定配置示例")
    print("=" * 60)
    
    print("""
# 骨科X光图像增强
def enhance_bone_xray(input_path, output_path):
    enhancer = XRayEnhancer()
    enhancer.load_image(input_path)
    
    enhanced = enhancer.comprehensive_enhancement(
        noise_method='multi_scale',    # 平衡去噪和细节
        hist_method='clahe',          # 增强骨骼对比度
        edge_method='anisotropic',    # 保护骨骼边缘
        use_unsharp=True,             # 锐化骨骼轮廓
        use_retinex=True,             # 改善动态范围
        use_morphology=True,          # 增强骨骼结构
        use_frequency=False           # 避免引入伪影
    )
    
    enhancer.save_enhanced_image(output_path)
    return enhanced

# 胸部X光图像增强
def enhance_chest_xray(input_path, output_path):
    enhancer = XRayEnhancer()
    enhancer.load_image(input_path)
    
    enhanced = enhancer.comprehensive_enhancement(
        noise_method='nl_means',      # 高质量去噪
        hist_method='adaptive_eq',    # 自适应增强
        edge_method='guided_filter',  # 保护肺部纹理
        use_unsharp=True,
        use_retinex=False,            # 避免过度增强
        use_morphology=False,         # 保持自然外观
        use_frequency=False
    )
    
    enhancer.save_enhanced_image(output_path)
    return enhanced

# 牙科X光图像增强
def enhance_dental_xray(input_path, output_path):
    enhancer = XRayEnhancer()
    enhancer.load_image(input_path)
    
    enhanced = enhancer.comprehensive_enhancement(
        noise_method='tv_chambolle',  # 保持牙齿边缘
        hist_method='clahe',          # 增强牙齿对比度
        edge_method='anisotropic',    # 精确边缘保护
        use_unsharp=True,             # 锐化牙齿细节
        use_retinex=True,             # 改善局部对比度
        use_morphology=True,          # 增强牙齿结构
        use_frequency=True            # 增强微细结构
    )
    
    enhancer.save_enhanced_image(output_path)
    return enhanced

# 使用示例
enhance_bone_xray('bone_xray.jpg', 'enhanced_bone.jpg')
enhance_chest_xray('chest_xray.jpg', 'enhanced_chest.jpg')  
enhance_dental_xray('dental_xray.jpg', 'enhanced_dental.jpg')
    """)


def example_gui_automation():
    """GUI自动化示例"""
    print("\n" + "=" * 60)
    print("GUI自动化示例")
    print("=" * 60)
    
    print("""
# 自动化GUI处理（适合批量处理时使用GUI预设）
import tkinter as tk
from xray_enhancement import XRayEnhancerGUI

def automated_gui_processing():
    # 创建隐藏的GUI实例
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    app = XRayEnhancerGUI(root)
    
    # 设置参数
    app.noise_method.set('nl_means')
    app.hist_method.set('clahe')
    app.edge_method.set('anisotropic')
    app.use_unsharp.set(True)
    app.use_retinex.set(True)
    app.use_morphology.set(True)
    
    # 处理图像列表
    image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    
    for image_file in image_files:
        app.file_path.set(image_file)
        app.current_image_path = image_file
        
        if app.enhancer.load_image(image_file):
            app.perform_enhancement()
            
            output_file = f"enhanced_{image_file}"
            app.enhancer.save_enhanced_image(output_file)
            print(f"处理完成: {image_file} -> {output_file}")
    
    root.destroy()

# 运行自动化处理
automated_gui_processing()
    """)


def main():
    """主函数"""
    print("X光图像增强 - 使用示例")
    print("本脚本展示了各种使用场景和配置方法")
    
    if not HAS_DEPENDENCIES:
        print("\n注意: 当前环境缺少必要的依赖包")
        print("要运行实际代码，请安装依赖:")
        print("pip install -r requirements_xray.txt")
    
    example_basic_usage()
    example_custom_parameters()
    example_batch_processing()
    example_quality_assessment()
    example_medical_specific()
    example_gui_automation()
    
    print("\n" + "=" * 60)
    print("示例展示完成!")
    print("=" * 60)
    print("\n快速开始:")
    print("1. 安装依赖: pip install -r requirements_xray.txt")
    print("2. GUI模式: python xray_enhancement.py")
    print("3. 命令行: python xray_cli.py input.jpg output.jpg")
    print("4. 批量处理: python batch_enhance.py input_folder output_folder")
    print("5. 运行测试: python xray_core_test.py")


if __name__ == "__main__":
    main()