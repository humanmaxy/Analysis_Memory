#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分析工具使用示例
演示如何在没有GUI的情况下使用核心功能
"""

import os
import re

def extract_filename_pattern(filename):
    """提取文件名模式"""
    patterns = {
        'barcode_timestamp': r'^([A-Za-z0-9]+)_(\d{8,14})$',
        'date_sequence': r'^(\d{4}-?\d{2}-?\d{2})_?(\d+)$',
        'number_prefix': r'^(\d+)_([A-Za-z0-9]+)$',
        'multi_underscore': r'^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$',
        'prefix_number': r'^([A-Za-z]+)(\d+)$',
    }
    
    for pattern_name, pattern in patterns.items():
        match = re.match(pattern, filename)
        if match:
            return {'name': pattern_name, 'regex': pattern, 'template': filename, 'groups': match.groups()}
    
    smart_pattern = create_smart_pattern(filename)
    return {'name': 'smart_pattern', 'regex': smart_pattern, 'template': filename}

def create_smart_pattern(filename):
    """创建智能模式匹配"""
    pattern = ""
    i = 0
    while i < len(filename):
        char = filename[i]
        if char.isdigit():
            pattern += r'\d+'
            while i < len(filename) and filename[i].isdigit():
                i += 1
        elif char.isalpha():
            pattern += r'[A-Za-z]+'
            while i < len(filename) and filename[i].isalpha():
                i += 1
        elif char.isalnum():
            pattern += r'[A-Za-z0-9]+'
            while i < len(filename) and filename[i].isalnum():
                i += 1
        else:
            pattern += re.escape(char)
            i += 1
    
    return f"^{pattern}$"

def demo_filename_analysis():
    """演示文件名分析功能"""
    print("=" * 60)
    print("图像文件命名规则分析演示")
    print("=" * 60)
    
    # 示例文件名
    sample_files = [
        "IMG_20231201_120000.jpg",
        "Photo001.png",
        "2023-12-01_001.jpg",
        "001_Sample.png",
        "ABC123_20231201120000.jpg",
        "CustomName123.jpg"
    ]
    
    print("\n1. 样本文件名模式识别：")
    print("-" * 40)
    
    for filename_with_ext in sample_files:
        filename = os.path.splitext(filename_with_ext)[0]  # 去除扩展名
        pattern = extract_filename_pattern(filename)
        print(f"文件: {filename_with_ext}")
        print(f"  模式类型: {pattern['name']}")
        print(f"  正则表达式: {pattern['regex']}")
        if 'groups' in pattern:
            print(f"  匹配组: {pattern['groups']}")
        print()
    
    print("\n2. 命名规则一致性检查演示：")
    print("-" * 40)
    
    # 以第一个文件为样本
    sample_filename = os.path.splitext(sample_files[0])[0]
    sample_pattern = extract_filename_pattern(sample_filename)
    print(f"样本文件: {sample_files[0]}")
    print(f"样本模式: {sample_pattern['name']} ({sample_pattern['regex']})")
    print()
    
    # 检查其他文件是否符合样本模式
    test_files = [
        "IMG_20231202_130000.jpg",
        "IMG_20231203_140000.jpg", 
        "Photo001.png",  # 不匹配
        "IMG_Test_001.jpg",
        "IMG_20231204_150000.jpg"
    ]
    
    print("检查结果:")
    for test_file in test_files:
        test_filename = os.path.splitext(test_file)[0]
        matches = bool(re.match(sample_pattern['regex'], test_filename))
        status = "✓ 匹配" if matches else "✗ 不匹配"
        print(f"  {test_file}: {status}")
    
    print("\n3. 智能模式生成演示：")
    print("-" * 40)
    
    custom_files = [
        "Report-2023-001.pdf",
        "Data_Set_A_v2.xlsx", 
        "IMG-20231201-HDR.jpg",
        "Backup_20231201_Final.zip"
    ]
    
    for filename_with_ext in custom_files:
        filename = os.path.splitext(filename_with_ext)[0]
        smart_pattern = create_smart_pattern(filename)
        print(f"文件: {filename_with_ext}")
        print(f"  智能模式: {smart_pattern}")
        
        # 验证模式是否能匹配原文件名
        matches = bool(re.match(smart_pattern, filename))
        print(f"  验证结果: {'✓ 正确' if matches else '✗ 错误'}")
        print()

def demo_batch_analysis():
    """演示批量分析功能"""
    print("\n" + "=" * 60)
    print("批量文件分析演示")
    print("=" * 60)
    
    # 模拟一个文件夹中的文件列表
    folder_files = [
        "IMG_20231201_120000.jpg",
        "IMG_20231201_120001.jpg", 
        "IMG_20231201_120002.jpg",
        "Photo001.png",  # 命名不一致
        "IMG_20231201_120003.jpg",
        "Document.pdf",  # 非图像文件
        "IMG_Test_001.jpg",  # 命名不一致
        "IMG_20231201_120004.jpg"
    ]
    
    # 图像文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 筛选图像文件
    image_files = [f for f in folder_files 
                  if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"文件夹总文件数: {len(folder_files)}")
    print(f"图像文件数: {len(image_files)}")
    print()
    
    if not image_files:
        print("没有找到图像文件")
        return
    
    # 以第一个图像文件为样本
    sample_file = image_files[0]
    sample_filename = os.path.splitext(sample_file)[0]
    sample_pattern = extract_filename_pattern(sample_filename)
    
    print(f"样本文件: {sample_file}")
    print(f"样本模式: {sample_pattern['name']}")
    print()
    
    # 分析所有文件
    print("分析结果:")
    print("-" * 50)
    print(f"{'文件名':<25} {'命名规则':<10} {'状态'}")
    print("-" * 50)
    
    naming_issues = []
    
    for image_file in image_files:
        filename = os.path.splitext(image_file)[0]
        matches = bool(re.match(sample_pattern['regex'], filename))
        status = "✓" if matches else "✗"
        
        if not matches:
            naming_issues.append(image_file)
        
        print(f"{image_file:<25} {sample_pattern['name']:<10} {status}")
    
    print("-" * 50)
    print(f"总计: {len(image_files)} 个文件")
    print(f"命名规则一致: {len(image_files) - len(naming_issues)} 个")
    print(f"命名规则不一致: {len(naming_issues)} 个")
    
    if naming_issues:
        print(f"\n命名不一致的文件:")
        for issue_file in naming_issues:
            print(f"  • {issue_file}")

def main():
    """主函数"""
    print("图像文件分析工具 - 使用演示")
    print("本演示展示了核心功能的使用方法")
    print("完整的GUI版本请运行: python image_analyzer.py")
    
    demo_filename_analysis()
    demo_batch_analysis()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n要使用完整功能（包括图像属性和色彩分布分析），")
    print("请安装依赖包后运行GUI版本:")
    print("  pip install opencv-python Pillow numpy matplotlib")
    print("  python image_analyzer.py")

if __name__ == "__main__":
    main()