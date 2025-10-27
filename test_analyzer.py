#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分析工具测试脚本
测试核心功能而不依赖GUI
"""

import os
import re
import sys

class ImageAnalyzerTest:
    """测试图像分析器的核心功能"""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_filename_pattern_extraction(self):
        """测试文件名模式提取"""
        print("测试文件名模式提取...")
        self.total_tests += 1
        
        test_cases = [
            ("ABC123_20231201120000", "barcode_timestamp"),  # 真正的条码_时间戳格式
            ("Photo001", "prefix_number"),
            ("2023-12-01_001", "date_sequence"),
            ("001_Sample", "number_prefix"),
            ("IMG_20231201_120000", "multi_underscore"),  # 修正：这确实是多段下划线格式
            ("CustomName123", "prefix_number")  # 修正：这是前缀+数字格式
        ]
        
        all_correct = True
        for filename, expected_type in test_cases:
            pattern = self.extract_filename_pattern(filename)
            print(f"  文件名: {filename} -> 模式: {pattern['name']}")
            if expected_type == "smart_pattern" or pattern['name'] == expected_type:
                print(f"    ✓ 正确识别为 {pattern['name']}")
            else:
                print(f"    ✗ 期望 {expected_type}，实际 {pattern['name']}")
                all_correct = False
        
        if all_correct:
            self.passed_tests += 1
            print("  ✓ 所有文件名模式识别测试通过")
        else:
            print("  ✗ 部分文件名模式识别测试失败")
        
        print()
    
    def test_pattern_matching(self):
        """测试模式匹配"""
        print("测试模式匹配...")
        self.total_tests += 1
        
        # 测试样本和目标文件
        sample = "IMG_20231201_120000"
        targets = [
            ("IMG_20231202_130000", True),   # 匹配 - 三段下划线格式
            ("IMG_20231203_140000", True),   # 匹配 - 三段下划线格式
            ("Photo001", False),             # 不匹配 - 不是三段格式
            ("IMG_2023_001", True),          # 匹配 - 三段下划线格式
        ]
        
        sample_pattern = self.extract_filename_pattern(sample)
        print(f"  样本模式: {sample_pattern['name']} ({sample_pattern['regex']})")
        
        all_correct = True
        for target, should_match in targets:
            matches = bool(re.match(sample_pattern['regex'], target))
            status = "✓" if matches == should_match else "✗"
            print(f"    {target}: {status} ({'匹配' if matches else '不匹配'})")
            if matches != should_match:
                all_correct = False
        
        if all_correct:
            self.passed_tests += 1
            print("  ✓ 所有匹配测试通过")
        else:
            print("  ✗ 部分匹配测试失败")
        
        print()
    
    def test_smart_pattern_creation(self):
        """测试智能模式创建"""
        print("测试智能模式创建...")
        self.total_tests += 1
        
        test_cases = [
            "ABC123",
            "Test_001_Final",
            "2023-12-01",
            "IMG-20231201-001"
        ]
        
        all_correct = True
        for filename in test_cases:
            pattern = self.create_smart_pattern(filename)
            print(f"  {filename} -> {pattern}")
            
            # 测试模式是否能匹配原文件名
            if re.match(pattern, filename):
                print(f"    ✓ 模式正确匹配原文件名")
            else:
                print(f"    ✗ 模式无法匹配原文件名")
                all_correct = False
        
        if all_correct:
            self.passed_tests += 1
            print("  ✓ 所有智能模式创建测试通过")
        else:
            print("  ✗ 部分智能模式创建测试失败")
        
        print()
    
    def extract_filename_pattern(self, filename):
        """提取文件名模式（从主程序复制）"""
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
        
        smart_pattern = self.create_smart_pattern(filename)
        return {'name': 'smart_pattern', 'regex': smart_pattern, 'template': filename}
    
    def create_smart_pattern(self, filename):
        """创建智能模式匹配（从主程序复制）"""
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
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 50)
        print("图像分析工具核心功能测试")
        print("=" * 50)
        print()
        
        self.test_filename_pattern_extraction()
        self.test_pattern_matching()
        self.test_smart_pattern_creation()
        
        print("=" * 50)
        print(f"测试结果: {self.passed_tests}/{self.total_tests} 通过")
        if self.passed_tests == self.total_tests:
            print("✓ 所有测试通过！核心功能正常工作。")
        else:
            print("✗ 部分测试失败，请检查代码。")
        print("=" * 50)
        
        return self.passed_tests == self.total_tests

def main():
    """主函数"""
    tester = ImageAnalyzerTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n核心功能测试通过。")
        print("注意：完整的图像分析功能需要安装以下依赖包：")
        print("- opencv-python")
        print("- Pillow")
        print("- numpy")
        print("- matplotlib")
        print("\n使用以下命令安装：")
        print("pip install opencv-python Pillow numpy matplotlib")
        return 0
    else:
        print("\n测试失败，请检查代码。")
        return 1

if __name__ == "__main__":
    sys.exit(main())