#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分析工具启动器
检查依赖并启动主程序
"""

import sys
import subprocess
import importlib.util

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow', 
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'tkinter': 'tkinter (系统内置)'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            else:
                importlib.import_module(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装 (需要: {install_name})")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    print("图像文件分析工具")
    print("=" * 30)
    
    # 检查依赖
    if not check_dependencies():
        input("按回车键退出...")
        sys.exit(1)
    
    print("\n所有依赖已满足，启动程序...")
    
    # 启动主程序
    try:
        from image_analyzer import main as analyzer_main
        analyzer_main()
    except Exception as e:
        print(f"启动程序时出错: {e}")
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()