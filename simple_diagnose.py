#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的图像问题诊断工具
不依赖OpenCV等外部库，仅使用Python标准库
"""

import os
import sys


def diagnose_file_basic(file_path):
    """基础文件诊断"""
    print("=" * 80)
    print(f"诊断图像文件: {file_path}")
    print("=" * 80)
    
    # 1. 文件存在性检查
    print("\n1. 文件存在性检查:")
    print("-" * 40)
    
    exists = os.path.exists(file_path)
    print(f"文件存在: {exists}")
    
    if not exists:
        print("❌ 文件不存在！")
        
        # 检查目录是否存在
        dir_path = os.path.dirname(file_path)
        print(f"检查目录: {dir_path}")
        
        if os.path.exists(dir_path):
            print("✓ 目录存在")
            try:
                files = os.listdir(dir_path)
                print(f"目录中有 {len(files)} 个文件")
                
                # 显示前几个文件
                print("目录内容（前10个）:")
                for i, f in enumerate(files[:10]):
                    print(f"  {i+1:2d}. {f}")
                
                if len(files) > 10:
                    print(f"  ... 还有 {len(files) - 10} 个文件")
                    
            except Exception as e:
                print(f"❌ 无法读取目录: {e}")
        else:
            print("❌ 目录也不存在！")
        
        return False
    
    # 2. 文件基本信息
    print("\n2. 文件基本信息:")
    print("-" * 40)
    
    try:
        stat = os.stat(file_path)
        file_size = stat.st_size
        
        print(f"文件大小: {file_size:,} 字节")
        print(f"文件大小: {file_size/1024:.2f} KB")
        print(f"文件大小: {file_size/1024/1024:.2f} MB")
        
        if file_size == 0:
            print("❌ 文件为空！")
            return False
        elif file_size < 100:
            print("⚠️ 文件很小，可能损坏")
        
        # 权限检查
        readable = os.access(file_path, os.R_OK)
        print(f"可读权限: {readable}")
        
        if not readable:
            print("❌ 没有读取权限！")
            return False
            
    except Exception as e:
        print(f"❌ 获取文件信息失败: {e}")
        return False
    
    # 3. 路径分析
    print("\n3. 路径分析:")
    print("-" * 40)
    
    print(f"完整路径: {file_path}")
    print(f"目录: {os.path.dirname(file_path)}")
    print(f"文件名: {os.path.basename(file_path)}")
    
    # 检查中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in file_path)
    has_special = any(ord(char) > 127 for char in file_path)
    
    print(f"包含中文字符: {has_chinese}")
    print(f"包含特殊字符: {has_special}")
    
    if has_chinese:
        print("⚠️ 路径包含中文字符，这是OpenCV读取失败的主要原因！")
    
    if has_special:
        print("⚠️ 路径包含非ASCII字符")
    
    # 4. 文件格式检查
    print("\n4. 文件格式检查:")
    print("-" * 40)
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        print(f"文件头(前16字节): {header}")
        print(f"文件头(十六进制): {header.hex()}")
        
        # PNG文件签名检查
        png_signature = b'\x89PNG\r\n\x1a\n'
        is_valid_png = header.startswith(png_signature)
        
        print(f"PNG格式验证: {is_valid_png}")
        
        if not is_valid_png:
            print("❌ 不是有效的PNG文件！")
            
            # 检查其他常见格式
            if header.startswith(b'\xff\xd8'):
                print("✓ 检测到JPEG格式")
            elif header.startswith(b'BM'):
                print("✓ 检测到BMP格式")
            elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                print("✓ 检测到GIF格式")
            else:
                print("❓ 未知或损坏的图像格式")
        else:
            print("✓ 有效的PNG文件")
            
    except Exception as e:
        print(f"❌ 文件格式检查失败: {e}")
        return False
    
    return True


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 80)
    print("💡 解决方案")
    print("=" * 80)
    
    print("\n🔧 主要解决方案（按优先级排序）:")
    
    print("\n1. 【推荐】使用增强的加载方法:")
    print("   - 使用我提供的修改后的 xray_enhancement.py")
    print("   - 该版本包含3种加载方法，可以处理中文路径")
    
    print("\n2. 路径问题解决:")
    print("   - 将图像文件移动到纯英文路径")
    print("   - 例如: C:/images/sample.png")
    print("   - 避免路径中的中文、空格、特殊字符")
    
    print("\n3. 文件重命名:")
    print("   - 将文件名改为英文")
    print("   - 例如: 15_13_16_0594.png -> image_001.png")
    
    print("\n4. 使用替代库:")
    print("   - 使用PIL/Pillow库替代OpenCV")
    print("   - 使用skimage或matplotlib加载图像")
    
    print("\n5. 代码修改方案:")
    print("   - 使用 cv2.imdecode 替代 cv2.imread")
    print("   - 先读取文件字节，再解码为图像")


def create_fix_code():
    """创建修复代码示例"""
    print("\n" + "=" * 80)
    print("📝 修复代码示例")
    print("=" * 80)
    
    code = '''
# 方法1: 使用cv2.imdecode处理中文路径
import cv2
import numpy as np

def load_image_chinese_path(file_path):
    """加载包含中文路径的图像"""
    try:
        # 读取文件字节
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # 转换为numpy数组
        nparr = np.frombuffer(file_data, np.uint8)
        
        # 解码图像
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            print(f"成功加载图像: {image.shape}")
            return image
        else:
            print("解码失败")
            return None
            
    except Exception as e:
        print(f"加载失败: {e}")
        return None

# 方法2: 使用PIL库
from PIL import Image

def load_image_with_pil(file_path):
    """使用PIL加载图像"""
    try:
        with Image.open(file_path) as pil_image:
            # 转换为灰度图
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            # 转换为numpy数组
            image = np.array(pil_image)
            print(f"PIL加载成功: {image.shape}")
            return image
            
    except Exception as e:
        print(f"PIL加载失败: {e}")
        return None

# 使用示例
file_path = r"F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/15_13_16_0594.png"

# 尝试方法1
image = load_image_chinese_path(file_path)

# 如果方法1失败，尝试方法2
if image is None:
    image = load_image_with_pil(file_path)

if image is not None:
    print("图像加载成功！")
    # 继续处理图像...
else:
    print("所有方法都失败了")
'''
    
    print(code)
    
    # 保存到文件
    try:
        with open("fix_chinese_path.py", "w", encoding="utf-8") as f:
            f.write(code)
        print("\n📁 代码已保存到: fix_chinese_path.py")
    except Exception as e:
        print(f"保存代码失败: {e}")


def main():
    """主函数"""
    # 您遇到问题的文件路径
    test_files = [
        "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/15_13_16_0594.png",
        "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/16_59_44_0419.png"
    ]
    
    print("OpenCV图像加载问题诊断工具")
    print("专门分析中文路径和编码问题")
    
    all_ok = True
    
    for file_path in test_files:
        result = diagnose_file_basic(file_path)
        if not result:
            all_ok = False
        print("\n" + "="*50 + "\n")
    
    # 提供解决方案
    provide_solutions()
    
    # 创建修复代码
    create_fix_code()
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    
    if all_ok:
        print("✓ 文件检查通过，问题主要是OpenCV对中文路径的支持问题")
    else:
        print("❌ 发现文件问题，请先解决文件本身的问题")
    
    print("\n推荐解决方案:")
    print("1. 使用修改后的 xray_enhancement.py（已包含中文路径支持）")
    print("2. 或者将文件移动到英文路径")
    print("3. 或者使用生成的 fix_chinese_path.py 代码")


if __name__ == "__main__":
    main()