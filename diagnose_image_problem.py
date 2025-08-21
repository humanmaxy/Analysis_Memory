#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加载问题诊断工具
专门诊断和解决OpenCV无法读取图像的问题
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil


def diagnose_image_file(file_path):
    """
    诊断图像文件问题
    
    Args:
        file_path: 图像文件路径
    """
    print("=" * 80)
    print(f"诊断图像文件: {file_path}")
    print("=" * 80)
    
    # 1. 基本文件检查
    print("\n1. 基本文件检查:")
    print("-" * 40)
    
    try:
        exists = os.path.exists(file_path)
        print(f"✓ 文件存在: {exists}")
        
        if not exists:
            print("❌ 文件不存在！请检查路径是否正确。")
            
            # 提供路径建议
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path):
                print(f"📁 目录存在，列出目录内容:")
                try:
                    files = os.listdir(dir_path)
                    for f in files[:10]:  # 只显示前10个文件
                        print(f"   - {f}")
                    if len(files) > 10:
                        print(f"   ... 还有 {len(files) - 10} 个文件")
                except Exception as e:
                    print(f"   无法列出目录内容: {e}")
            else:
                print("❌ 目录也不存在！")
            return
        
        # 文件大小
        file_size = os.path.getsize(file_path)
        print(f"✓ 文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
        
        if file_size == 0:
            print("❌ 文件为空！")
            return
        elif file_size < 100:
            print("⚠️ 文件太小，可能损坏")
        
        # 文件权限
        readable = os.access(file_path, os.R_OK)
        print(f"✓ 可读权限: {readable}")
        
        if not readable:
            print("❌ 没有读取权限！")
            return
            
    except Exception as e:
        print(f"❌ 基本检查失败: {e}")
        return
    
    # 2. 路径编码检查
    print("\n2. 路径编码检查:")
    print("-" * 40)
    
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in file_path)
    has_special_chars = any(ord(char) > 127 for char in file_path)
    
    print(f"✓ 包含中文字符: {has_chinese}")
    print(f"✓ 包含特殊字符: {has_special_chars}")
    
    if has_chinese or has_special_chars:
        print("⚠️ 路径包含非ASCII字符，可能导致OpenCV读取失败")
    
    # 3. 文件格式检查
    print("\n3. 文件格式检查:")
    print("-" * 40)
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        print(f"✓ 文件头 (hex): {header.hex()}")
        
        # 检查PNG签名
        png_signature = b'\x89PNG\r\n\x1a\n'
        is_png = header.startswith(png_signature)
        print(f"✓ PNG格式验证: {is_png}")
        
        if not is_png:
            print("❌ 不是有效的PNG文件！")
            
            # 检查其他格式
            if header.startswith(b'\xff\xd8'):
                print("✓ 检测到JPEG格式")
            elif header.startswith(b'BM'):
                print("✓ 检测到BMP格式")
            elif header.startswith(b'RIFF') and b'WEBP' in header:
                print("✓ 检测到WEBP格式")
            else:
                print("❓ 未知文件格式")
                
    except Exception as e:
        print(f"❌ 文件格式检查失败: {e}")
    
    # 4. OpenCV加载测试
    print("\n4. OpenCV加载测试:")
    print("-" * 40)
    
    try:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            print(f"✓ OpenCV加载成功: {image.shape}, {image.dtype}")
        else:
            print("❌ OpenCV加载失败")
            
            # 尝试不同的加载方式
            print("\n尝试替代方法:")
            
            # 方法1: 使用cv2.imdecode
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                nparr = np.frombuffer(file_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    print(f"✓ cv2.imdecode成功: {image.shape}")
                else:
                    print("❌ cv2.imdecode失败")
            except Exception as e:
                print(f"❌ cv2.imdecode异常: {e}")
            
    except Exception as e:
        print(f"❌ OpenCV测试异常: {e}")
    
    # 5. PIL加载测试
    print("\n5. PIL加载测试:")
    print("-" * 40)
    
    try:
        with Image.open(file_path) as pil_image:
            print(f"✓ PIL加载成功: {pil_image.size}, {pil_image.mode}")
            
            # 转换测试
            if pil_image.mode != 'L':
                gray_image = pil_image.convert('L')
                print(f"✓ 转换为灰度图: {gray_image.size}")
            
            # 转换为numpy数组
            np_array = np.array(pil_image)
            print(f"✓ 转换为numpy数组: {np_array.shape}, {np_array.dtype}")
            
    except Exception as e:
        print(f"❌ PIL加载失败: {e}")
    
    # 6. 临时文件测试
    print("\n6. 临时文件测试:")
    print("-" * 40)
    
    try:
        # 创建临时文件（英文路径）
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        print(f"✓ 临时文件路径: {temp_path}")
        
        # 复制文件
        shutil.copy2(file_path, temp_path)
        print("✓ 文件复制成功")
        
        # 尝试用OpenCV读取临时文件
        temp_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        
        if temp_image is not None:
            print(f"✓ 临时文件OpenCV加载成功: {temp_image.shape}")
            print("💡 建议: 路径问题，建议将文件移到英文路径或重命名")
        else:
            print("❌ 临时文件OpenCV加载也失败")
            print("💡 建议: 文件本身可能损坏")
        
        # 清理临时文件
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ 临时文件测试失败: {e}")


def provide_solutions(file_path):
    """提供解决方案"""
    print("\n" + "=" * 80)
    print("💡 解决方案建议")
    print("=" * 80)
    
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in file_path)
    
    if has_chinese:
        print("\n🔧 中文路径问题解决方案:")
        print("1. 将文件移动到纯英文路径")
        print("2. 重命名文件和文件夹为英文")
        print("3. 使用程序中的鲁棒加载方法")
        print("4. 在代码中使用 cv2.imdecode 替代 cv2.imread")
    
    print("\n🔧 通用解决方案:")
    print("1. 检查文件完整性:")
    print("   - 用图像查看器打开文件验证")
    print("   - 重新下载或复制文件")
    
    print("\n2. 权限问题:")
    print("   - 以管理员身份运行程序")
    print("   - 检查文件夹权限设置")
    
    print("\n3. 编码问题:")
    print("   - 使用UTF-8编码保存Python文件")
    print("   - 在代码开头添加: # -*- coding: utf-8 -*-")
    
    print("\n4. OpenCV替代方案:")
    print("   - 使用PIL/Pillow库加载图像")
    print("   - 使用skimage.io.imread")
    print("   - 使用matplotlib.pyplot.imread")


def create_test_fix_script(file_path):
    """创建测试修复脚本"""
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成的图像加载修复脚本
"""

import cv2
import numpy as np
from PIL import Image
import os

def load_image_robust(file_path):
    """鲁棒的图像加载函数"""
    
    # 方法1: cv2.imdecode (推荐用于中文路径)
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            print("✓ 使用cv2.imdecode成功加载")
            return image
    except Exception as e:
        print(f"cv2.imdecode失败: {{e}}")
    
    # 方法2: PIL
    try:
        with Image.open(file_path) as pil_image:
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            image = np.array(pil_image)
            print("✓ 使用PIL成功加载")
            return image
    except Exception as e:
        print(f"PIL加载失败: {{e}}")
    
    # 方法3: 临时文件
    try:
        import tempfile
        import shutil
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        shutil.copy2(file_path, temp_path)
        image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.unlink(temp_path)
        
        if image is not None:
            print("✓ 使用临时文件成功加载")
            return image
    except Exception as e:
        print(f"临时文件方法失败: {{e}}")
    
    print("❌ 所有方法都失败了")
    return None

# 测试加载您的文件
if __name__ == "__main__":
    file_path = r"{file_path}"
    print(f"测试加载: {{file_path}}")
    
    image = load_image_robust(file_path)
    
    if image is not None:
        print(f"成功! 图像尺寸: {{image.shape}}")
        print(f"数据类型: {{image.dtype}}")
        print(f"像素值范围: {{image.min()}} - {{image.max()}}")
        
        # 保存到当前目录作为测试
        cv2.imwrite("test_loaded_image.png", image)
        print("测试图像已保存为: test_loaded_image.png")
    else:
        print("加载失败!")
'''
    
    with open("fix_image_loading.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"\n📝 已生成修复脚本: fix_image_loading.py")
    print("运行该脚本测试图像加载: python fix_image_loading.py")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python diagnose_image_problem.py <图像文件路径>")
        print("示例: python diagnose_image_problem.py 'F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/15_13_16_0594.png'")
        return
    
    file_path = sys.argv[1]
    
    # 诊断文件
    diagnose_image_file(file_path)
    
    # 提供解决方案
    provide_solutions(file_path)
    
    # 创建修复脚本
    create_test_fix_script(file_path)
    
    print("\n" + "=" * 80)
    print("诊断完成!")
    print("=" * 80)


if __name__ == "__main__":
    # 如果没有命令行参数，使用您提供的示例路径
    if len(sys.argv) == 1:
        test_files = [
            "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/15_13_16_0594.png",
            "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/16_59_44_0419.png"
        ]
        
        for file_path in test_files:
            diagnose_image_file(file_path)
            print("\n" + "="*20 + " 下一个文件 " + "="*20 + "\n")
        
        provide_solutions(test_files[0])
        create_test_fix_script(test_files[0])
    else:
        main()