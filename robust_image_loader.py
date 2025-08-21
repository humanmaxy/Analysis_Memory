#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒的图像加载器
专门处理中文路径、文件编码等问题
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys


class RobustImageLoader:
    """鲁棒的图像加载器"""
    
    @staticmethod
    def load_image_safe(file_path):
        """
        安全加载图像，处理各种可能的问题
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            tuple: (success, image, error_message)
        """
        # 方法1: 使用cv2.imdecode处理中文路径
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, None, f"文件不存在: {file_path}"
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, None, f"文件为空: {file_path}"
            
            # 方法1: 使用numpy读取文件，避免中文路径问题
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # 转换为numpy数组
            nparr = np.frombuffer(file_data, np.uint8)
            
            # 使用cv2.imdecode解码
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                return True, image, None
                
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 使用PIL加载
        try:
            with Image.open(file_path) as pil_image:
                # 转换为灰度图
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                
                # 转换为numpy数组
                image = np.array(pil_image)
                
                return True, image, None
                
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 方法3: 复制文件到临时位置（英文路径）
        try:
            import tempfile
            import shutil
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 复制文件
            shutil.copy2(file_path, temp_path)
            
            # 使用OpenCV读取
            image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            if image is not None:
                return True, image, None
                
        except Exception as e:
            print(f"方法3失败: {e}")
        
        return False, None, f"所有加载方法都失败了: {file_path}"
    
    @staticmethod
    def validate_image_file(file_path):
        """
        验证图像文件的完整性
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # 检查文件存在性
            if not os.path.exists(file_path):
                return False, "文件不存在"
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "文件为空"
            
            if file_size < 100:  # 小于100字节可能不是有效图像
                return False, "文件太小，可能损坏"
            
            # 检查文件扩展名
            _, ext = os.path.splitext(file_path.lower())
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
            if ext not in valid_extensions:
                return False, f"不支持的文件格式: {ext}"
            
            # 尝试读取文件头
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # 检查PNG文件头
            if ext == '.png':
                png_signature = b'\x89PNG\r\n\x1a\n'
                if not header.startswith(png_signature):
                    return False, "PNG文件头损坏"
            
            # 检查JPEG文件头
            elif ext in {'.jpg', '.jpeg'}:
                if not header.startswith(b'\xff\xd8'):
                    return False, "JPEG文件头损坏"
            
            return True, None
            
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
    
    @staticmethod
    def fix_file_path(file_path):
        """
        修复文件路径
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            str: 修复后的文件路径
        """
        # 标准化路径分隔符
        fixed_path = file_path.replace('\\', '/')
        
        # 处理路径编码
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'cp936', 'latin1']
            for encoding in encodings:
                try:
                    if isinstance(fixed_path, bytes):
                        fixed_path = fixed_path.decode(encoding)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
        except Exception:
            pass
        
        return fixed_path
    
    @staticmethod
    def get_file_info(file_path):
        """
        获取文件详细信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 文件信息
        """
        info = {
            'path': file_path,
            'exists': False,
            'size': 0,
            'readable': False,
            'extension': '',
            'encoding_issues': False
        }
        
        try:
            info['exists'] = os.path.exists(file_path)
            if info['exists']:
                info['size'] = os.path.getsize(file_path)
                info['readable'] = os.access(file_path, os.R_OK)
                info['extension'] = os.path.splitext(file_path.lower())[1]
        except Exception as e:
            info['error'] = str(e)
            # 可能是编码问题
            info['encoding_issues'] = True
        
        return info


def test_image_loading():
    """测试图像加载功能"""
    print("=" * 60)
    print("图像加载测试")
    print("=" * 60)
    
    # 测试文件路径（请根据实际情况修改）
    test_paths = [
        "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/15_13_16_0594.png",
        "F:/data/阳极多胶/6mm×5mm/01 OrgImgC1/16_59_44_0419.png"
    ]
    
    loader = RobustImageLoader()
    
    for file_path in test_paths:
        print(f"\n测试文件: {file_path}")
        print("-" * 50)
        
        # 获取文件信息
        info = loader.get_file_info(file_path)
        print("文件信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 验证文件
        is_valid, error_msg = loader.validate_image_file(file_path)
        print(f"\n文件验证: {'通过' if is_valid else '失败'}")
        if error_msg:
            print(f"  错误: {error_msg}")
        
        # 尝试加载图像
        print("\n尝试加载图像...")
        success, image, error = loader.load_image_safe(file_path)
        
        if success:
            print(f"  ✅ 加载成功!")
            print(f"  图像尺寸: {image.shape}")
            print(f"  数据类型: {image.dtype}")
            print(f"  像素值范围: {image.min()} - {image.max()}")
        else:
            print(f"  ❌ 加载失败: {error}")
            
            # 提供解决建议
            print("\n💡 解决建议:")
            if not info['exists']:
                print("  - 检查文件路径是否正确")
                print("  - 确认文件是否存在")
            elif info['size'] == 0:
                print("  - 文件为空，可能损坏")
                print("  - 尝试重新获取文件")
            elif not info['readable']:
                print("  - 检查文件权限")
                print("  - 尝试以管理员身份运行程序")
            elif info.get('encoding_issues'):
                print("  - 路径包含特殊字符，尝试:")
                print("    1. 将文件移到英文路径")
                print("    2. 重命名文件为英文名")
                print("    3. 使用短路径名")
            else:
                print("  - 文件可能损坏，尝试用其他软件打开验证")
                print("  - 检查文件格式是否正确")


def create_enhanced_xray_loader():
    """创建增强的X光图像加载器"""
    
    class EnhancedXRayLoader:
        """增强的X光图像加载器"""
        
        def __init__(self):
            self.loader = RobustImageLoader()
        
        def load_image(self, file_path):
            """
            加载图像，增强版本
            
            Args:
                file_path: 图像文件路径
                
            Returns:
                bool: 是否成功加载
            """
            print(f"尝试加载图像: {file_path}")
            
            # 修复路径
            fixed_path = self.loader.fix_file_path(file_path)
            if fixed_path != file_path:
                print(f"路径已修复: {fixed_path}")
            
            # 验证文件
            is_valid, error_msg = self.loader.validate_image_file(fixed_path)
            if not is_valid:
                print(f"文件验证失败: {error_msg}")
                return False
            
            # 加载图像
            success, image, error = self.loader.load_image_safe(fixed_path)
            
            if success:
                # 归一化到0-1范围
                self.original_image = image.astype(np.float64) / 255.0
                print(f"图像加载成功! 尺寸: {image.shape}")
                return True
            else:
                print(f"图像加载失败: {error}")
                return False
    
    return EnhancedXRayLoader()


if __name__ == "__main__":
    test_image_loading()