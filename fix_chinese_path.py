
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
