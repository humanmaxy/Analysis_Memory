#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X光图像增强算法
专门针对黑色X光图像的高质量增强处理
重点解决噪声过滤和细节保持的平衡问题
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import filters, morphology, restoration, exposure
from skimage.filters import unsharp_mask, wiener
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
from skimage.segmentation import chan_vese
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time


class XRayEnhancer:
    """X光图像增强器"""
    
    def __init__(self):
        self.original_image = None
        self.enhanced_image = None
        self.processing_steps = []
        
    def load_image(self, image_path):
        """加载图像 - 增强版本，支持中文路径和各种编码问题"""
        try:
            print(f"尝试加载图像: {image_path}")
            
            # 方法1: 使用cv2.imdecode处理中文路径
            success, image = self._load_with_numpy_decode(image_path)
            if success:
                print("使用numpy解码方法成功加载")
            else:
                # 方法2: 使用PIL加载
                success, image = self._load_with_pil(image_path)
                if success:
                    print("使用PIL方法成功加载")
                else:
                    # 方法3: 复制到临时文件
                    success, image = self._load_with_temp_copy(image_path)
                    if success:
                        print("使用临时文件方法成功加载")
            
            if not success:
                raise ValueError("所有加载方法都失败了")
            
            # 归一化到0-1范围
            self.original_image = image.astype(np.float64) / 255.0
            self.processing_steps = [("原始图像", self.original_image.copy())]
            
            print(f"图像加载成功! 尺寸: {image.shape}, 数据类型: {image.dtype}")
            return True
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def _load_with_numpy_decode(self, image_path):
        """使用numpy解码方法加载图像"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None
            
            # 检查文件大小
            if os.path.getsize(image_path) == 0:
                return False, None
            
            # 读取文件数据
            with open(image_path, 'rb') as f:
                file_data = f.read()
            
            # 转换为numpy数组
            nparr = np.frombuffer(file_data, np.uint8)
            
            # 使用cv2.imdecode解码
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            return image is not None, image
            
        except Exception as e:
            print(f"numpy解码方法失败: {e}")
            return False, None
    
    def _load_with_pil(self, image_path):
        """使用PIL方法加载图像"""
        try:
            from PIL import Image
            
            with Image.open(image_path) as pil_image:
                # 转换为灰度图
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                
                # 转换为numpy数组
                image = np.array(pil_image)
                
                return True, image
                
        except Exception as e:
            print(f"PIL方法失败: {e}")
            return False, None
    
    def _load_with_temp_copy(self, image_path):
        """使用临时文件复制方法加载图像"""
        try:
            import tempfile
            import shutil
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 复制文件
            shutil.copy2(image_path, temp_path)
            
            # 使用OpenCV读取
            image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return image is not None, image
            
        except Exception as e:
            print(f"临时文件方法失败: {e}")
            return False, None
    
    def adaptive_noise_reduction(self, image, method='multi_scale'):
        """自适应噪声减少"""
        if method == 'multi_scale':
            # 多尺度去噪
            return self._multi_scale_denoising(image)
        elif method == 'nl_means':
            # 非局部均值去噪
            return self._nl_means_denoising(image)
        elif method == 'tv_chambolle':
            # 全变分去噪
            return self._tv_denoising(image)
        elif method == 'bilateral':
            # 双边滤波
            return self._bilateral_denoising(image)
        else:
            return image
    
    def _multi_scale_denoising(self, image):
        """多尺度去噪算法"""
        print("执行多尺度去噪...")
        
        # 创建多个尺度的高斯核
        scales = [0.5, 1.0, 1.5, 2.0, 3.0]
        denoised_images = []
        
        for scale in scales:
            # 高斯滤波
            gaussian = gaussian_filter(image, sigma=scale)
            
            # 双边滤波（保边去噪）
            bilateral = self._bilateral_filter_float(image, d=int(scale*6), 
                                                   sigma_color=0.1, sigma_space=scale*2)
            
            # 结合两种滤波结果
            combined = 0.6 * bilateral + 0.4 * gaussian
            denoised_images.append(combined)
        
        # 自适应权重融合
        weights = self._compute_adaptive_weights(image, denoised_images)
        result = np.zeros_like(image)
        
        for i, denoised in enumerate(denoised_images):
            result += weights[i] * denoised
        
        return np.clip(result, 0, 1)
    
    def _bilateral_filter_float(self, image, d, sigma_color, sigma_space):
        """浮点数双边滤波"""
        # 转换为uint8进行双边滤波
        image_uint8 = (image * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(image_uint8, d, sigma_color*255, sigma_space)
        return filtered.astype(np.float64) / 255.0
    
    def _compute_adaptive_weights(self, original, denoised_images):
        """计算自适应权重"""
        # 计算每个尺度的局部方差
        weights = []
        
        for denoised in denoised_images:
            # 计算梯度幅度
            grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 基于梯度的权重
            weight = np.mean(gradient_magnitude)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _nl_means_denoising(self, image):
        """非局部均值去噪"""
        print("执行非局部均值去噪...")
        
        # 估计噪声水平
        noise_var = restoration.estimate_sigma(image, average_sigmas=True)
        
        # 非局部均值去噪
        denoised = denoise_nl_means(
            image, 
            h=0.8 * noise_var,  # 滤波强度
            fast_mode=False,     # 使用慢速但高质量模式
            patch_size=7,        # 补丁大小
            patch_distance=11,   # 搜索窗口大小
            preserve_range=True
        )
        
        return np.clip(denoised, 0, 1)
    
    def _tv_denoising(self, image):
        """全变分去噪"""
        print("执行全变分去噪...")
        
        # 估计噪声水平来调整权重
        noise_var = restoration.estimate_sigma(image, average_sigmas=True)
        weight = min(0.2, noise_var * 2)  # 自适应权重
        
        denoised = denoise_tv_chambolle(image, weight=weight, max_num_iter=200)
        return np.clip(denoised, 0, 1)
    
    def _bilateral_denoising(self, image):
        """双边滤波去噪"""
        print("执行双边滤波去噪...")
        
        # 多次双边滤波，参数递减
        result = image.copy()
        
        params = [
            (9, 0.1, 10),   # (d, sigma_color, sigma_space)
            (7, 0.08, 8),
            (5, 0.06, 6)
        ]
        
        for d, sigma_color, sigma_space in params:
            result = self._bilateral_filter_float(result, d, sigma_color, sigma_space)
        
        return result
    
    def adaptive_histogram_equalization(self, image, method='clahe'):
        """自适应直方图均衡化"""
        print(f"执行{method}直方图均衡化...")
        
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            image_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_uint8)
            return enhanced.astype(np.float64) / 255.0
            
        elif method == 'adaptive_eq':
            # 自适应均衡化
            return exposure.equalize_adapthist(image, clip_limit=0.03)
            
        elif method == 'gamma_correction':
            # 伽马校正
            gamma = self._estimate_optimal_gamma(image)
            return np.power(image, gamma)
        
        return image
    
    def _estimate_optimal_gamma(self, image):
        """估计最优伽马值"""
        # 基于图像统计特性估计伽马值
        mean_intensity = np.mean(image)
        
        if mean_intensity < 0.3:  # 暗图像
            gamma = 0.5
        elif mean_intensity > 0.7:  # 亮图像
            gamma = 1.5
        else:
            gamma = 1.0
        
        return gamma
    
    def edge_preserving_smoothing(self, image, method='anisotropic'):
        """边缘保护平滑"""
        print(f"执行{method}边缘保护平滑...")
        
        if method == 'anisotropic':
            return self._anisotropic_diffusion(image)
        elif method == 'guided_filter':
            return self._guided_filter(image)
        elif method == 'domain_transform':
            return self._domain_transform_filter(image)
        
        return image
    
    def _anisotropic_diffusion(self, image, num_iter=50, delta_t=0.14, kappa=15):
        """各向异性扩散滤波"""
        # Perona-Malik扩散
        image = image.copy()
        
        for i in range(num_iter):
            # 计算梯度
            nabla_N = np.roll(image, -1, axis=0) - image
            nabla_S = np.roll(image, 1, axis=0) - image
            nabla_E = np.roll(image, -1, axis=1) - image
            nabla_W = np.roll(image, 1, axis=1) - image
            
            # 计算扩散系数
            c_N = np.exp(-(nabla_N/kappa)**2)
            c_S = np.exp(-(nabla_S/kappa)**2)
            c_E = np.exp(-(nabla_E/kappa)**2)
            c_W = np.exp(-(nabla_W/kappa)**2)
            
            # 更新图像
            image += delta_t * (c_N*nabla_N + c_S*nabla_S + c_E*nabla_E + c_W*nabla_W)
        
        return np.clip(image, 0, 1)
    
    def _guided_filter(self, image, radius=8, epsilon=0.04):
        """引导滤波"""
        # 简化的引导滤波实现
        guide = image  # 使用自身作为引导图像
        
        # 计算均值
        mean_I = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(image, cv2.CV_64F, (radius, radius))
        corr_Ip = cv2.boxFilter(guide * image, cv2.CV_64F, (radius, radius))
        cov_Ip = corr_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        # 计算系数
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        # 计算输出
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        output = mean_a * guide + mean_b
        return np.clip(output, 0, 1)
    
    def _domain_transform_filter(self, image, sigma_s=60, sigma_r=0.4, num_iter=3):
        """域变换滤波"""
        # 简化的域变换滤波实现
        result = image.copy()
        
        for i in range(num_iter):
            # 水平滤波
            for y in range(image.shape[0]):
                result[y, :] = self._domain_transform_1d(result[y, :], sigma_s, sigma_r)
            
            # 垂直滤波
            for x in range(image.shape[1]):
                result[:, x] = self._domain_transform_1d(result[:, x], sigma_s, sigma_r)
        
        return result
    
    def _domain_transform_1d(self, signal, sigma_s, sigma_r):
        """一维域变换"""
        n = len(signal)
        if n < 2:
            return signal
        
        # 计算域变换
        dt = np.zeros(n)
        dt[0] = sigma_s
        
        for i in range(1, n):
            dt[i] = sigma_s * np.sqrt(1 + (signal[i] - signal[i-1])**2 / sigma_r**2)
        
        # 累积和
        dt_cumsum = np.cumsum(dt)
        
        # 滤波
        filtered = signal.copy()
        for i in range(1, n):
            weight = np.exp(-dt_cumsum[i] / sigma_s)
            filtered[i] = weight * filtered[i-1] + (1 - weight) * signal[i]
        
        return filtered
    
    def unsharp_masking(self, image, radius=2, amount=1.5):
        """非锐化掩模"""
        print("执行非锐化掩模增强...")
        
        # 使用scikit-image的unsharp_mask
        enhanced = unsharp_mask(image, radius=radius, amount=amount, preserve_range=True)
        
        return np.clip(enhanced, 0, 1)
    
    def multi_scale_retinex(self, image, scales=[15, 80, 250]):
        """多尺度Retinex增强"""
        print("执行多尺度Retinex增强...")
        
        # 防止log(0)
        image_safe = np.maximum(image, 1e-6)
        
        retinex = np.zeros_like(image)
        
        for scale in scales:
            # 高斯模糊作为环绕函数
            surround = gaussian_filter(image_safe, sigma=scale)
            surround = np.maximum(surround, 1e-6)
            
            # 计算Retinex
            single_scale_retinex = np.log(image_safe) - np.log(surround)
            retinex += single_scale_retinex
        
        # 平均
        retinex = retinex / len(scales)
        
        # 归一化到0-1
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        
        return retinex
    
    def morphological_enhancement(self, image, operation='tophat'):
        """形态学增强"""
        print(f"执行{operation}形态学增强...")
        
        # 创建结构元素
        kernel = morphology.disk(3)
        
        if operation == 'tophat':
            # 顶帽变换，增强亮细节
            enhanced = morphology.white_tophat(image, kernel)
            return image + enhanced * 0.5
            
        elif operation == 'blackhat':
            # 黑帽变换，增强暗细节
            enhanced = morphology.black_tophat(image, kernel)
            return image - enhanced * 0.5
            
        elif operation == 'combined':
            # 组合顶帽和黑帽
            white_tophat = morphology.white_tophat(image, kernel)
            black_tophat = morphology.black_tophat(image, kernel)
            return image + white_tophat * 0.3 - black_tophat * 0.3
        
        return image
    
    def frequency_domain_enhancement(self, image):
        """频域增强"""
        print("执行频域增强...")
        
        # FFT变换
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建高通滤波器
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建掩模
        mask = np.ones((rows, cols), dtype=np.float64)
        r = 30  # 截止频率
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
        mask[mask_area] = 0.3  # 保留一些低频
        
        # 应用滤波器
        f_shift_filtered = f_shift * mask
        
        # 逆变换
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        enhanced = np.fft.ifft2(f_ishift)
        enhanced = np.abs(enhanced)
        
        # 与原图像结合
        result = 0.7 * image + 0.3 * enhanced
        
        return np.clip(result, 0, 1)
    
    def comprehensive_enhancement(self, noise_method='multi_scale', 
                                hist_method='clahe', 
                                edge_method='anisotropic',
                                use_unsharp=True,
                                use_retinex=True,
                                use_morphology=True,
                                use_frequency=False):
        """综合增强处理"""
        if self.original_image is None:
            raise ValueError("请先加载图像")
        
        print("开始综合增强处理...")
        result = self.original_image.copy()
        
        # 步骤1: 噪声减少
        if noise_method != 'none':
            result = self.adaptive_noise_reduction(result, noise_method)
            self.processing_steps.append((f"去噪({noise_method})", result.copy()))
        
        # 步骤2: 直方图均衡化
        if hist_method != 'none':
            result = self.adaptive_histogram_equalization(result, hist_method)
            self.processing_steps.append((f"直方图均衡({hist_method})", result.copy()))
        
        # 步骤3: 边缘保护平滑
        if edge_method != 'none':
            result = self.edge_preserving_smoothing(result, edge_method)
            self.processing_steps.append((f"边缘保护({edge_method})", result.copy()))
        
        # 步骤4: 非锐化掩模
        if use_unsharp:
            result = self.unsharp_masking(result)
            self.processing_steps.append(("非锐化掩模", result.copy()))
        
        # 步骤5: 多尺度Retinex
        if use_retinex:
            retinex_result = self.multi_scale_retinex(result)
            # 与原结果融合
            result = 0.7 * result + 0.3 * retinex_result
            self.processing_steps.append(("Retinex增强", result.copy()))
        
        # 步骤6: 形态学增强
        if use_morphology:
            result = self.morphological_enhancement(result, 'combined')
            self.processing_steps.append(("形态学增强", result.copy()))
        
        # 步骤7: 频域增强（可选）
        if use_frequency:
            result = self.frequency_domain_enhancement(result)
            self.processing_steps.append(("频域增强", result.copy()))
        
        # 最终后处理
        result = self._post_processing(result)
        self.processing_steps.append(("后处理", result.copy()))
        
        self.enhanced_image = result
        print("综合增强处理完成!")
        
        return result
    
    def _post_processing(self, image):
        """后处理"""
        # 轻微的高斯平滑去除处理伪影
        smoothed = gaussian_filter(image, sigma=0.5)
        
        # 与原图像混合
        result = 0.9 * image + 0.1 * smoothed
        
        # 确保在有效范围内
        result = np.clip(result, 0, 1)
        
        # 轻微的对比度调整
        result = exposure.rescale_intensity(result, out_range=(0, 1))
        
        return result
    
    def calculate_quality_metrics(self, original, enhanced):
        """计算图像质量指标"""
        metrics = {}
        
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((original - enhanced) ** 2)
        if mse == 0:
            metrics['PSNR'] = float('inf')
        else:
            metrics['PSNR'] = 20 * np.log10(1.0 / np.sqrt(mse))
        
        # 结构相似性指标 (简化版SSIM)
        metrics['SSIM'] = self._calculate_ssim(original, enhanced)
        
        # 对比度改善
        original_contrast = np.std(original)
        enhanced_contrast = np.std(enhanced)
        metrics['Contrast_Improvement'] = enhanced_contrast / original_contrast
        
        # 边缘保持指标
        metrics['Edge_Preservation'] = self._calculate_edge_preservation(original, enhanced)
        
        return metrics
    
    def _calculate_ssim(self, img1, img2, window_size=11, k1=0.01, k2=0.03):
        """计算结构相似性指标"""
        mu1 = gaussian_filter(img1, sigma=1.5)
        mu2 = gaussian_filter(img2, sigma=1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
        
        c1 = (k1) ** 2
        c2 = (k2) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return np.mean(ssim_map)
    
    def _calculate_edge_preservation(self, original, enhanced):
        """计算边缘保持指标"""
        # 计算梯度
        grad_orig_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        grad_orig_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
        grad_orig = np.sqrt(grad_orig_x**2 + grad_orig_y**2)
        
        grad_enh_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_enh_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        grad_enh = np.sqrt(grad_enh_x**2 + grad_enh_y**2)
        
        # 计算相关系数
        correlation = np.corrcoef(grad_orig.flatten(), grad_enh.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0
    
    def save_enhanced_image(self, output_path):
        """保存增强后的图像"""
        if self.enhanced_image is None:
            raise ValueError("没有增强后的图像可保存")
        
        # 转换为uint8
        image_uint8 = (self.enhanced_image * 255).astype(np.uint8)
        
        # 保存
        cv2.imwrite(output_path, image_uint8)
        print(f"增强图像已保存到: {output_path}")


class XRayEnhancerGUI:
    """X光图像增强GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("X光图像增强工具")
        self.root.geometry("1400x900")
        
        self.enhancer = XRayEnhancer()
        self.current_image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件操作", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=80).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="选择图像", command=self.select_image).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="保存结果", command=self.save_result).grid(row=0, column=2, padx=5)
        
        # 参数设置区域
        params_frame = ttk.LabelFrame(main_frame, text="增强参数", padding="5")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=5, padx=(0, 5))
        
        self.setup_parameters(params_frame)
        
        # 控制按钮
        control_frame = ttk.Frame(params_frame)
        control_frame.grid(row=10, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="开始增强", command=self.start_enhancement,
                  style="Accent.TButton").grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="重置参数", command=self.reset_parameters).grid(row=0, column=1, padx=5)
        
        # 进度条
        self.progress = ttk.Progressbar(params_frame, mode='indeterminate')
        self.progress.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 图像显示区域
        self.setup_image_display(main_frame)
        
        # 质量指标显示
        self.setup_metrics_display(main_frame)
    
    def setup_parameters(self, parent):
        """设置参数控制"""
        row = 0
        
        # 去噪方法
        ttk.Label(parent, text="去噪方法:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.noise_method = tk.StringVar(value='multi_scale')
        noise_combo = ttk.Combobox(parent, textvariable=self.noise_method, 
                                  values=['multi_scale', 'nl_means', 'tv_chambolle', 'bilateral', 'none'])
        noise_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # 直方图均衡方法
        ttk.Label(parent, text="直方图均衡:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.hist_method = tk.StringVar(value='clahe')
        hist_combo = ttk.Combobox(parent, textvariable=self.hist_method,
                                 values=['clahe', 'adaptive_eq', 'gamma_correction', 'none'])
        hist_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # 边缘保护方法
        ttk.Label(parent, text="边缘保护:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.edge_method = tk.StringVar(value='anisotropic')
        edge_combo = ttk.Combobox(parent, textvariable=self.edge_method,
                                 values=['anisotropic', 'guided_filter', 'domain_transform', 'none'])
        edge_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # 增强选项
        ttk.Label(parent, text="增强选项:").grid(row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.use_unsharp = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="非锐化掩模", variable=self.use_unsharp).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
        row += 1
        
        self.use_retinex = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Retinex增强", variable=self.use_retinex).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
        row += 1
        
        self.use_morphology = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="形态学增强", variable=self.use_morphology).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
        row += 1
        
        self.use_frequency = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="频域增强", variable=self.use_frequency).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
        row += 1
        
        # 配置列权重
        parent.columnconfigure(1, weight=1)
    
    def setup_image_display(self, parent):
        """设置图像显示区域"""
        # 创建笔记本控件
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 原始图像标签页
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="原始图像")
        
        self.original_canvas = tk.Canvas(self.original_frame, bg='black')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 增强图像标签页
        self.enhanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.enhanced_frame, text="增强图像")
        
        self.enhanced_canvas = tk.Canvas(self.enhanced_frame, bg='black')
        self.enhanced_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理步骤标签页
        self.steps_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.steps_frame, text="处理步骤")
        
        # 创建处理步骤的滚动框架
        steps_canvas = tk.Canvas(self.steps_frame)
        scrollbar = ttk.Scrollbar(self.steps_frame, orient="vertical", command=steps_canvas.yview)
        self.scrollable_frame = ttk.Frame(steps_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: steps_canvas.configure(scrollregion=steps_canvas.bbox("all"))
        )
        
        steps_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        steps_canvas.configure(yscrollcommand=scrollbar.set)
        
        steps_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_metrics_display(self, parent):
        """设置质量指标显示"""
        metrics_frame = ttk.LabelFrame(parent, text="图像质量指标", padding="5")
        metrics_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.metrics_text = tk.Text(metrics_frame, height=4, width=80)
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)
    
    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择X光图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.dcm"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.file_path.set(file_path)
            self.current_image_path = file_path
            
            # 加载图像
            if self.enhancer.load_image(file_path):
                self.display_original_image()
                messagebox.showinfo("成功", "图像加载成功!")
            else:
                messagebox.showerror("错误", "无法加载图像文件")
    
    def display_original_image(self):
        """显示原始图像"""
        if self.enhancer.original_image is not None:
            self.display_image_on_canvas(self.enhancer.original_image, self.original_canvas)
    
    def display_image_on_canvas(self, image, canvas):
        """在画布上显示图像"""
        # 转换为PIL图像
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # 获取画布大小
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 计算缩放比例
            img_width, img_height = image_pil.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h) * 0.9  # 留一些边距
            
            # 缩放图像
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image_resized = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image_resized)
            
            # 清除画布并显示图像
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            canvas.image = photo  # 保持引用
    
    def start_enhancement(self):
        """开始图像增强"""
        if self.enhancer.original_image is None:
            messagebox.showerror("错误", "请先选择图像文件")
            return
        
        # 在新线程中执行增强
        self.progress.start()
        threading.Thread(target=self.perform_enhancement, daemon=True).start()
    
    def perform_enhancement(self):
        """执行增强处理"""
        try:
            start_time = time.time()
            
            # 获取参数
            params = {
                'noise_method': self.noise_method.get(),
                'hist_method': self.hist_method.get(),
                'edge_method': self.edge_method.get(),
                'use_unsharp': self.use_unsharp.get(),
                'use_retinex': self.use_retinex.get(),
                'use_morphology': self.use_morphology.get(),
                'use_frequency': self.use_frequency.get()
            }
            
            # 执行增强
            enhanced_image = self.enhancer.comprehensive_enhancement(**params)
            
            processing_time = time.time() - start_time
            
            # 在主线程中更新UI
            self.root.after(0, self.enhancement_completed, enhanced_image, processing_time)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"增强处理失败: {str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def enhancement_completed(self, enhanced_image, processing_time):
        """增强完成后的处理"""
        # 显示增强图像
        self.display_image_on_canvas(enhanced_image, self.enhanced_canvas)
        
        # 显示处理步骤
        self.display_processing_steps()
        
        # 计算和显示质量指标
        metrics = self.enhancer.calculate_quality_metrics(
            self.enhancer.original_image, enhanced_image
        )
        
        metrics_text = f"处理时间: {processing_time:.2f}秒\n"
        metrics_text += f"PSNR: {metrics['PSNR']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics['SSIM']:.4f}\n"
        metrics_text += f"对比度改善: {metrics['Contrast_Improvement']:.2f}x\n"
        metrics_text += f"边缘保持: {metrics['Edge_Preservation']:.4f}"
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, metrics_text)
        
        # 切换到增强图像标签页
        self.notebook.select(1)
        
        messagebox.showinfo("完成", f"图像增强完成!\n处理时间: {processing_time:.2f}秒")
    
    def display_processing_steps(self):
        """显示处理步骤"""
        # 清除之前的步骤
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 显示每个处理步骤
        for i, (step_name, step_image) in enumerate(self.enhancer.processing_steps):
            step_frame = ttk.LabelFrame(self.scrollable_frame, text=f"步骤 {i+1}: {step_name}", padding="5")
            step_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 创建小图显示
            canvas = tk.Canvas(step_frame, width=200, height=150, bg='black')
            canvas.pack()
            
            # 显示步骤图像
            self.display_small_image(step_image, canvas)
    
    def display_small_image(self, image, canvas):
        """在小画布上显示图像"""
        # 转换为PIL图像
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # 缩放到合适大小
        image_resized = image_pil.resize((180, 135), Image.Resampling.LANCZOS)
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(image_resized)
        
        # 显示图像
        canvas.create_image(100, 75, image=photo)
        canvas.image = photo  # 保持引用
    
    def save_result(self):
        """保存增强结果"""
        if self.enhancer.enhanced_image is None:
            messagebox.showerror("错误", "没有增强结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存增强图像",
            defaultextension=".png",
            filetypes=[
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg"),
                ("TIFF图像", "*.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.enhancer.save_enhanced_image(file_path)
                messagebox.showinfo("成功", f"图像已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def reset_parameters(self):
        """重置参数"""
        self.noise_method.set('multi_scale')
        self.hist_method.set('clahe')
        self.edge_method.set('anisotropic')
        self.use_unsharp.set(True)
        self.use_retinex.set(True)
        self.use_morphology.set(True)
        self.use_frequency.set(False)


def main():
    """主函数"""
    root = tk.Tk()
    app = XRayEnhancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()