#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高精度X光图像增强算法
专注于细节保持和精度最大化
使用更先进的算法减少细节损失
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import os


class AdvancedXRayEnhancer:
    """高精度X光图像增强器"""
    
    def __init__(self):
        self.original_image = None
        self.enhanced_image = None
        self.processing_steps = []
        
    def load_image(self, image_path):
        """鲁棒的图像加载方法"""
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
            
            if not success:
                raise ValueError("所有加载方法都失败了")
            
            # 转换为双精度浮点数，保持最大精度
            self.original_image = image.astype(np.float64) / 255.0
            self.processing_steps = [("原始图像", self.original_image.copy())]
            
            print(f"图像加载成功! 尺寸: {image.shape}, 精度: {self.original_image.dtype}")
            return True
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return False
    
    def _load_with_numpy_decode(self, image_path):
        """使用numpy解码方法加载图像"""
        try:
            if not os.path.exists(image_path):
                return False, None
            
            with open(image_path, 'rb') as f:
                file_data = f.read()
            
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            return image is not None, image
            
        except Exception as e:
            print(f"numpy解码方法失败: {e}")
            return False, None
    
    def _load_with_pil(self, image_path):
        """使用PIL方法加载图像"""
        try:
            with Image.open(image_path) as pil_image:
                if pil_image.mode != 'L':
                    pil_image = pil_image.convert('L')
                image = np.array(pil_image)
                return True, image
        except Exception as e:
            print(f"PIL方法失败: {e}")
            return False, None
    
    def wavelet_denoising(self, image, wavelet='db8', levels=4):
        """
        小波去噪 - 保持细节的高精度去噪方法
        使用软阈值和自适应阈值选择
        """
        print("执行小波去噪...")
        
        try:
            from pywt import wavedec2, waverec2, threshold
            
            # 小波分解
            coeffs = wavedec2(image, wavelet, levels=levels)
            
            # 估计噪声标准差（使用最高频子带）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # 自适应阈值计算
            threshold_value = sigma * np.sqrt(2 * np.log(image.size))
            
            # 对细节系数进行软阈值处理
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs)):
                if isinstance(coeffs[i], tuple):
                    # 处理每个方向的子带
                    coeffs_thresh[i] = tuple(
                        threshold(c, threshold_value, 'soft') for c in coeffs[i]
                    )
                else:
                    coeffs_thresh[i] = threshold(coeffs[i], threshold_value, 'soft')
            
            # 小波重构
            denoised = waverec2(coeffs_thresh, wavelet)
            
            return np.clip(denoised, 0, 1)
            
        except ImportError:
            print("PyWavelets未安装，使用替代方法...")
            return self._alternative_wavelet_denoising(image)
    
    def _alternative_wavelet_denoising(self, image):
        """替代的小波去噪方法"""
        # 使用Haar小波的简化实现
        def haar_transform_2d(data):
            """2D Haar小波变换"""
            rows, cols = data.shape
            result = data.copy().astype(np.float64)
            
            # 行变换
            for i in range(rows):
                result[i, :] = self._haar_transform_1d(result[i, :])
            
            # 列变换
            for j in range(cols):
                result[:, j] = self._haar_transform_1d(result[:, j])
            
            return result
        
        def haar_inverse_2d(data):
            """2D Haar小波逆变换"""
            rows, cols = data.shape
            result = data.copy().astype(np.float64)
            
            # 列逆变换
            for j in range(cols):
                result[:, j] = self._haar_inverse_1d(result[:, j])
            
            # 行逆变换
            for i in range(rows):
                result[i, :] = self._haar_inverse_1d(result[i, :])
            
            return result
        
        # 执行多级小波变换
        transformed = image.copy()
        for level in range(3):  # 3级分解
            h, w = transformed.shape
            if h < 4 or w < 4:
                break
            
            # 对左上角区域进行变换
            region = transformed[:h//2*2, :w//2*2]
            transformed[:h//2*2, :w//2*2] = haar_transform_2d(region)
            
            # 软阈值处理高频部分
            threshold = 0.1 * np.std(transformed)
            mask = np.abs(transformed) > threshold
            transformed = transformed * mask + np.sign(transformed) * np.maximum(0, np.abs(transformed) - threshold) * (1 - mask)
        
        # 逆变换
        for level in range(3):
            h, w = transformed.shape
            region = transformed[:h//2*2, :w//2*2]
            transformed[:h//2*2, :w//2*2] = haar_inverse_2d(region)
        
        return np.clip(transformed, 0, 1)
    
    def _haar_transform_1d(self, data):
        """1D Haar变换"""
        n = len(data)
        if n < 2:
            return data
        
        result = np.zeros(n)
        half = n // 2
        
        # 低频部分（平均）
        for i in range(half):
            result[i] = (data[2*i] + data[2*i+1]) / np.sqrt(2)
        
        # 高频部分（差分）
        for i in range(half):
            result[half + i] = (data[2*i] - data[2*i+1]) / np.sqrt(2)
        
        return result
    
    def _haar_inverse_1d(self, data):
        """1D Haar逆变换"""
        n = len(data)
        if n < 2:
            return data
        
        result = np.zeros(n)
        half = n // 2
        
        for i in range(half):
            result[2*i] = (data[i] + data[half + i]) / np.sqrt(2)
            result[2*i+1] = (data[i] - data[half + i]) / np.sqrt(2)
        
        return result
    
    def bm3d_denoising(self, image, sigma=None):
        """
        BM3D去噪算法的简化实现
        Block-matching and 3D filtering
        """
        print("执行BM3D去噪...")
        
        if sigma is None:
            # 估计噪声水平
            sigma = self._estimate_noise_level(image)
        
        # 参数设置
        patch_size = 8
        search_window = 39
        max_matched_blocks = 16
        
        # 第一步：基础去噪
        basic_estimate = self._bm3d_step1(image, sigma, patch_size, search_window, max_matched_blocks)
        
        # 第二步：最终去噪
        final_estimate = self._bm3d_step2(image, basic_estimate, sigma, patch_size, search_window, max_matched_blocks)
        
        return np.clip(final_estimate, 0, 1)
    
    def _bm3d_step1(self, noisy_image, sigma, patch_size, search_window, max_matched_blocks):
        """BM3D第一步：基础估计"""
        h, w = noisy_image.shape
        basic_estimate = np.zeros_like(noisy_image)
        weight_map = np.zeros_like(noisy_image)
        
        # 遍历图像块
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                # 提取参考块
                ref_patch = noisy_image[i:i+patch_size, j:j+patch_size]
                
                # 寻找相似块
                similar_patches, positions = self._find_similar_patches(
                    noisy_image, ref_patch, (i, j), search_window, max_matched_blocks
                )
                
                if len(similar_patches) > 1:
                    # 3D变换去噪
                    denoised_patches = self._collaborative_filtering_step1(
                        similar_patches, sigma
                    )
                    
                    # 聚合结果
                    for k, (pi, pj) in enumerate(positions):
                        if pi + patch_size <= h and pj + patch_size <= w:
                            basic_estimate[pi:pi+patch_size, pj:pj+patch_size] += denoised_patches[k]
                            weight_map[pi:pi+patch_size, pj:pj+patch_size] += 1
        
        # 归一化
        weight_map[weight_map == 0] = 1
        basic_estimate /= weight_map
        
        return basic_estimate
    
    def _bm3d_step2(self, noisy_image, basic_estimate, sigma, patch_size, search_window, max_matched_blocks):
        """BM3D第二步：最终估计"""
        h, w = noisy_image.shape
        final_estimate = np.zeros_like(noisy_image)
        weight_map = np.zeros_like(noisy_image)
        
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                # 使用基础估计寻找相似块
                ref_patch = basic_estimate[i:i+patch_size, j:j+patch_size]
                
                similar_patches_noisy, positions = self._find_similar_patches(
                    noisy_image, ref_patch, (i, j), search_window, max_matched_blocks
                )
                similar_patches_basic, _ = self._find_similar_patches(
                    basic_estimate, ref_patch, (i, j), search_window, max_matched_blocks
                )
                
                if len(similar_patches_noisy) > 1:
                    # 协同滤波
                    denoised_patches = self._collaborative_filtering_step2(
                        similar_patches_noisy, similar_patches_basic, sigma
                    )
                    
                    # 聚合结果
                    for k, (pi, pj) in enumerate(positions):
                        if pi + patch_size <= h and pj + patch_size <= w:
                            final_estimate[pi:pi+patch_size, pj:pj+patch_size] += denoised_patches[k]
                            weight_map[pi:pi+patch_size, pj:pj+patch_size] += 1
        
        weight_map[weight_map == 0] = 1
        final_estimate /= weight_map
        
        return final_estimate
    
    def _find_similar_patches(self, image, ref_patch, ref_pos, search_window, max_patches):
        """寻找相似图像块"""
        h, w = image.shape
        patch_size = ref_patch.shape[0]
        ref_i, ref_j = ref_pos
        
        # 搜索窗口边界
        start_i = max(0, ref_i - search_window // 2)
        end_i = min(h - patch_size + 1, ref_i + search_window // 2)
        start_j = max(0, ref_j - search_window // 2)
        end_j = min(w - patch_size + 1, ref_j + search_window // 2)
        
        distances = []
        positions = []
        
        # 计算距离
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                patch = image[i:i+patch_size, j:j+patch_size]
                distance = np.sum((patch - ref_patch) ** 2)
                distances.append(distance)
                positions.append((i, j))
        
        # 排序并选择最相似的块
        sorted_indices = np.argsort(distances)[:max_patches]
        
        similar_patches = []
        selected_positions = []
        
        for idx in sorted_indices:
            i, j = positions[idx]
            patch = image[i:i+patch_size, j:j+patch_size]
            similar_patches.append(patch)
            selected_positions.append((i, j))
        
        return similar_patches, selected_positions
    
    def _collaborative_filtering_step1(self, patches, sigma):
        """协同滤波第一步"""
        # 将块堆叠成3D数组
        patches_3d = np.stack(patches, axis=2)
        
        # 对每个位置应用2D DCT
        dct_coeffs = np.zeros_like(patches_3d)
        for i in range(patches_3d.shape[0]):
            for j in range(patches_3d.shape[1]):
                # 1D DCT沿第三维
                dct_coeffs[i, j, :] = self._dct_1d(patches_3d[i, j, :])
        
        # 硬阈值
        threshold = 2.7 * sigma
        mask = np.abs(dct_coeffs) > threshold
        dct_coeffs = dct_coeffs * mask
        
        # 逆DCT
        filtered_patches_3d = np.zeros_like(patches_3d)
        for i in range(patches_3d.shape[0]):
            for j in range(patches_3d.shape[1]):
                filtered_patches_3d[i, j, :] = self._idct_1d(dct_coeffs[i, j, :])
        
        # 分离回单独的块
        filtered_patches = []
        for k in range(filtered_patches_3d.shape[2]):
            filtered_patches.append(filtered_patches_3d[:, :, k])
        
        return filtered_patches
    
    def _collaborative_filtering_step2(self, noisy_patches, basic_patches, sigma):
        """协同滤波第二步"""
        # 使用Wiener滤波
        filtered_patches = []
        
        for noisy, basic in zip(noisy_patches, basic_patches):
            # 简化的Wiener滤波
            power_basic = np.mean(basic ** 2)
            wiener_gain = power_basic / (power_basic + sigma ** 2)
            filtered = wiener_gain * noisy + (1 - wiener_gain) * basic
            filtered_patches.append(filtered)
        
        return filtered_patches
    
    def _dct_1d(self, x):
        """1D DCT变换"""
        N = len(x)
        result = np.zeros(N)
        
        for k in range(N):
            sum_val = 0
            for n in range(N):
                sum_val += x[n] * np.cos(np.pi * k * (2*n + 1) / (2*N))
            
            if k == 0:
                result[k] = sum_val / np.sqrt(N)
            else:
                result[k] = sum_val * np.sqrt(2/N)
        
        return result
    
    def _idct_1d(self, X):
        """1D IDCT变换"""
        N = len(X)
        result = np.zeros(N)
        
        for n in range(N):
            sum_val = X[0] / np.sqrt(N)
            for k in range(1, N):
                sum_val += X[k] * np.sqrt(2/N) * np.cos(np.pi * k * (2*n + 1) / (2*N))
            result[n] = sum_val
        
        return result
    
    def _estimate_noise_level(self, image):
        """估计图像噪声水平"""
        # 使用Laplacian算子估计噪声
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = ndimage.convolve(image, laplacian)
        sigma = np.sqrt(np.pi/2) * np.mean(np.abs(convolved))
        return sigma
    
    def adaptive_wiener_filter(self, image, noise_variance=None):
        """
        自适应Wiener滤波器
        根据局部统计特性自适应调整滤波强度
        """
        print("执行自适应Wiener滤波...")
        
        if noise_variance is None:
            noise_variance = self._estimate_noise_level(image) ** 2
        
        # 局部均值和方差计算
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        local_mean = ndimage.convolve(image, kernel, mode='reflect')
        local_var = ndimage.convolve(image ** 2, kernel, mode='reflect') - local_mean ** 2
        
        # Wiener滤波
        wiener_gain = np.maximum(0, (local_var - noise_variance) / (local_var + 1e-10))
        filtered_image = local_mean + wiener_gain * (image - local_mean)
        
        return np.clip(filtered_image, 0, 1)
    
    def multiscale_detail_enhancement(self, image, num_scales=5):
        """
        多尺度细节增强
        使用拉普拉斯金字塔保持细节
        """
        print("执行多尺度细节增强...")
        
        # 构建高斯金字塔
        gaussian_pyramid = [image]
        current = image
        
        for i in range(num_scales - 1):
            # 高斯模糊和下采样
            blurred = ndimage.gaussian_filter(current, sigma=1.0)
            downsampled = blurred[::2, ::2]
            gaussian_pyramid.append(downsampled)
            current = downsampled
        
        # 构建拉普拉斯金字塔
        laplacian_pyramid = []
        
        for i in range(len(gaussian_pyramid) - 1):
            # 上采样
            upsampled = np.repeat(np.repeat(gaussian_pyramid[i+1], 2, axis=0), 2, axis=1)
            
            # 调整尺寸
            h, w = gaussian_pyramid[i].shape
            if upsampled.shape[0] > h:
                upsampled = upsampled[:h, :]
            if upsampled.shape[1] > w:
                upsampled = upsampled[:, :w]
            if upsampled.shape[0] < h:
                upsampled = np.pad(upsampled, ((0, h - upsampled.shape[0]), (0, 0)), mode='edge')
            if upsampled.shape[1] < w:
                upsampled = np.pad(upsampled, ((0, 0), (0, w - upsampled.shape[1])), mode='edge')
            
            # 计算拉普拉斯层
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)
        
        # 最后一层
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        # 增强每个尺度的细节
        enhanced_pyramid = []
        enhancement_factors = [1.5, 1.3, 1.2, 1.1, 1.0]  # 不同尺度的增强因子
        
        for i, laplacian in enumerate(laplacian_pyramid):
            if i < len(enhancement_factors):
                factor = enhancement_factors[i]
            else:
                factor = 1.0
            
            if i < len(laplacian_pyramid) - 1:  # 细节层
                enhanced = laplacian * factor
            else:  # 基础层
                enhanced = laplacian
            
            enhanced_pyramid.append(enhanced)
        
        # 重构图像
        reconstructed = enhanced_pyramid[-1]
        
        for i in range(len(enhanced_pyramid) - 2, -1, -1):
            # 上采样
            upsampled = np.repeat(np.repeat(reconstructed, 2, axis=0), 2, axis=1)
            
            # 调整尺寸
            h, w = enhanced_pyramid[i].shape
            if upsampled.shape[0] > h:
                upsampled = upsampled[:h, :]
            if upsampled.shape[1] > w:
                upsampled = upsampled[:, :w]
            if upsampled.shape[0] < h:
                upsampled = np.pad(upsampled, ((0, h - upsampled.shape[0]), (0, 0)), mode='edge')
            if upsampled.shape[1] < w:
                upsampled = np.pad(upsampled, ((0, 0), (0, w - upsampled.shape[1])), mode='edge')
            
            # 重构
            reconstructed = upsampled + enhanced_pyramid[i]
        
        return np.clip(reconstructed, 0, 1)
    
    def shock_filter(self, image, iterations=5, dt=0.1):
        """
        冲击滤波器 - 锐化边缘同时平滑区域
        """
        print("执行冲击滤波...")
        
        result = image.copy()
        
        for _ in range(iterations):
            # 计算梯度
            grad_x = ndimage.sobel(result, axis=1)
            grad_y = ndimage.sobel(result, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 计算二阶导数（拉普拉斯算子）
            laplacian = ndimage.laplace(result)
            
            # 冲击滤波更新
            # 在边缘处锐化，在平滑区域模糊
            sign_laplacian = np.sign(laplacian)
            update = -sign_laplacian * grad_magnitude
            
            result = result + dt * update
            result = np.clip(result, 0, 1)
        
        return result
    
    def coherence_enhancing_diffusion(self, image, iterations=20, alpha=0.001, sigma=1.0):
        """
        相干增强扩散 - 沿结构方向扩散，保持边缘
        """
        print("执行相干增强扩散...")
        
        result = image.copy()
        
        for _ in range(iterations):
            # 计算结构张量
            grad_x = ndimage.gaussian_filter(result, sigma, order=[0, 1])
            grad_y = ndimage.gaussian_filter(result, sigma, order=[1, 0])
            
            # 结构张量分量
            J11 = ndimage.gaussian_filter(grad_x * grad_x, sigma)
            J12 = ndimage.gaussian_filter(grad_x * grad_y, sigma)
            J22 = ndimage.gaussian_filter(grad_y * grad_y, sigma)
            
            # 特征值和特征向量
            trace = J11 + J22
            det = J11 * J22 - J12 * J12
            
            lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det + 1e-10))
            lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det + 1e-10))
            
            # 扩散张量
            c1 = np.exp(-(lambda1 - lambda2)**2 / (2 * alpha**2))
            c2 = np.ones_like(c1)
            
            # 更新图像
            div_x = ndimage.gaussian_filter(c1 * grad_x, 1, order=[0, 1]) + \
                   ndimage.gaussian_filter(c2 * grad_x, 1, order=[0, 1])
            div_y = ndimage.gaussian_filter(c1 * grad_y, 1, order=[1, 0]) + \
                   ndimage.gaussian_filter(c2 * grad_y, 1, order=[1, 0])
            
            result = result + 0.1 * (div_x + div_y)
            result = np.clip(result, 0, 1)
        
        return result
    
    def precision_enhancement_pipeline(self, 
                                     use_wavelet=True,
                                     use_bm3d=False,  # BM3D计算量大，默认关闭
                                     use_wiener=True,
                                     use_multiscale=True,
                                     use_shock=True,
                                     use_coherence=False):  # 相干扩散计算量大，默认关闭
        """
        高精度增强管道
        """
        if self.original_image is None:
            raise ValueError("请先加载图像")
        
        print("开始高精度增强处理...")
        result = self.original_image.copy()
        
        # 步骤1: 小波去噪
        if use_wavelet:
            result = self.wavelet_denoising(result)
            self.processing_steps.append(("小波去噪", result.copy()))
        
        # 步骤2: BM3D去噪（可选，计算量大）
        if use_bm3d:
            result = self.bm3d_denoising(result)
            self.processing_steps.append(("BM3D去噪", result.copy()))
        
        # 步骤3: 自适应Wiener滤波
        if use_wiener:
            result = self.adaptive_wiener_filter(result)
            self.processing_steps.append(("自适应Wiener滤波", result.copy()))
        
        # 步骤4: 多尺度细节增强
        if use_multiscale:
            result = self.multiscale_detail_enhancement(result)
            self.processing_steps.append(("多尺度细节增强", result.copy()))
        
        # 步骤5: 冲击滤波
        if use_shock:
            result = self.shock_filter(result)
            self.processing_steps.append(("冲击滤波", result.copy()))
        
        # 步骤6: 相干增强扩散（可选）
        if use_coherence:
            result = self.coherence_enhancing_diffusion(result)
            self.processing_steps.append(("相干增强扩散", result.copy()))
        
        # 最终微调
        result = self._final_precision_adjustment(result)
        self.processing_steps.append(("最终微调", result.copy()))
        
        self.enhanced_image = result
        print("高精度增强处理完成!")
        
        return result
    
    def _final_precision_adjustment(self, image):
        """最终精度调整"""
        # 轻微的对比度自适应
        local_mean = ndimage.uniform_filter(image, size=15)
        local_contrast = image - local_mean
        
        # 自适应增强
        enhanced_contrast = local_contrast * 1.1
        result = local_mean + enhanced_contrast
        
        return np.clip(result, 0, 1)
    
    def calculate_precision_metrics(self, original, enhanced):
        """计算精度相关指标"""
        metrics = {}
        
        # 高频内容保持指标
        def high_frequency_content(img):
            grad_x = ndimage.sobel(img, axis=1)
            grad_y = ndimage.sobel(img, axis=0)
            return np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        original_hfc = high_frequency_content(original)
        enhanced_hfc = high_frequency_content(enhanced)
        metrics['High_Freq_Preservation'] = enhanced_hfc / original_hfc
        
        # 边缘锐度
        def edge_sharpness(img):
            edges = ndimage.canny(img, sigma=1.0)
            grad_x = ndimage.sobel(img, axis=1)
            grad_y = ndimage.sobel(img, axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(grad_mag[edges])
        
        try:
            metrics['Edge_Sharpness_Ratio'] = edge_sharpness(enhanced) / edge_sharpness(original)
        except:
            metrics['Edge_Sharpness_Ratio'] = 1.0
        
        # 细节保持指标
        def detail_preservation(orig, enh):
            # 使用拉普拉斯算子检测细节
            laplacian_orig = ndimage.laplace(orig)
            laplacian_enh = ndimage.laplace(enh)
            
            correlation = np.corrcoef(laplacian_orig.flatten(), laplacian_enh.flatten())[0, 1]
            return correlation if not np.isnan(correlation) else 0
        
        metrics['Detail_Preservation'] = detail_preservation(original, enhanced)
        
        # 信噪比改善
        noise_orig = np.std(original - ndimage.gaussian_filter(original, 1))
        noise_enh = np.std(enhanced - ndimage.gaussian_filter(enhanced, 1))
        metrics['Noise_Reduction_Ratio'] = noise_orig / (noise_enh + 1e-10)
        
        return metrics
    
    def save_enhanced_image(self, output_path):
        """保存增强后的图像"""
        if self.enhanced_image is None:
            raise ValueError("没有增强后的图像可保存")
        
        # 转换为16位精度保存，减少量化损失
        image_16bit = (self.enhanced_image * 65535).astype(np.uint16)
        
        # 保存为16位TIFF格式以保持精度
        if output_path.lower().endswith('.tiff') or output_path.lower().endswith('.tif'):
            cv2.imwrite(output_path, image_16bit)
        else:
            # 如果是其他格式，转换为8位
            image_8bit = (self.enhanced_image * 255).astype(np.uint8)
            cv2.imwrite(output_path, image_8bit)
        
        print(f"高精度增强图像已保存到: {output_path}")


# GUI界面类
class AdvancedXRayEnhancerGUI:
    """高精度X光图像增强GUI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("高精度X光图像增强工具")
        self.root.geometry("1400x900")
        
        self.enhancer = AdvancedXRayEnhancer()
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
        
        # 高精度参数设置
        params_frame = ttk.LabelFrame(main_frame, text="高精度增强参数", padding="5")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=5, padx=(0, 5))
        
        self.setup_precision_parameters(params_frame)
        
        # 图像显示和结果
        self.setup_display_and_results(main_frame)
    
    def setup_precision_parameters(self, parent):
        """设置高精度参数"""
        row = 0
        
        # 算法选择
        ttk.Label(parent, text="高精度算法选择:").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        self.use_wavelet = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="小波去噪 (推荐)", variable=self.use_wavelet).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        self.use_bm3d = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="BM3D去噪 (高质量，慢)", variable=self.use_bm3d).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        self.use_wiener = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="自适应Wiener滤波", variable=self.use_wiener).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        self.use_multiscale = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="多尺度细节增强", variable=self.use_multiscale).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        self.use_shock = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="冲击滤波 (边缘锐化)", variable=self.use_shock).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        self.use_coherence = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="相干增强扩散 (慢)", variable=self.use_coherence).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        # 预设配置
        ttk.Label(parent, text="预设配置:").grid(row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
        preset_frame = ttk.Frame(parent)
        preset_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(preset_frame, text="快速模式", command=self.set_fast_preset).grid(row=0, column=0, padx=2)
        ttk.Button(preset_frame, text="平衡模式", command=self.set_balanced_preset).grid(row=0, column=1, padx=2)
        ttk.Button(preset_frame, text="最高质量", command=self.set_highest_quality_preset).grid(row=0, column=2, padx=2)
        
        row += 1
        
        # 开始处理按钮
        ttk.Button(parent, text="开始高精度增强", command=self.start_enhancement,
                  style="Accent.TButton").grid(row=row, column=0, columnspan=2, pady=10)
        
        # 进度条
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=row+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def setup_display_and_results(self, parent):
        """设置显示和结果区域"""
        # 创建笔记本控件
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 原始图像
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="原始图像")
        self.original_canvas = tk.Canvas(self.original_frame, bg='black')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 增强图像
        self.enhanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.enhanced_frame, text="增强图像")
        self.enhanced_canvas = tk.Canvas(self.enhanced_frame, bg='black')
        self.enhanced_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 精度指标显示
        metrics_frame = ttk.LabelFrame(parent, text="精度指标", padding="5")
        metrics_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.metrics_text = tk.Text(metrics_frame, height=6, width=80)
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)
    
    def set_fast_preset(self):
        """快速模式预设"""
        self.use_wavelet.set(True)
        self.use_bm3d.set(False)
        self.use_wiener.set(True)
        self.use_multiscale.set(False)
        self.use_shock.set(False)
        self.use_coherence.set(False)
    
    def set_balanced_preset(self):
        """平衡模式预设"""
        self.use_wavelet.set(True)
        self.use_bm3d.set(False)
        self.use_wiener.set(True)
        self.use_multiscale.set(True)
        self.use_shock.set(True)
        self.use_coherence.set(False)
    
    def set_highest_quality_preset(self):
        """最高质量预设"""
        self.use_wavelet.set(True)
        self.use_bm3d.set(True)
        self.use_wiener.set(True)
        self.use_multiscale.set(True)
        self.use_shock.set(True)
        self.use_coherence.set(True)
    
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
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = image_pil.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h) * 0.9
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image_resized = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image_resized)
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            canvas.image = photo
    
    def start_enhancement(self):
        """开始增强处理"""
        if self.enhancer.original_image is None:
            messagebox.showerror("错误", "请先选择图像文件")
            return
        
        self.progress.start()
        threading.Thread(target=self.perform_enhancement, daemon=True).start()
    
    def perform_enhancement(self):
        """执行增强处理"""
        try:
            start_time = time.time()
            
            # 获取参数
            params = {
                'use_wavelet': self.use_wavelet.get(),
                'use_bm3d': self.use_bm3d.get(),
                'use_wiener': self.use_wiener.get(),
                'use_multiscale': self.use_multiscale.get(),
                'use_shock': self.use_shock.get(),
                'use_coherence': self.use_coherence.get()
            }
            
            # 执行增强
            enhanced_image = self.enhancer.precision_enhancement_pipeline(**params)
            
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
        
        # 计算精度指标
        metrics = self.enhancer.calculate_precision_metrics(
            self.enhancer.original_image, enhanced_image
        )
        
        metrics_text = f"处理时间: {processing_time:.2f}秒\n"
        metrics_text += f"高频保持率: {metrics['High_Freq_Preservation']:.4f}\n"
        metrics_text += f"边缘锐度比: {metrics['Edge_Sharpness_Ratio']:.4f}\n"
        metrics_text += f"细节保持度: {metrics['Detail_Preservation']:.4f}\n"
        metrics_text += f"降噪比率: {metrics['Noise_Reduction_Ratio']:.2f}\n"
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, metrics_text)
        
        # 切换到增强图像标签页
        self.notebook.select(1)
        
        messagebox.showinfo("完成", f"高精度图像增强完成!\n处理时间: {processing_time:.2f}秒")
    
    def save_result(self):
        """保存增强结果"""
        if self.enhancer.enhanced_image is None:
            messagebox.showerror("错误", "没有增强结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存高精度增强图像",
            defaultextension=".tiff",
            filetypes=[
                ("TIFF图像 (16位)", "*.tiff"),
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.enhancer.save_enhanced_image(file_path)
                messagebox.showinfo("成功", f"高精度图像已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = AdvancedXRayEnhancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()