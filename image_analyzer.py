#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像文件分析工具
功能：
1. 检测文件命名规则是否与样本一致
2. 检测图像属性（通道数、分辨率）是否与样本一致
3. 可选：检测灰度和色彩分布差异
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from datetime import datetime


class ImageAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("图像文件分析工具")
        self.root.geometry("1200x800")
        
        # 变量
        self.sample_file = tk.StringVar()
        self.target_folder = tk.StringVar()
        self.enable_distribution_check = tk.BooleanVar(value=False)
        
        # 分析结果
        self.sample_pattern = None
        self.sample_properties = None
        self.analysis_results = {}
        
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
        
        # 样本文件选择
        ttk.Label(main_frame, text="样本图像文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.sample_file, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_sample_file).grid(row=0, column=2, padx=5)
        
        # 目标文件夹选择
        ttk.Label(main_frame, text="目标文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.target_folder, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_target_folder).grid(row=1, column=2, padx=5)
        
        # 选项
        options_frame = ttk.LabelFrame(main_frame, text="检测选项", padding="5")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Checkbutton(options_frame, text="检测灰度和色彩分布差异", 
                       variable=self.enable_distribution_check).grid(row=0, column=0, sticky=tk.W)
        
        # 分析按钮
        ttk.Button(main_frame, text="开始分析", command=self.start_analysis, 
                  style="Accent.TButton").grid(row=3, column=0, columnspan=3, pady=10)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 结果显示
        self.setup_results_ui(main_frame)
    
    def setup_results_ui(self, parent):
        """设置结果显示界面"""
        # 创建笔记本控件用于分页显示结果
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 配置行权重
        parent.rowconfigure(5, weight=1)
        
        # 概要页面
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="分析概要")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD, height=15)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 详细结果页面
        self.details_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.details_frame, text="详细结果")
        
        # 创建树形视图显示详细结果
        columns = ("文件名", "命名规则", "分辨率", "通道数", "分布差异")
        self.results_tree = ttk.Treeview(self.details_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(self.details_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 分布图页面（如果启用）
        self.distribution_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.distribution_frame, text="分布对比图")
    
    def select_sample_file(self):
        """选择样本文件"""
        file_path = filedialog.askopenfilename(
            title="选择样本图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.sample_file.set(file_path)
    
    def select_target_folder(self):
        """选择目标文件夹"""
        folder_path = filedialog.askdirectory(title="选择目标文件夹")
        if folder_path:
            self.target_folder.set(folder_path)
    
    def start_analysis(self):
        """开始分析"""
        if not self.sample_file.get() or not self.target_folder.get():
            messagebox.showerror("错误", "请选择样本文件和目标文件夹")
            return
        
        if not os.path.exists(self.sample_file.get()):
            messagebox.showerror("错误", "样本文件不存在")
            return
        
        if not os.path.exists(self.target_folder.get()):
            messagebox.showerror("错误", "目标文件夹不存在")
            return
        
        # 在新线程中执行分析
        self.progress.start()
        threading.Thread(target=self.perform_analysis, daemon=True).start()
    
    def perform_analysis(self):
        """执行分析"""
        try:
            # 分析样本文件
            self.analyze_sample_file()
            
            # 获取目标文件夹中的图像文件
            image_files = self.get_image_files()
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showwarning("警告", "目标文件夹中没有找到图像文件"))
                return
            
            # 分析每个文件
            self.analysis_results = {}
            for i, file_path in enumerate(image_files):
                result = self.analyze_file(file_path)
                self.analysis_results[file_path] = result
                
                # 更新进度（这里简化处理）
                progress = (i + 1) / len(image_files) * 100
            
            # 在主线程中更新UI
            self.root.after(0, self.update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中出现错误: {str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def analyze_sample_file(self):
        """分析样本文件"""
        sample_path = self.sample_file.get()
        
        # 提取文件名模式
        filename = os.path.splitext(os.path.basename(sample_path))[0]
        self.sample_pattern = self.extract_filename_pattern(filename)
        
        # 获取图像属性
        self.sample_properties = self.get_image_properties(sample_path)
    
    def extract_filename_pattern(self, filename):
        """提取文件名模式"""
        # 尝试识别常见的命名模式
        patterns = {
            'barcode_timestamp': r'^([A-Za-z0-9]+)_(\d{8,14})$',  # 条码_时间戳
            'date_sequence': r'^(\d{4}-?\d{2}-?\d{2})_?(\d+)$',  # 日期_序号
            'number_prefix': r'^(\d+)_([A-Za-z0-9]+)$',  # 数字_前缀
            'multi_underscore': r'^([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)$',  # 多段下划线分隔
            'prefix_number': r'^([A-Za-z]+)(\d+)$',  # 前缀+数字 (放在最后，避免过度匹配)
        }
        
        for pattern_name, pattern in patterns.items():
            match = re.match(pattern, filename)
            if match:
                return {'name': pattern_name, 'regex': pattern, 'template': filename, 'groups': match.groups()}
        
        # 如果没有匹配到预定义模式，创建智能模式
        # 分析文件名结构，识别数字、字母、特殊字符的模式
        smart_pattern = self.create_smart_pattern(filename)
        return {'name': 'smart_pattern', 'regex': smart_pattern, 'template': filename}
    
    def create_smart_pattern(self, filename):
        """创建智能模式匹配"""
        # 将文件名分解为组件
        pattern = ""
        i = 0
        while i < len(filename):
            char = filename[i]
            if char.isdigit():
                # 连续数字
                pattern += r'\d+'
                while i < len(filename) and filename[i].isdigit():
                    i += 1
            elif char.isalpha():
                # 连续字母
                pattern += r'[A-Za-z]+'
                while i < len(filename) and filename[i].isalpha():
                    i += 1
            elif char.isalnum():
                # 字母数字混合
                pattern += r'[A-Za-z0-9]+'
                while i < len(filename) and filename[i].isalnum():
                    i += 1
            else:
                # 特殊字符，直接转义
                pattern += re.escape(char)
                i += 1
        
        return f"^{pattern}$"
    
    def get_image_properties(self, file_path):
        """获取图像属性"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {'error': '文件不存在'}
            
            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                return {'error': '文件为空'}
            
            # 使用OpenCV读取图像
            img = cv2.imread(file_path)
            if img is None:
                # 尝试使用PIL
                try:
                    with Image.open(file_path) as pil_img:
                        img_array = np.array(pil_img)
                        if len(img_array.shape) == 2:
                            channels = 1
                        elif len(img_array.shape) == 3:
                            channels = img_array.shape[2]
                        else:
                            return {'error': '不支持的图像格式'}
                        height, width = img_array.shape[:2]
                except Exception as pil_error:
                    return {'error': f'无法读取图像文件: {str(pil_error)}'}
            else:
                if len(img.shape) == 3:
                    height, width, channels = img.shape
                else:
                    height, width = img.shape
                    channels = 1
            
            properties = {
                'width': width,
                'height': height,
                'channels': channels,
                'resolution': f"{width}x{height}"
            }
            
            # 如果启用分布检测，计算直方图
            if self.enable_distribution_check.get():
                histogram = self.calculate_histogram(file_path)
                if histogram is not None:
                    properties['histogram'] = histogram
            
            return properties
            
        except MemoryError:
            return {'error': '图像文件过大，内存不足'}
        except Exception as e:
            return {'error': f'读取图像时出错: {str(e)}'}
    
    def calculate_histogram(self, file_path):
        """计算图像直方图"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
            
            # 转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 计算每个通道的直方图
            histograms = []
            colors = ['red', 'green', 'blue']
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                histograms.append(hist.flatten())
            
            return histograms
            
        except Exception as e:
            return None
    
    def get_image_files(self):
        """获取目标文件夹中的图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        folder_path = self.target_folder.get()
        for filename in os.listdir(folder_path):
            if os.path.splitext(filename.lower())[1] in image_extensions:
                image_files.append(os.path.join(folder_path, filename))
        
        return image_files
    
    def analyze_file(self, file_path):
        """分析单个文件"""
        filename = os.path.splitext(os.path.basename(file_path))[0]
        result = {
            'filename': filename,
            'full_path': file_path,
            'naming_match': False,
            'properties_match': True,
            'distribution_similarity': None
        }
        
        # 检查命名规则
        if self.sample_pattern:
            if re.match(self.sample_pattern['regex'], filename):
                result['naming_match'] = True
        
        # 检查图像属性
        properties = self.get_image_properties(file_path)
        result['properties'] = properties
        
        if 'error' not in properties and self.sample_properties and 'error' not in self.sample_properties:
            if (properties['width'] != self.sample_properties['width'] or
                properties['height'] != self.sample_properties['height'] or
                properties['channels'] != self.sample_properties['channels']):
                result['properties_match'] = False
        
        # 检查分布相似性（如果启用）
        if (self.enable_distribution_check.get() and 
            'histogram' in properties and properties['histogram'] and
            'histogram' in self.sample_properties and self.sample_properties['histogram']):
            similarity = self.calculate_histogram_similarity(
                properties['histogram'], self.sample_properties['histogram']
            )
            result['distribution_similarity'] = similarity
        
        return result
    
    def calculate_histogram_similarity(self, hist1, hist2):
        """计算直方图相似性"""
        try:
            similarities = []
            for i in range(len(hist1)):
                # 使用相关系数计算相似性
                correlation = np.corrcoef(hist1[i], hist2[i])[0, 1]
                if not np.isnan(correlation):
                    similarities.append(correlation)
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def update_results(self):
        """更新结果显示"""
        # 清空之前的结果
        self.summary_text.delete(1.0, tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # 统计结果
        total_files = len(self.analysis_results)
        naming_mismatches = []
        property_mismatches = []
        distribution_issues = []
        
        # 更新详细结果树形视图
        for file_path, result in self.analysis_results.items():
            filename = result['filename']
            naming_status = "✓" if result['naming_match'] else "✗"
            properties_status = "✓" if result['properties_match'] else "✗"
            
            if not result['naming_match']:
                naming_mismatches.append(filename)
            if not result['properties_match']:
                property_mismatches.append(filename)
            
            # 分布相似性状态
            if result['distribution_similarity'] is not None:
                if result['distribution_similarity'] < 0.8:  # 相似性阈值
                    distribution_status = f"✗ ({result['distribution_similarity']:.2f})"
                    distribution_issues.append(filename)
                else:
                    distribution_status = f"✓ ({result['distribution_similarity']:.2f})"
            else:
                distribution_status = "N/A"
            
            # 属性信息
            props = result['properties']
            if 'error' not in props:
                resolution = props['resolution']
                channels = str(props['channels'])
            else:
                resolution = "错误"
                channels = "错误"
            
            self.results_tree.insert("", tk.END, values=(
                filename, naming_status, resolution, channels, distribution_status
            ))
        
        # 更新概要
        summary = f"分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        summary += f"样本文件: {os.path.basename(self.sample_file.get())}\n"
        summary += f"样本命名模式: {self.sample_pattern['name'] if self.sample_pattern else 'N/A'}\n"
        if self.sample_properties and 'error' not in self.sample_properties:
            summary += f"样本属性: {self.sample_properties['resolution']}, {self.sample_properties['channels']}通道\n\n"
        
        summary += f"总文件数: {total_files}\n"
        summary += f"命名规则不匹配: {len(naming_mismatches)} 个文件\n"
        summary += f"属性不匹配: {len(property_mismatches)} 个文件\n"
        
        if self.enable_distribution_check.get():
            summary += f"分布差异较大: {len(distribution_issues)} 个文件\n"
        
        summary += "\n" + "="*50 + "\n\n"
        
        if naming_mismatches:
            summary += "命名规则不匹配的文件:\n"
            for filename in naming_mismatches:
                summary += f"  • {filename}\n"
            summary += "\n"
        
        if property_mismatches:
            summary += "属性不匹配的文件:\n"
            for filename in property_mismatches:
                summary += f"  • {filename}\n"
            summary += "\n"
        
        if distribution_issues:
            summary += "分布差异较大的文件:\n"
            for filename in distribution_issues:
                summary += f"  • {filename}\n"
            summary += "\n"
        
        self.summary_text.insert(tk.END, summary)
        
        # 如果启用了分布检测，生成对比图
        if self.enable_distribution_check.get() and self.sample_properties and 'histogram' in self.sample_properties:
            self.create_distribution_plot()
    
    def create_distribution_plot(self):
        """创建分布对比图"""
        try:
            # 清空之前的图表
            for widget in self.distribution_frame.winfo_children():
                widget.destroy()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('色彩分布对比图', fontsize=16)
            
            # 样本图像直方图
            sample_hist = self.sample_properties['histogram']
            colors = ['red', 'green', 'blue']
            
            for i, (color, hist) in enumerate(zip(colors, sample_hist)):
                axes[0, 0].plot(hist, color=color, alpha=0.7, label=f'{color.upper()} 通道')
            axes[0, 0].set_title('样本图像直方图')
            axes[0, 0].legend()
            axes[0, 0].set_xlabel('像素值')
            axes[0, 0].set_ylabel('频次')
            
            # 选择一个有代表性的对比文件
            comparison_file = None
            min_similarity = float('inf')
            
            for file_path, result in self.analysis_results.items():
                if (result['distribution_similarity'] is not None and 
                    result['distribution_similarity'] < min_similarity):
                    min_similarity = result['distribution_similarity']
                    comparison_file = file_path
            
            if comparison_file and comparison_file in self.analysis_results:
                comp_result = self.analysis_results[comparison_file]
                if 'histogram' in comp_result['properties']:
                    comp_hist = comp_result['properties']['histogram']
                    
                    for i, (color, hist) in enumerate(zip(colors, comp_hist)):
                        axes[0, 1].plot(hist, color=color, alpha=0.7, label=f'{color.upper()} 通道')
                    axes[0, 1].set_title(f'差异最大文件: {comp_result["filename"]}')
                    axes[0, 1].legend()
                    axes[0, 1].set_xlabel('像素值')
                    axes[0, 1].set_ylabel('频次')
                    
                    # 相似性分数分布
                    similarities = []
                    for result in self.analysis_results.values():
                        if result['distribution_similarity'] is not None:
                            similarities.append(result['distribution_similarity'])
                    
                    if similarities:
                        axes[1, 0].hist(similarities, bins=20, alpha=0.7, color='skyblue')
                        axes[1, 0].set_title('相似性分数分布')
                        axes[1, 0].set_xlabel('相似性分数')
                        axes[1, 0].set_ylabel('文件数量')
                        axes[1, 0].axvline(x=0.8, color='red', linestyle='--', label='阈值 (0.8)')
                        axes[1, 0].legend()
            
            # 隐藏空的子图
            axes[1, 1].set_visible(False)
            
            plt.tight_layout()
            
            # 将图表嵌入到tkinter中
            canvas = FigureCanvasTkAgg(fig, self.distribution_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            error_label = ttk.Label(self.distribution_frame, text=f"生成分布图时出错: {str(e)}")
            error_label.pack(pady=20)


def main():
    """主函数"""
    root = tk.Tk()
    app = ImageAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()