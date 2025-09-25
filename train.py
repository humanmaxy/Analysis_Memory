import os
import torch

# CRITICAL: Apply patches BEFORE importing rfdetr
from patch_matcher import apply_comprehensive_patches
apply_comprehensive_patches()

from rfdetr import RFDETRBase
from debug_utils import setup_cuda_debugging, check_cuda_memory

os.environ["PYTHONUTF8"] = "1"
def main():
    # 设置调试环境
    setup_cuda_debugging()
    
    # 打印环境信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        check_cuda_memory()
    
    # 模型配置
    MODEL_PATH = "F:/code/rf-detr/rf-detr-base.pth"
    
    # 创建模型实例
    # model = RFDETRBase(
    #     config_path=None,
    #     checkpoint_path=MODEL_PATH,
        
    # )
    model = RFDETRBase()


    # model.train()
    model.train(
            dataset_dir="F:/res/data",
            epochs=10,
            batch_size=4,  # 减少批处理大小
            grad_accum_steps=2, 
             )
    

if __name__ == '__main__':
    # 设置PyTorch的并行计算线程数
    torch.set_num_threads(4)  # 设置合理的线程数
    
    # 运行主函数
    main()
