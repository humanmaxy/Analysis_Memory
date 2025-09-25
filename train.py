import os
import torch
from rfdetr import RFDETRBase
import os
os.environ["PYTHONUTF8"] = "1"
def main():
    # 打印环境信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
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
    # 设置环境变量
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 有助于更精确的错误定位
    # os.environ['PYTHONWARNINGS'] = "ignore"
    
    # 设置PyTorch的并行计算线程数
    # torch.set_num_threads(1)
    
    # 运行主函数
    main()
