import os
import torch
from ultralytics import YOLO
import argparse


def main(args):
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 加载预训练的YOLO11n-pose模型
    model = YOLO("yolo11n-pose.pt")

    # 训练模型
    print(f"开始训练，使用设备: {args.device}")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,  # 控制数据加载进程数
        batch=args.batch,  # 批处理大小
        patience=150,  # 早停耐心值
        save=True,  # 保存训练模型
        verbose=True,  # 显示详细日志
        project=args.project,  # 项目文件夹
        name=args.name,  # 实验名称
        exist_ok=True  # 允许覆盖现有实验
    )

    # 在验证集上评估模型性能
    print("开始模型评估...")
    metrics = model.val()
    print(f"验证结果: {metrics}")

    # 导出模型为ONNX格式用于部署
    print("导出模型为ONNX格式...")
    path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"模型已导出至: {path}")


if __name__ == "__main__":
    # Windows系统多进程支持
    if os.name == 'nt':
        import torch.multiprocessing as mp

        mp.freeze_support()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO11 Pose Training Script')
    parser.add_argument('--data', type=str, default="coco8-pose.yaml", help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='训练图像尺寸')
    parser.add_argument('--device', type=str, default="cuda", help='运行设备 (cpu, cuda, 0, 1等)')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--project', type=str, default="runs/pose", help='项目保存路径')
    parser.add_argument('--name', type=str, default="train", help='实验名称')

    args = parser.parse_args()

    # 检查设备是否可用
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行训练")
        args.device = "cpu"
        # CPU模式下建议减少进程数
        if args.workers > 2:
            args.workers = 2

    main(args)
