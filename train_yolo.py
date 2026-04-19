

import os
import subprocess
import sys


DATASET_YAML = r'D:/garbage_system/dataset/dataset.yaml'
MODEL_OUT    = r'D:/garbage_system/app/models'
RUN_NAME     = 'garbage_yolov8'


TRAIN_CONFIG = {
    'model':    'yolov8n.pt',
    'data':     DATASET_YAML,
    'epochs':   100,
    'imgsz':    640,
    'batch':    16,
    'device':   0,
    'workers':  4,
    'project':  MODEL_OUT,
    'name':     RUN_NAME,
    'exist_ok': True,
    'patience': 20,
    'save':     True,
    'plots':    True,
    'cache':    False,
    'augment':  True,
    'degrees':  10,
    'flipud':   0.3,
    'fliplr':   0.5,
    'mosaic':   0.8,
    'mixup':    0.1,
}


def check_env():

    print("=" * 55)
    print("  YOLOv8 训练环境检测")
    print("=" * 55)


    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] ultralytics 未安装，正在安装...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'])
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__} 安装完成")


    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__}, CUDA={cuda_ok}")
        if not cuda_ok:
            print("[提示] 未检测到 GPU，将使用 CPU 训练（速度较慢，建议改 device='cpu'）")
            TRAIN_CONFIG['device'] = 'cpu'
            TRAIN_CONFIG['batch'] = 8
    except ImportError:
        print("[提示] PyTorch 未安装，ultralytics 会自动处理")


    if not os.path.exists(DATASET_YAML):
        print(f"[ERROR] 数据集不存在: {DATASET_YAML}")
        print("请先运行: py -3 D:\\garbage_system\\convert_voc2yolo.py")
        sys.exit(1)
    else:
        train_count = len(os.listdir(os.path.join(os.path.dirname(DATASET_YAML), 'train', 'images')))
        val_count = len(os.listdir(os.path.join(os.path.dirname(DATASET_YAML), 'val', 'images')))
        print(f"[OK] 数据集: train={train_count}, val={val_count}")

    os.makedirs(MODEL_OUT, exist_ok=True)
    print(f"[OK] 模型输出目录: {MODEL_OUT}")
    return True


def train():

    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  开始训练 - 社区垃圾分类检测模型")
    print("=" * 55)

    model = YOLO(TRAIN_CONFIG.pop('model'))

    print(f"配置参数: {TRAIN_CONFIG}")
    print()

    results = model.train(**TRAIN_CONFIG)


    best_model = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    system_model = os.path.join(MODEL_OUT, 'garbage_yolov8.pt')

    if os.path.exists(best_model):
        import shutil
        shutil.copy2(best_model, system_model)
        print(f"\n[完成] 最佳模型已复制至系统路径: {system_model}")
        print(f"[提示] 重启系统后将自动加载真实模型")
    else:
        print(f"[警告] 最佳模型未找到: {best_model}")

    return results


def validate():

    from ultralytics import YOLO

    model_path = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print("[ERROR] 模型不存在，请先训练")
        return

    print("\n" + "=" * 55)
    print("  模型验证")
    print("=" * 55)
    model = YOLO(model_path)
    metrics = model.val(data=DATASET_YAML)
    print(f"\nmAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 垃圾检测训练')
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    args = parser.parse_args()

    check_env()
    if args.mode == 'train':
        train()
    else:
        validate()
