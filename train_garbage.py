import os


DATASET_YAML = 'dataset/dataset.yaml'


MODEL_OUT    = 'runs/detect'
RUN_NAME     = 'models/garbage_yolov8'


TRAIN_CONFIG = {
    'model':    'yolov8n.pt',
    'data':     DATASET_YAML,
    'epochs':   100,
    'imgsz':    640,
    'batch':    16,
    'device':   '0',             # '0'=GPU 0, '1'=GPU 1, 'cpu'=CPU
    'workers':  4,
    'project':  MODEL_OUT,
    'name':     RUN_NAME,
    'exist_ok': True,
    'patience': 20,
    'save':     True,
    'plots':    True,
    'cache':    'ram',
    'augment':  True,
    'degrees':  10,
    'flipud':   0.3,
    'fliplr':   0.5,
    'mosaic':   0.8,
    'mixup':    0.1,
    'hsv_h':    0.015,
    'hsv_s':    0.7,
    'hsv_v':    0.4,
    'lr0':      0.01,
    'lrf':      0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
}


def check_env():

    print("=" * 55)
    print("  YOLOv8 训练环境检测")
    print("=" * 55)


    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] ultralytics 未安装")
        print("运行: pip install ultralytics")
        return False


    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__}")
        if cuda_ok:
            print(f"[OK] CUDA 可用: GPU数量={torch.cuda.device_count()}")
            print(f"[OK] GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] 未检测到 GPU，将使用 CPU 训练（速度较慢）")
            TRAIN_CONFIG['device'] = 'cpu'
            TRAIN_CONFIG['batch'] = 8
    except ImportError:
        print("[WARNING] PyTorch 未安装，ultralytics 会自动处理")


    if not os.path.exists(DATASET_YAML):
        print(f"[ERROR] 数据集配置文件不存在: {DATASET_YAML}")
        print("请先运行: python convert_voc2yolo.py")
        return False


    try:
        train_imgs = os.listdir(os.path.join('dataset', 'yolo', 'train', 'images'))
        val_imgs = os.listdir(os.path.join('dataset', 'yolo', 'val', 'images'))
        print(f"[OK] 数据集: train={len(train_imgs)}, val={len(val_imgs)}")
    except Exception as e:
        print(f"[ERROR] 数据集检查失败: {e}")
        return False

    os.makedirs(MODEL_OUT, exist_ok=True)
    print(f"[OK] 模型输出目录: {MODEL_OUT}")
    return True


def train():

    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  开始训练 - 社区垃圾分类检测模型")
    print("=" * 55)

    print(f"训练配置:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")
    print()


    model = YOLO(TRAIN_CONFIG.pop('model'))


    results = model.train(**TRAIN_CONFIG)


    best_model = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')

    if os.path.exists(best_model):
        print(f"\n[完成] 训练完成！")
        print(f"[OK] 最佳模型: {best_model}")
        print(f"[提示] 下载此文件，重命名为 garbage_yolov8.pt 放到本地系统 app/models/ 目录")
    else:
        print(f"[警告] 最佳模型未找到")

    return results


def validate():

    from ultralytics import YOLO

    model_path = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        print("请先训练模型")
        return

    print("\n" + "=" * 55)
    print("  模型验证（验证集）")
    print("=" * 55)

    model = YOLO(model_path)
    metrics = model.val(data=DATASET_YAML, split='val')

    print(f"\n验证集结果:")
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")


def test():

    from ultralytics import YOLO

    model_path = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        print("请先训练模型")
        return

    print("\n" + "=" * 55)
    print("  模型测试（测试集 - 未参与训练）")
    print("=" * 55)

    model = YOLO(model_path)
    metrics = model.val(data=DATASET_YAML, split='test')

    print(f"\n测试集结果:")
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

    print("\n提示：")
    print("  - 如果测试集指标与验证集相近，说明模型泛化能力好")
    print("  - 如果测试集指标明显下降，可能存在过拟合")


def predict(image_path):

    from ultralytics import YOLO

    model_path = os.path.join(MODEL_OUT, RUN_NAME, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"[ERROR] 图片不存在: {image_path}")
        return

    print(f"[OK] 加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"[OK] 推理图片: {image_path}")
    results = model.predict(image_path, conf=0.25)


    for r in results:
        print(f"检测到 {len(r.boxes)} 个目标:")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"  {cls_name} (置信度: {conf:.3f})")


    output_dir = 'runs/predict'
    for r in results:
        r.save(output_dir)
    print(f"[OK] 结果图已保存到: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 垃圾检测训练（云 GPU 版）')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'predict'], default='train',
                        help='运行模式: train=训练, val=验证, test=测试, predict=推理测试')
    parser.add_argument('--image', type=str, help='推理模式下的图片路径')
    args = parser.parse_args()

    if not check_env():
        print("\n[ERROR] 环境检查失败，请修复后重试")
        exit(1)

    if args.mode == 'train':
        train()
    elif args.mode == 'val':
        validate()
    elif args.mode == 'test':
        test()
    elif args.mode == 'predict':
        if not args.image:
            print("[ERROR] 推理模式需要指定 --image 参数")
            exit(1)
        predict(args.image)
