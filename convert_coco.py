#!/usr/bin/env python3


import json
import os
import shutil
import argparse
from pathlib import Path


def convert_bbox_coco_to_yolo(bbox, img_width, img_height):

    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def create_category_mapping(categories):

    return {cat['id']: cat['name'] for cat in categories}


def convert_single_category(json_path, image_folder, output_base, target_category=None, image_mapping=None):


    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_name_to_id = {cat['name']: i for i, cat in enumerate(data['categories'])}

    print(f"[{target_category or '全部'}] 找到 {len(data['categories'])} 个类别: {list(cat_id_to_name.values())}")
    print(f"[{target_category or '全部'}] 共 {len(data['images'])} 张图片")


    output_images_dir = os.path.join(output_base, 'images')
    output_labels_dir = os.path.join(output_base, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)


    image_map = {img['id']: img for img in data['images']}

    processed_count = 0

    for img_info in data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']


        if image_mapping and img_filename in image_mapping:
            actual_filename = image_mapping[img_filename]
        else:
            actual_filename = img_filename


        img_annots = [ann for ann in data['annotations'] if ann['image_id'] == img_id]


        if target_category is not None:
            img_annots = [
                ann for ann in img_annots
                if cat_id_to_name[ann['category_id']] == target_category
            ]


        if not img_annots:
            continue


        src_img_path = os.path.join(image_folder, actual_filename)
        dst_img_path = os.path.join(output_images_dir, img_filename)

        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"  警告: 图片不存在 {src_img_path}")
            continue


        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)

        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in img_annots:
                cat_id = ann['category_id']
                cat_name = cat_id_to_name[cat_id]


                if target_category is not None:
                    yolo_class_id = 0
                else:
                    yolo_class_id = list(cat_name_to_id.keys()).index(cat_name)


                bbox = ann['bbox']
                x_center, y_center, w_norm, h_norm = convert_bbox_coco_to_yolo(bbox, img_width, img_height)


                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        processed_count += 1


    yaml_name = os.path.basename(output_base)
    yaml_path = os.path.join(output_base, f"{yaml_name}.yaml")

    with open(yaml_path, 'w', encoding='utf-8') as f:
        if target_category is not None:

            f.write(f"# YOLOv8 单类别数据集 - {target_category}\n")
            f.write(f"# 由 convert_coco.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['{target_category}']\n")
        else:

            f.write(f"# YOLOv8 多类别数据集\n")
            f.write(f"# 由 convert_coco.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: {len(data['categories'])}\n")
            f.write(f"names: {[cat['name'] for cat in data['categories']]}\n")

    print(f"[{target_category or '全部'}] 完成! 处理 {processed_count} 张图片")
    print(f"[{target_category or '全部'}] 数据集保存到: {output_base}")
    print(f"[{target_category or '全部'}] 配置文件: {yaml_path}")

    return processed_count


def main():
    parser = argparse.ArgumentParser(description='COCO格式转YOLO格式工具')
    parser.add_argument('--json', '-j', required=True, help='COCO JSON文件路径')
    parser.add_argument('--images', '-i', required=True, help='图片文件夹路径')
    parser.add_argument('--output', '-o', required=True, help='输出数据集路径')
    parser.add_argument('--category', '-c', help='只提取指定类别（如fire或smoke）')
    parser.add_argument('--split', '-s', nargs='+', help='分离多个类别为独立数据集（如 fire smoke）')
    parser.add_argument('--mapping', '-m', help='图片文件名映射JSON文件')

    args = parser.parse_args()


    image_mapping = None
    if args.mapping and os.path.exists(args.mapping):
        with open(args.mapping, 'r', encoding='utf-8') as f:
            image_mapping = json.load(f)


    if args.split:
        print("=" * 50)
        print("分离模式：创建多个独立数据集")
        print("=" * 50)

        for category in args.split:
            output_dir = os.path.join(args.output, f"dataset_{category}")
            print(f"\n处理类别: {category}")
            convert_single_category(
                args.json,
                args.images,
                output_dir,
                target_category=category,
                image_mapping=image_mapping
            )
    else:

        print("=" * 50)
        print("转换模式：创建单个数据集")
        print("=" * 50)

        convert_single_category(
            args.json,
            args.images,
            args.output,
            target_category=args.category,
            image_mapping=image_mapping
        )

    print("\n" + "=" * 50)
    print("全部完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
