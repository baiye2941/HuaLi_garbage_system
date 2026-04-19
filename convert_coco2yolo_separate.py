#!/usr/bin/env python3


import json
import os
import glob
import shutil
from pathlib import Path

def convert_coco_to_yolo(json_path, image_folder, output_base, target_category=None):


    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_name_to_id = {cat['name']: i for i, cat in enumerate(data['categories'])}
    
    print(f"找到 {len(data['categories'])} 个类别: {list(cat_id_to_name.values())}")
    print(f"共 {len(data['images'])} 张图片")
    

    output_images_dir = os.path.join(output_base, 'images')
    output_labels_dir = os.path.join(output_base, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    

    config_path = os.path.join(output_base, f"{os.path.basename(output_base)}.yaml")
    

    image_map = {img['id']: img for img in data['images']}
    
    for img_info in data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        

        img_annots = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        

        if target_category is not None:
            img_annots = [
                ann for ann in img_annots 
                if cat_id_to_name[ann['category_id']] == target_category
            ]
        

        if not img_annots:
            continue
        

        src_img_path = os.path.join(image_folder, img_filename)
        dst_img_path = os.path.join(output_images_dir, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"警告: 图片不存在 {src_img_path}")
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
                x_min, y_min, w, h = bbox
                

                x_center = (x_min + w / 2) / img_width
                y_center = (y_min + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                

                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    

    with open(config_path, 'w', encoding='utf-8') as f:
        if target_category is not None:

            f.write(f"# YOLOv8 单类别数据集 - {target_category}\n")
            f.write(f"# 由 convert_coco2yolo_separate.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['{target_category}']\n")
        else:

            f.write(f"# YOLOv8 多类别数据集\n")
            f.write(f"# 由 convert_coco2yolo_separate.py 自动生成\n\n")
            f.write(f"path: {os.path.abspath(output_base)}\n")
            f.write(f"train: images\n\n")
            f.write(f"nc: {len(data['categories'])}\n")
            f.write(f"names: {[cat['name'] for cat in data['categories']]}\n")
    
    print(f"完成! 数据集保存到: {output_base}")
    print(f"配置文件: {config_path}")
    print(f"图片数: {len(os.listdir(output_images_dir))}")
    print(f"标注数: {len(os.listdir(output_labels_dir))}")

def main():

    json_files = [
        "C:/Users/24039/Downloads/111.json",
        "C:/Users/24039/Downloads/222.json",
        "C:/Users/24039/Downloads/333.json",
        "C:/Users/24039/Downloads/444.json",
        "C:/Users/24039/Downloads/555.json"
    ]
    

    image_folder = "C:/Users/24039/Downloads"
    

    print("="*50)
    print("正在创建火数据集...")
    print("="*50)
    fire_output = "d:/garbage_system/dataset_fire_new"
    

    combined_data = {
        "info": {"description": "fire-only dataset"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"}]
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        

        for img in data['images']:

            img_annots = [ann for ann in data['annotations'] 
                         if ann['image_id'] == img['id'] 
                         and data['categories'][ann['category_id']-1]['name'] == 'fire']
            
            if img_annots:

                new_img_id = len(combined_data['images']) + 1
                img_copy = img.copy()
                img_copy['id'] = new_img_id
                combined_data['images'].append(img_copy)
                

                for ann in img_annots:
                    new_ann = ann.copy()
                    new_ann['id'] = len(combined_data['annotations']) + 1
                    new_ann['image_id'] = new_img_id
                    new_ann['category_id'] = 1
                    combined_data['annotations'].append(new_ann)
    

    temp_json = "d:/garbage_system/temp_fire.json"
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    

    convert_coco_to_yolo(temp_json, image_folder, fire_output, target_category='fire')
    

    print("\n" + "="*50)
    print("正在创建烟雾数据集...")
    print("="*50)
    smoke_output = "d:/garbage_system/dataset_smoke"
    
    combined_data = {
        "info": {"description": "smoke-only dataset"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "smoke"}]
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for img in data['images']:
            img_annots = [ann for ann in data['annotations'] 
                         if ann['image_id'] == img['id'] 
                         and data['categories'][ann['category_id']-1]['name'] == 'smoke']
            
            if img_annots:
                new_img_id = len(combined_data['images']) + 1
                img_copy = img.copy()
                img_copy['id'] = new_img_id
                combined_data['images'].append(img_copy)
                
                for ann in img_annots:
                    new_ann = ann.copy()
                    new_ann['id'] = len(combined_data['annotations']) + 1
                    new_ann['image_id'] = new_img_id
                    new_ann['category_id'] = 1
                    combined_data['annotations'].append(new_ann)
    
    temp_json = "d:/garbage_system/temp_smoke.json"
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    convert_coco_to_yolo(temp_json, image_folder, smoke_output, target_category='smoke')
    

    os.remove("d:/garbage_system/temp_fire.json")
    os.remove("d:/garbage_system/temp_smoke.json")
    
    print("\n" + "="*50)
    print("完成！")
    print("- 火数据集: d:/garbage_system/dataset_fire_new")
    print("- 烟雾数据集: d:/garbage_system/dataset_smoke")
    print("="*50)

if __name__ == "__main__":
    main()