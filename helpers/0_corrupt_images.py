import os
import glob
import argparse
import filetype
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from enum import Enum
import warnings

# Suppress the pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning)

from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
from imagecorruptions import corruption_dict

class OutputType(Enum):
    SUBDIRS = 'subdirs'
    FILENAME = 'filename'

    def __str__(self) -> str:
        return self.value

class AnnotationFormat(Enum):
    COCO = 'coco'
    VOC = 'voc'
    NONE = 'none'

# --- UPDATED: VOC Category Manager with COCO Alignment ---
class VocCategoryManager:
    def __init__(self):
        # HARDCODED MAPPING: VOC Class Name -> COCO Category ID
        # This ensures that 'person' is always 1, 'car' is always 3, etc.
        self.voc_to_coco_map = {
            'person': 1,
            'bicycle': 2,
            'car': 3,
            'motorbike': 4,       # COCO: motorcycle
            'aeroplane': 5,       # COCO: airplane
            'bus': 6,
            'train': 7,
            'truck': 8,
            'boat': 9,
            'bird': 16,
            'cat': 17,
            'dog': 18,
            'horse': 19,
            'sheep': 20,
            'cow': 21,
            'bottle': 44,
            'chair': 62,
            'sofa': 63,           # COCO: couch
            'pottedplant': 64,    # COCO: potted plant
            'diningtable': 67,    # COCO: dining table
            'tvmonitor': 72       # COCO: tv
        }
        
        # We will track which categories are actually used to build the final categories list
        self.used_categories = {}

    def get_id(self, name):
        name_lower = name.lower()
        
        if name_lower in self.voc_to_coco_map:
            cat_id = self.voc_to_coco_map[name_lower]
            
            # Record usage for the final JSON header
            if cat_id not in self.used_categories:
                self.used_categories[cat_id] = name # Use VOC name or map to COCO name if preferred
                
            return cat_id
        else:
            print(f"Warning: Found VOC class '{name}' which is not in the COCO mapping. Skipping.")
            return None

    def get_coco_categories(self):
        # Returns the list of categories formatted for COCO JSON
        categories = []
        # Sort by ID to look tidy
        for cat_id in sorted(self.used_categories.keys()):
            categories.append({
                'id': cat_id,
                'name': self.used_categories[cat_id],
                'supercategory': 'none'
            })
        return categories

def get_yolo_compatible_size(width, height, multiple=32, min_size=416, max_size=1024):
    """Calculate YOLO-compatible size that maintains aspect ratio"""
    scale_min = min_size / min(width, height)
    scale_max = max_size / max(width, height)
    scale = min(scale_min, scale_max, 1.0)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    new_width = (new_width // multiple) * multiple
    new_height = (new_height // multiple) * multiple
    
    new_width = max(new_width, min_size)
    new_height = max(new_height, min_size)
    
    return new_width, new_height

def center_crop_image(image, target_width, target_height):
    """Center crop image to target dimensions"""
    width, height = image.size
    
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    
    if (right - left) < target_width or (bottom - top) < target_height:
        crop_width = min(target_width, width)
        crop_height = min(target_height, height)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
    
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)

def adapt_bbox(bbox, crop_coords, fmt='xywh'):
    """
    Adapt a single bbox to the crop. 
    fmt: 'xywh' (COCO standard) or 'xyxy' (VOC standard)
    INTERNAL CALCULATION IS ALWAYS XYXY for intersection logic.
    Returns: [x, y, w, h], new_area
    """
    left, top, right, bottom = crop_coords
    
    # Standardize input to x1, y1, x2, y2 for calculation
    if fmt == 'xywh':
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    else: # xyxy
        x1, y1, x2, y2 = bbox
        # Calculate original w/h just for area check
        w = x2 - x1
        h = y2 - y1

    # Calculate intersection with crop
    ix1 = max(x1, left)
    iy1 = max(y1, top)
    ix2 = min(x2, right)
    iy2 = min(y2, bottom)
    
    # Check if there's any intersection
    if ix1 >= ix2 or iy1 >= iy2:
        return None 
    
    # Calculate new coordinates relative to crop (0,0 is top-left of crop)
    new_x1 = ix1 - left
    new_y1 = iy1 - top
    new_x2 = ix2 - left
    new_y2 = iy2 - top
    
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    
    # Filter by area visibility (20%)
    original_area = w * h
    new_area = new_w * new_h
    area_ratio = new_area / original_area if original_area > 0 else 0
    
    if area_ratio < 0.2:
        return None

    # ALWAYS return COCO format (xywh) as requested
    return [new_x1, new_y1, new_w, new_h], new_area

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        # VOC is xyxy
        bbox = [
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ]
        objects.append({'name': name, 'bbox': bbox})
    return objects

def process_and_crop_image(image_path, output_dir, annotation_source=None, 
                          ann_format=AnnotationFormat.NONE, voc_cat_manager=None, 
                          forced_image_id=None):
    """
    Crops image and returns a list of ADAPTED ANNOTATIONS in COCO FORMAT (dict).
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Calculate YOLO-compatible size
        target_width, target_height = get_yolo_compatible_size(orig_width, orig_height)
        
        # Center crop image
        cropped_image, crop_coords = center_crop_image(image, target_width, target_height)
        
        # Save cropped image
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, image_name)
        cropped_image.save(output_path)
        
        image_id = forced_image_id # Default to the sequential ID passed in
        final_annotations = []
        
        # --- Handle COCO Input ---
        if ann_format == AnnotationFormat.COCO and annotation_source:
            # Find original image ID in COCO data
            original_id = None
            for img_info in annotation_source['images']:
                if img_info['file_name'] == image_name:
                    original_id = img_info['id']
                    break
            
            if original_id is not None:
                # Use the original ID if found, otherwise use forced
                image_id = original_id 
                image_annotations = [ann for ann in annotation_source['annotations'] 
                                   if ann['image_id'] == original_id]
                
                for ann in image_annotations:
                    res = adapt_bbox(ann['bbox'], crop_coords, fmt='xywh')
                    if res:
                        new_bbox, new_area = res
                        adapted_ann = ann.copy()
                        adapted_ann['bbox'] = new_bbox
                        adapted_ann['area'] = new_area
                        # Ensure image_id matches
                        adapted_ann['image_id'] = image_id
                        final_annotations.append(adapted_ann)

        # --- Handle VOC Input ---
        elif ann_format == AnnotationFormat.VOC and annotation_source:
            base_name = os.path.splitext(image_name)[0]
            xml_path = os.path.join(annotation_source, base_name + '.xml')
            
            if os.path.exists(xml_path):
                objects = parse_voc_xml(xml_path)
                
                for obj in objects:
                    # Adapt bbox (Input is XYXY from XML)
                    res = adapt_bbox(obj['bbox'], crop_coords, fmt='xyxy')
                    if res:
                        new_bbox, new_area = res # Returns xywh
                        
                        # Get Category ID via the Manager (Mapped to COCO IDs)
                        cat_id = voc_cat_manager.get_id(obj['name'])
                        
                        if cat_id is not None:
                            # Construct COCO Annotation Dict
                            ann_struct = {
                                'image_id': image_id,
                                'category_id': cat_id,
                                'bbox': new_bbox, # [x,y,w,h]
                                'area': new_area,
                                'iscrowd': 0,
                                'segmentation': [] # Bbox only
                            }
                            final_annotations.append(ann_struct)

        crop_info = {
            'image_id': image_id,
            'image_name': image_name,
            'cropped_size': cropped_image.size,
            'adapted_annotations': final_annotations,
        }
        
        return crop_info
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save_final_coco_json(crop_infos, output_file, categories):
    """
    Consolidate all crop infos into a single COCO JSON file.
    """
    final_data = {
        'info': {'description': 'Adapted Dataset (Cropped/Corrupted)'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': categories
    }

    ann_id_counter = 1

    for info in crop_infos:
        # Add Image Entry
        final_data['images'].append({
            'id': info['image_id'],
            'file_name': info['image_name'],
            'width': info['cropped_size'][0],
            'height': info['cropped_size'][1]
        })

        # Add Annotation Entries
        for ann in info['adapted_annotations']:
            ann['id'] = ann_id_counter
            ann_id_counter += 1
            final_data['annotations'].append(ann)

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Saved COCO JSON to {output_file}")
    print(f"Images: {len(final_data['images'])}, Annotations: {len(final_data['annotations'])}, Categories: {len(categories)}")

def corrupt_image(image_path: str, image_path_base: str,
                  output_directory: str, output_type: OutputType,
                  corruptions: list, severity_levels: list) -> bool:
    try:
        kind = filetype.guess(image_path)
        if kind is None or not kind.mime.startswith('image'):
            return False

        if kind.extension == 'png':
            img_array = plt.imread(image_path) * 255
            img_array = img_array.astype(dtype=np.uint8)
        else:
            img_array = plt.imread(image_path)
            
        # Ensure we have 3 channels (handle grayscale or RGBA)
        if len(img_array.shape) == 2:
             img_array = np.stack((img_array,)*3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
             img_array = img_array[:,:,:3]

        output_path_stub = os.path.relpath(os.path.dirname(image_path), image_path_base)

        for corruption in corruptions:
            for severity in severity_levels:
                if output_type == OutputType.SUBDIRS:
                    output_path = os.path.join(output_directory, output_path_stub, corruption,
                                               str(severity), os.path.basename(image_path))
                elif output_type == OutputType.FILENAME:
                    fname, ext = os.path.splitext(os.path.basename(image_path))
                    fn = "{}_{}_{}{}".format(fname, corruption, str(severity), ext)
                    output_path = os.path.join(output_directory, output_path_stub, fn)
                
                out_dir = os.path.dirname(output_path)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                corrupted = corrupt(img_array, corruption_name=corruption, severity=severity)
                if corrupted.dtype != np.uint8:
                     corrupted = corrupted.astype(np.uint8)
                     
                Image.fromarray(corrupted).save(output_path)

        return True
    except Exception as e:
        print(f"Error corrupting {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="Directory which has to be processed")
    parser.add_argument("out_path", help="Output folder")
    parser.add_argument("output_type", choices=list(OutputType), type=OutputType,
                        help="How should the output be organized")
    parser.add_argument("--annotation-file", help="COCO .json file OR path to folder containing VOC .xml files")
    parser.add_argument("--output-annotation-file", help="Output path for the unified COCO JSON file", required=False)
    parser.add_argument("--crop-only", action="store_true", help="Only crop images and adapt annotations")
    parser.add_argument("-su", "--subset", choices=['common', 'validation', 'all', 'noise', 'blur',
                        'weather', 'digital'], help="Which subsets of corruptions should be applied")
    parser.add_argument("-c", "--corruptions", type=str, choices=corruption_dict.keys(), nargs="+",
                        help="Kind of corruptions to be applied")
    parser.add_argument("-se", "--severity", type=int, choices=range(1, 6), nargs="*",
                        help="Severity level of corruption")
    parser.add_argument("-j", type=int, default=1, help="Multiprocessing cores")
    parser.add_argument("-n", type=int, help="Limit the number of input images")

    opt = parser.parse_args()
    severity_levels = list(range(1, 6)) if opt.severity is None else opt.severity

    if not os.path.exists(opt.in_path):
        print(f"Input path does not exist: {opt.in_path}")
        exit(1)

    # --- Detect Annotation Format Correctly ---
    ann_format = AnnotationFormat.NONE
    annotation_source = None
    voc_manager = VocCategoryManager()
    
    if opt.annotation_file:
        if os.path.isdir(opt.annotation_file):
            ann_format = AnnotationFormat.VOC
            annotation_source = opt.annotation_file # It's a directory path
            print(f"Detected VOC format (directory). Will convert to COCO JSON with standard COCO IDs.")
            if not opt.output_annotation_file:
                print("Error: You must provide --output-annotation-file when converting VOC to COCO.")
                exit(1)
        
        elif os.path.isfile(opt.annotation_file) and opt.annotation_file.endswith('.json'):
            ann_format = AnnotationFormat.COCO
            print(f"Detected COCO format (json). Loading file...")
            with open(opt.annotation_file, 'r') as f:
                annotation_source = json.load(f)
        
        else:
            print(f"Warning: Annotation path provided but not recognized as Valid Directory or JSON file. Skipping annotations.")

    if not os.path.exists(opt.out_path):
        os.makedirs(opt.out_path)

    # Step 1: Crop images
    print("Step 1: Cropping images...")
    cropped_dir = os.path.join(opt.out_path, "cropped_images")
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(opt.in_path, "**", ext), recursive=True))
    
    if opt.n:
        image_files = image_files[:opt.n]
    
    print(f"Found {len(image_files)} images to process")
    
    crop_infos = []
    
    # Process images sequentially
    for idx, image_path in tqdm(enumerate(image_files), total=len(image_files), desc="Cropping"):
        crop_info = process_and_crop_image(
            image_path, 
            cropped_dir, 
            annotation_source=annotation_source, 
            ann_format=ann_format,
            voc_cat_manager=voc_manager,
            forced_image_id=idx + 1
        )
        if crop_info:
            crop_infos.append(crop_info)
    
    # Step 2: Save Unified Annotations
    if opt.output_annotation_file and ann_format != AnnotationFormat.NONE:
        print("Generating COCO JSON...")
        
        final_categories = []
        if ann_format == AnnotationFormat.COCO:
            final_categories = annotation_source['categories']
        else:
            final_categories = voc_manager.get_coco_categories()
            
        save_final_coco_json(crop_infos, opt.output_annotation_file, final_categories)

    if opt.crop_only:
        print("Crop-only mode completed.")
        return
    
    # Step 3: Apply corruptions
    print("\nStep 2: Applying corruptions...")
    corruptions = opt.corruptions if opt.corruptions else []
    if opt.subset:
        corruptions.extend(get_corruption_names(opt.subset))
    
    corruptions = list(set(corruptions))
    if not corruptions:
        corruptions = get_corruption_names('all')
    
    corrupted_dir = os.path.join(opt.out_path, "corrupted_images")
    if not os.path.exists(corrupted_dir):
        os.makedirs(corrupted_dir)
    
    cropped_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        cropped_files.extend(glob.glob(os.path.join(cropped_dir, ext)))
    
    pool = Pool(opt.j)
    progress_bar = tqdm(total=len(cropped_files), ascii=True)

    def update_bar(result):
        progress_bar.update()

    for filename in cropped_files:
        pool.apply_async(corrupt_image,
                         args=[filename, cropped_dir, corrupted_dir, opt.output_type, 
                               corruptions, severity_levels],
                         callback=update_bar)

    pool.close()
    pool.join()
    progress_bar.close()
    print("Done.")

if __name__ == "__main__":
    main()

#python helpers/0_corrupt_images.py /equilibrium/datasets/COCO2017/COCO2017_val/val2017 COCO_C filename --annotation-file /equilibrium/datasets/COCO2017/COCO2017_val/annotations/instances_val2017.json --output-annotation-file COCO_adapted_annotations.json -j 22