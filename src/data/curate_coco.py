import os
import json
import zipfile
import urllib.request
from PIL import Image
from io import BytesIO
from collections import defaultdict

import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def curate_coco(num_images=500, target_size=(224, 224), out_dir=None):
    if out_dir is None:
        save_dir = os.path.join(BASE_DIR, "data", "coco_val")
    else:
        save_dir = out_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print("Downloading official COCO val2017 annotations...")
    
    # Official COCO annotations zip
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    try:
        req = urllib.request.Request(ann_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with zipfile.ZipFile(BytesIO(response.read())) as z:
                with z.open('annotations/instances_val2017.json') as f:
                    coco_data = json.load(f)
    except Exception as e:
        print(f"Failed to fetch official coco annotations: {e}")
        return
    
    # Map categories for better readability
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Build defaultdict
    ann_by_img = defaultdict(list)
    for ann in coco_data['annotations']:
        ann_by_img[ann['image_id']].append(ann)
    
        
    # Download first num_images metadata
    images_to_download = coco_data['images'][:num_images]
    
    ground_truth_map = {}
    print(f"Downloading {num_images} images directly from coco servers")
    
    for i, img_info in enumerate(images_to_download):
        img_id = img_info['id']
        img_url = img_info['coco_url']
        
        # Download the image
        try:
            req = urllib.request.Request(
                img_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                img_data = response.read()
            
            # Convert raw byte to image
            image = Image.open(BytesIO(img_data))
        
        except Exception as e:
            print(f"Skipping image {img_id} due to error")
            continue
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = image.resize(target_size)
        
        image_filename = f"coco_{img_id:012d}.jpg"
        image_path = os.path.join(save_dir, image_filename)
        image.save(image_path, "JPEG", quality=95)
        
        # Extract bounding boxes for this image
        annotations = ann_by_img.get(img_id, [])
        
        formatted_annotations = []
        for ann in annotations:
            formatted_annotations.append({
                'category_id': ann['category_id'],
                'category_name': categories.get(ann['category_id'], 'unknown'),
                'bbox': ann['bbox']
            })
            
        ground_truth_map[image_filename] = {
            'original_width': img_info['width'], # Store original width and height to correctly scale bbox
            'original_height': img_info['height'],
            'objects': formatted_annotations
        }
        
        if (i + 1) % 50 == 0:
            print(f"Downloaded {i+1}/{num_images}")
            
        
    labels_path = os.path.join(save_dir, "coco_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_map, f, indent=4)
    
    print(f"Success! COCO dataset saved to {save_dir}")
    print(f"Ground truth mapping saved to {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate COCO val2017 dataset for AFS-VFM")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to sample (default: 500)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory path (default: data/coco_val)")
    args = parser.parse_args()
    curate_coco(num_images=args.num_images, out_dir=args.out_dir)