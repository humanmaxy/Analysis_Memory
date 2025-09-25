#!/usr/bin/env python3
"""
Fix dataset class IDs from 1-indexed to 0-indexed
"""
import json
import os
import shutil
from pathlib import Path

def fix_coco_annotations(annotation_file, output_file=None, backup=True):
    """
    Fix COCO annotation file to use 0-indexed class IDs instead of 1-indexed
    
    Args:
        annotation_file: Path to the COCO annotation JSON file
        output_file: Path for the fixed file (default: overwrites original)
        backup: Whether to create a backup of the original file
    """
    annotation_path = Path(annotation_file)
    
    if not annotation_path.exists():
        print(f"Error: Annotation file {annotation_path} does not exist")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = annotation_path.with_suffix('.json.backup')
        if not backup_path.exists():
            shutil.copy2(annotation_path, backup_path)
            print(f"Created backup: {backup_path}")
    
    # Load annotations
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return False
    
    # Check current class IDs
    if 'categories' in data:
        original_ids = [cat['id'] for cat in data['categories']]
        print(f"Original category IDs: {original_ids}")
        
        # Check if already 0-indexed
        if min(original_ids) == 0:
            print("Categories are already 0-indexed, no changes needed")
            return True
    
    # Fix categories (convert from 1-indexed to 0-indexed)
    if 'categories' in data:
        print("Fixing category IDs...")
        id_mapping = {}
        for i, category in enumerate(data['categories']):
            old_id = category['id']
            new_id = i  # 0-indexed
            category['id'] = new_id
            id_mapping[old_id] = new_id
            print(f"  Category '{category['name']}': {old_id} -> {new_id}")
    
    # Fix annotations (update category_id references)
    if 'annotations' in data:
        print("Fixing annotation category IDs...")
        fixed_count = 0
        for annotation in data['annotations']:
            old_cat_id = annotation['category_id']
            if old_cat_id in id_mapping:
                annotation['category_id'] = id_mapping[old_cat_id]
                fixed_count += 1
        print(f"Fixed {fixed_count} annotation category IDs")
    
    # Save the fixed file
    output_path = Path(output_file) if output_file else annotation_path
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved fixed annotations to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving fixed file: {e}")
        return False

def fix_dataset_directory(dataset_dir):
    """
    Fix all annotation files in a dataset directory
    
    Args:
        dataset_dir: Path to the dataset directory (should contain train/, val/, etc.)
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist")
        return False
    
    print(f"Fixing dataset in: {dataset_path}")
    
    # Common annotation file patterns
    annotation_patterns = [
        "**/_annotations.coco.json",
        "**/annotations.json",
        "**/instances_*.json",
        "**/*annotations*.json"
    ]
    
    fixed_files = []
    
    for pattern in annotation_patterns:
        for annotation_file in dataset_path.glob(pattern):
            print(f"\nProcessing: {annotation_file}")
            if fix_coco_annotations(annotation_file):
                fixed_files.append(annotation_file)
    
    if fixed_files:
        print(f"\nSuccessfully fixed {len(fixed_files)} annotation files:")
        for file in fixed_files:
            print(f"  - {file}")
    else:
        print("\nNo annotation files found or fixed")
    
    return len(fixed_files) > 0

def validate_fixed_dataset(dataset_dir):
    """
    Validate that the dataset has been fixed correctly
    """
    dataset_path = Path(dataset_dir)
    
    annotation_files = list(dataset_path.glob("**/_annotations.coco.json"))
    annotation_files.extend(dataset_path.glob("**/annotations.json"))
    
    all_valid = True
    
    for annotation_file in annotation_files:
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            if 'categories' in data:
                cat_ids = [cat['id'] for cat in data['categories']]
                min_id = min(cat_ids) if cat_ids else 0
                max_id = max(cat_ids) if cat_ids else 0
                
                print(f"{annotation_file.name}: Categories {min_id}-{max_id}")
                
                if min_id != 0:
                    print(f"  WARNING: Categories don't start from 0!")
                    all_valid = False
                
                if 'annotations' in data and data['annotations']:
                    ann_cat_ids = set(ann['category_id'] for ann in data['annotations'])
                    if min(ann_cat_ids) < 0 or max(ann_cat_ids) >= len(data['categories']):
                        print(f"  WARNING: Invalid annotation category IDs!")
                        all_valid = False
        
        except Exception as e:
            print(f"Error validating {annotation_file}: {e}")
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix dataset class IDs from 1-indexed to 0-indexed")
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't fix")
    
    args = parser.parse_args()
    
    if args.validate_only:
        print("Validating dataset...")
        if validate_fixed_dataset(args.dataset_dir):
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
    else:
        print("Fixing dataset class IDs...")
        if fix_dataset_directory(args.dataset_dir):
            print("\n✅ Dataset fixed successfully!")
            print("\nValidating the fix...")
            if validate_fixed_dataset(args.dataset_dir):
                print("✅ Validation passed!")
            else:
                print("❌ Validation failed - please check manually")
        else:
            print("❌ Failed to fix dataset")
    
    print("\nNote: Original files have been backed up with .backup extension")