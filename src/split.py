import os
import shutil
import argparse
import yaml
import random

from get_data import get_data, read_params

def train_and_test(config_path):
    config = get_data(config_path)
    root_dir = config["raw_data"]["data_src"]
    dest = config["load_data"]["preprocessed_data"]

    os.makedirs(os.path.join(dest, "train"), exist_ok=True)
    os.makedirs(os.path.join(dest, "test"), exist_ok=True)
    classes = ["NonViolence","Violence"]  
    
    for class_name in classes:
        os.makedirs(os.path.join(dest, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(dest, 'test', class_name), exist_ok=True)

    train_ratio = config["train"]["split_ratio"] 

    for class_name in classes:
        src_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(src_dir):
            print(f"Warning: Directory {src_dir} does not exist. Skipping...")
            continue
        
        files = os.listdir(src_dir)
        random.shuffle(files)  
        
        train_size = int(len(files) * train_ratio)
        train_files = files[:train_size]
        test_files = files[train_size:]

    
        for f in train_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest, "train", class_name, f))

      
        for f in test_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest, "test", class_name, f))

        print(f"Done splitting {class_name}: {len(train_files)} train, {len(test_files)} test images.")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_and_test(config_path=passed_args.config)