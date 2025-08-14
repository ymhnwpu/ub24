import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
'''将datasets_npy/wh/train目录下按日期组织的数据，转换为训练所需的格式：
从 0 开始的连续编号的 .npy 文件（0.npy, 1.npy, 2.npy, ...），每个文件包含 4 个图像，每个图像有 6 个通道。
python prepare_training_data.py --src_dir datasets_npy/wh/train --dest_dir 数据集目录路径 --num_samples 生成的数据数量
参数--sample_num的值要与data.py中self.total_index严格保持一致。
随机选择日期目录和图像文件
将4个图像合并成一个训练样本
生成12000个训练样本（默认数量）
保存为连续编号的.npy文件
'''
def prepare_training_data(src_dir, dest_dir, num_samples=10, image_size=(256, 256)):
    """
    将按日期组织的数据转换为训练所需的格式
    
    Args:
        src_dir: 源数据目录 (如 datasets_npy/wh/train)
        dest_dir: 目标数据目录
        num_samples: 生成的训练样本数量
        image_size: 输出图像的尺寸 (height, width)
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    # 创建目标目录
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有日期目录
    date_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    print(f"找到 {len(date_dirs)} 个日期目录")
    
    # 为每个样本生成数据，添加进度条
    for i in tqdm(range(num_samples), desc="生成训练数据", unit="样本"):
        # 随机选择一个日期目录
        date_dir = np.random.choice(date_dirs)
        
        # 获取该目录下的所有.npy文件
        npy_files = list(date_dir.glob("*.npy"))
        if len(npy_files) < 4:
            print(f"警告: {date_dir} 目录下文件不足4个，跳过")
            continue
            
        # 随机选择4个文件
        selected_files = np.random.choice(npy_files, 4, replace=False)
        
        # 加载并合并这些文件
        images = []
        for npy_file in selected_files:
            img = np.load(npy_file)
            # 确保图像有6个通道
            if img.shape[0] < 6:
                # 如果通道数不足6个，用零填充
                padding = np.zeros((6 - img.shape[0], img.shape[1], img.shape[2]))
                img = np.concatenate([img, padding], axis=0)
            elif img.shape[0] > 6:
                # 如果通道数超过6个，只取前6个
                img = img[:6, :, :]
                
            # 调整图像尺寸
            if img.shape[1] != image_size[0] or img.shape[2] != image_size[1]:
                # 简单的裁剪或填充以适应目标尺寸
                h, w = img.shape[1], img.shape[2]
                target_h, target_w = image_size
                
                # 如果原图大于目标尺寸，则进行裁剪
                if h > target_h:
                    start_h = (h - target_h) // 2
                    img = img[:, start_h:start_h+target_h, :]
                elif h < target_h:
                    # 如果原图小于目标尺寸，则进行填充
                    pad_h = target_h - h
                    padding = np.zeros((6, pad_h, img.shape[2]))
                    img = np.concatenate([img, padding], axis=1)
                    
                # 处理宽度
                if w > target_w:
                    start_w = (w - target_w) // 2
                    img = img[:, :, start_w:start_w+target_w]
                elif w < target_w:
                    pad_w = target_w - w
                    padding = np.zeros((6, img.shape[1], pad_w))
                    img = np.concatenate([img, padding], axis=2)
            
            images.append(img)
        
        # 合并为一个数组
        combined_image = np.concatenate(images, axis=0)  # 形状: (24, H, W)
        
        # 保存为训练格式
        np.save(dest_path / f"{i}.npy", combined_image)
        
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 个样本")
    
    print(f"数据预处理完成，共生成 {num_samples} 个训练样本，图像尺寸为 {image_size}")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for STFMamba')
    parser.add_argument('--src_dir', required=True, help='Source data directory')
    parser.add_argument('--dest_dir', required=True, help='Destination directory for training data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of training samples to generate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256], help='Image size (height, width)')
    
    args = parser.parse_args()
    
    prepare_training_data(args.src_dir, args.dest_dir, args.num_samples, tuple(args.image_size))

if __name__ == '__main__':
    main()