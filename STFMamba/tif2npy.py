import rasterio
import numpy as np
import os
import argparse
from pathlib import Path
# 通义灵码

def convert_tif_to_npy(src_path, dest_path=None):
    """
    将单个.tif文件转换为.npy文件
    
    Args:
        src_path (str): 输入.tif文件路径
        dest_path (str): 输出.npy文件路径，默认为同名文件但扩展名为.npy
    
    Returns:
        str: 输出文件路径
    """
    if dest_path is None:
        dest_path = Path(src_path).with_suffix('.npy')
    
    with rasterio.open(src_path) as src:
        img_data = src.read()
        np.save(dest_path, img_data)
    
    print(f"成功转换: {src_path} -> {dest_path}")
    return dest_path


def convert_folder_tif_to_npy(src_folder, dest_folder):
    """
    将文件夹中的所有.tif文件转换为.npy文件
    
    Args:
        src_folder (str): 包含.tif文件的源文件夹路径
        dest_folder (str): 存放.npy文件的目标文件夹路径
    """
    src_path = Path(src_folder)
    dest_path = Path(dest_folder)
    
    # 创建目标文件夹
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.tif文件
    tif_files = list(src_path.rglob('*.tif')) + list(src_path.rglob('*.tiff'))
    
    if not tif_files:
        print(f"在 {src_folder} 中未找到.tif文件")
        return
    
    print(f"找到 {len(tif_files)} 个.tif文件，开始转换...")
    
    for tif_file in tif_files:
        try:
            # 计算相对于源文件夹的路径
            relative_path = tif_file.relative_to(src_path)
            npy_file = dest_path / relative_path.with_suffix('.npy')
            
            # 创建目标子文件夹
            npy_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换文件
            convert_tif_to_npy(str(tif_file), str(npy_file))
        except Exception as e:
            print(f"转换 {tif_file} 时出错: {e}")


def main():
    # 使用命令行运行本程序：python tif2npy.py 输入文件或文件夹 -o 输出文件或文件夹
    parser = argparse.ArgumentParser(description='将.tif文件转换为.npy文件')
    parser.add_argument('src', help='输入.tif文件路径或包含.tif文件的文件夹路径')
    parser.add_argument('-o', '--output', help='输出.npy文件路径或文件夹路径')
    
    args = parser.parse_args()
    
    src_path = Path(args.src)
    
    if src_path.is_file():
        # 处理单个文件
        if not args.output:
            args.output = src_path.with_suffix('.npy')
        convert_tif_to_npy(str(src_path), args.output)
    elif src_path.is_dir():
        # 处理文件夹
        if not args.output:
            args.output = str(src_path) + '_npy'
        convert_folder_tif_to_npy(str(src_path), args.output)
    else:
        print(f"错误: {src_path} 不是有效的文件或文件夹")


if __name__ == '__main__':
    main()