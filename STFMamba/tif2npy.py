import rasterio
import numpy as np
import os
 
if __name__ == '__main__':
    src_folder = 'datasets/wh/train/20151018_20171030/'
    dest_folder = 'npy_datasets/wh/train/20151018_20171030/'
 
    src_folder_names = os.listdir(src_folder)
    for name in src_folder_names:
        dest_folder_name = os.path.join(dest_folder, name)
        if not os.path.exists(dest_folder_name):
            os.mkdir(dest_folder_name)
    print("Folder name copied successfully!")
 
    for name in src_folder_names:
        full_name = os.path.join(src_folder, name)
        file_names = os.listdir(full_name)
        for file_name in file_names:
            with rasterio.open(os.path.join(full_name, file_name)) as src:
                img_data = src.read()
                basename, ext = os.path.splitext(file_name)
                np.save(os.path.join(dest_folder, name, basename+'.npy'), img_data)