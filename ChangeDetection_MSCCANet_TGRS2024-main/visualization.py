'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics
import os
from tqdm import tqdm
import cv2

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载测试集的路径为 dataset/test/，其中 dataset 是 metadata.json 文件中 dataset_dir 的值
# 测试集加载流程：helpers.py的get_test_loaders() -> dataloaders.py的full_test_loader()
test_loader = get_test_loaders(opt, batch_size=1)

# the path of the model
# path = 'weights/snunet-32.pt'
path = './tmp/best_test_checkpoint_epoch_54.pt'
model = torch.load(path)

model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        file_path = './output_img/' + str(index_img).zfill(5)
        cv2.imwrite(file_path + '.png', cd_preds)

        index_img += 1
