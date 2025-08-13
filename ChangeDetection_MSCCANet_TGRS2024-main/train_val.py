import matplotlib
matplotlib.use('Agg')  # 在导入 plt 之前设置后端
import matplotlib.pyplot as plt
# 添加上述代码以支持无显示环境

import datetime
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support as prfs
"""precision_recall_fscore_support计算精确率、召回率和F1分数
prfs(labels, preds, average='binary', pos_label=1)计算二分类的精确率、召回率和F1分数
labels是真实标签， preds是预测标签
average='binary'表示二分类，pos_label=1表示正类标签
返回值是一个包含精确率、召回率和F1分数的元组
"""
from utils.parser import get_parser_with_args
# get_parser_with_args获取参数解析器
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics,exp_lr_scheduler_with_warmup,get_test_loaders)
"""get_loaders获取训练和验证数据加载器
get_criterion获取损失函数
load_model加载模型
initialize_metrics初始化指标
get_mean_metrics获取平均指标
set_metrics设置指标
exp_lr_scheduler_with_warmup获取学习率调度器
get_test_loaders获取测试数据加载器
"""
import os
import logging
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# tqdm是一个进度条库，用于显示循环的进度
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
"""confusion_matrix计算混淆矩阵
混淆矩阵是一个表格，用于描述分类模型的性能
它显示了真实标签和预测标签之间的关系
confusion_matrix(y_true, y_pred, labels=None)计算混淆矩阵
y_true是真实标签，y_pred是预测标签，labels是标签列表
"""


# Initialize Parser and define arguments 初始化解析器，定义参数
parser, metadata = get_parser_with_args()
_, metadata_val = get_parser_with_args()
_, metadata_test = get_parser_with_args()
opt = parser.parse_args() # opt是解析后的参数对象，opt包含了所有解析后的参数，可以通过opt.参数名访问具体的参数值
print("---Total epochs in this training---opt.epochs:", opt.epochs)

# Initialize experiments log 初始化实验日志，保存到/log目录下
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

# Set up environment: define paths, download data, and set device
# 用户输入GPU_ID
gpu_num = input("GPU_ID:")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
"""设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 如果只使用第0个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
如果使用多个GPU，可以设置为"0,1"等
例如，如果只使用第0个GPU，可以设置为"0"
如果使用多个GPU，可以设置为"0,1"等
这将限制PyTorch只使用指定的GPU设备
如果不设置CUDA_VISIBLE_DEVICES，PyTorch将使用所有可用的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

# 设置随机种子以确保实验可重复性
def seed_torch(seed):
    random.seed(seed) # 随机种子值
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(seed=777) # 设置随机种子为777，这将确保每次运行代码时，随机数生成器产生的序列是相同的，这对于实验的可重复性非常重要

# 计算返回评估指标Precision, Recall, F1, IoU, Accuracy
# opt配置参数， model模型， batch_iter批次迭代器， tbar进度条， epoch当前轮数， state状态（train, val, test）
def metrics_calculation(opt, model, batch_iter, tbar, epoch, state):     
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    info = state + "_epoch {} info "
    with torch.no_grad():
        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description(info.format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            cd_preds = model(batch_img1, batch_img2)
            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)
            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                            cd_preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()              
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp
    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    IoU = tp / (tp + fp + fn) 
    Acc = (tp + tn) / (tp + fp + tn + fn) 
    return P, R, F1, IoU, Acc

# 获取训练和验证数据加载器
train_loader, val_loader = get_loaders(opt)
test_loader = get_test_loaders(opt)

# Load Model then define other aspects of the model
logging.info('LOADING Model')
net_name = "MSCCANet" # 可选"SNUNet_ECAM", "UTNetV2", "MSTransception", "MSCCANet"，具体见helpers.py中的load_model函数
if not os.path.exists('./tmp'):
    os.mkdir('./tmp')
if not os.path.exists('./tmp/train'):
    os.mkdir('./tmp/train')
if not os.path.exists('./tmp/val'):
    os.mkdir('./tmp/val')
if not os.path.exists('./tmp/test'):
    os.mkdir('./tmp/test')
if not os.path.exists('./chart'):
    os.mkdir('./chart')
if not os.path.exists('./chart/test'):
    os.mkdir('./chart/test')
if not os.path.exists('./chart/train'):
    os.mkdir('./chart/train')
if not os.path.exists('./chart/val'):
    os.mkdir('./chart/val')
path_in = './tmp/train'
files = os.listdir(path_in)
b = []
# 获取所有以't'结尾的文件名，并提取数字部分
# 例如，文件名为'checkpoint_epoch_10.pt'，提取出数字10
for f in files:
    if(f[-1]=='t'):
        f = f[17:-3]
        f = int(f)
        b.append(f)
# 对提取的数字进行排序，不排序可能会导致加载模型时不正确（不是最新模型）
b.sort()
# 如果没有找到以't'结尾的文件，则b为空。如果b为空，则表示没有保存的模型，重新训练；如果b不为空，则表示有保存的模型，加载最新的模型
if(len(b)==0):
    model = load_model(net_name, opt, dev)
    start_epoch = 0
    print('无保存模型，将从头开始训练！')
else:
    path_model = path_in+'/checkpoint_epoch_'+str(b[-1])+'.pt'
    model = torch.load(path_model)
    start_epoch = b[-1]
    print('加载 epoch {} 成功！'.format(start_epoch))

criterion = get_criterion(opt) # 获取损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
# torch.optim.AdamW是Adam优化器的变种，具有权重衰减功能
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)


# Set starting values设置初始值
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1} # 初始化最佳指标
logging.info('STARTING training')
total_step = -1
best_val_f1 = 0
best_test_f1 = 0
best_epoch = -1
best_val_epoch = -1
plt.figure(num=1)
plt.figure(num=2)
plt.figure(num=3)

# 在每次训练前调用此函数加载可用的历史训练数据
def load_training_history(path):
    """加载历史训练数据用于断点续训"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)
        return history
    return None
# 在每个epoch结束时调用此函数保存训练历史
def save_training_history(epoch, history_dict):
    """保存训练历史数据用于断点续训"""
    with open('./tmp/train/training_history.json', 'w') as f:
        json.dump(history_dict, f)

history = load_training_history('./tmp/train/training_history.json')

 # np.linspace(start, stop, num)生成一个从start到stop的等间距数组，num是生成的元素个数
t = np.linspace(0, opt.epochs+1, opt.epochs+1)
# 注意下面18个数组的长度都是opt.epochs+1，且第一个元素（下标为0）不保存历史数据，从下标1开始保存
epoch_test_loss = 0 * t
epoch_test_corrects = 0 * t
epoch_test_recalls = 0 * t
epoch_test_precisions = 0 * t
epoch_test_f1scores = 0 * t
epoch_test_learning_rate = 0 * t

epoch_val_loss = 0 * t
epoch_val_corrects = 0 * t
epoch_val_precisions = 0 * t
epoch_val_recalls = 0 * t
epoch_val_f1scores = 0 * t
epoch_val_learning_rate = 0 * t

epoch_train_loss = 0 * t
epoch_train_corrects = 0 * t
epoch_train_precisions = 0 * t
epoch_train_recalls = 0 * t
epoch_train_f1scores = 0 * t
epoch_train_learning_rate = 0 * t
# 断点续训加载历史数据。如果没有训练历史（history为空）则使用上面的全0数组从头训练
if history and start_epoch > 0:
    epoch_test_loss = np.array(history['test_loss'])
    epoch_test_corrects = np.array(history['test_corrects'])
    epoch_test_recalls = np.array(history['test_recalls'])
    epoch_test_precisions = np.array(history['test_precisions'])
    epoch_test_f1scores = np.array(history['test_f1scores'])
    epoch_test_learning_rate = np.array(history['test_learning_rate'])
    # np.resize调整ndarray数组长度，否则后续无法将新一轮数据保存到ndarray数组中
    epoch_test_loss = np.resize(epoch_test_loss, (opt.epochs+1))
    epoch_test_corrects = np.resize(epoch_test_corrects, (opt.epochs+1))
    epoch_test_recalls = np.resize(epoch_test_recalls, (opt.epochs+1))
    epoch_test_precisions = np.resize(epoch_test_precisions, (opt.epochs+1))
    epoch_test_f1scores = np.resize(epoch_test_f1scores, (opt.epochs+1))
    epoch_test_learning_rate = np.resize(epoch_test_learning_rate, (opt.epochs+1))

    epoch_val_loss = np.array(history['test_loss'])
    epoch_val_corrects = np.array(history['test_corrects'])
    epoch_val_recalls = np.array(history['test_recalls'])
    epoch_val_precisions = np.array(history['test_precisions'])
    epoch_val_f1scores = np.array(history['test_f1scores'])
    epoch_val_learning_rate = np.array(history['test_learning_rate'])

    epoch_val_loss = np.resize(epoch_val_loss, (opt.epochs+1))
    epoch_val_corrects = np.resize(epoch_val_corrects, (opt.epochs+1))
    epoch_val_recalls = np.resize(epoch_val_recalls, (opt.epochs+1))
    epoch_val_precisions = np.resize(epoch_val_precisions, (opt.epochs+1))
    epoch_val_f1scores = np.resize(epoch_val_f1scores, (opt.epochs+1))
    epoch_val_learning_rate = np.resize(epoch_val_learning_rate, (opt.epochs+1))

    epoch_train_loss = np.array(history['test_loss'])
    epoch_train_corrects = np.array(history['test_corrects'])
    epoch_train_recalls = np.array(history['test_recalls'])
    epoch_train_precisions = np.array(history['test_precisions'])
    epoch_train_f1scores = np.array(history['test_f1scores'])
    epoch_train_learning_rate = np.array(history['test_learning_rate'])

    epoch_train_loss = np.resize(epoch_train_loss, (opt.epochs+1))
    epoch_train_corrects = np.resize(epoch_train_corrects, (opt.epochs+1))
    epoch_train_recalls = np.resize(epoch_train_recalls, (opt.epochs+1))
    epoch_train_precisions = np.resize(epoch_train_precisions, (opt.epochs+1))
    epoch_train_f1scores = np.resize(epoch_train_f1scores, (opt.epochs+1))
    epoch_train_learning_rate = np.resize(epoch_train_learning_rate, (opt.epochs+1))
print("len(epoch_train_loss):", len(epoch_train_loss))

epoch_train_list = [epoch_train_loss, epoch_train_corrects, epoch_train_precisions, epoch_train_recalls, epoch_train_f1scores, epoch_train_learning_rate]
epoch_train_loss, epoch_train_corrects, epoch_train_precisions, epoch_train_recalls, epoch_train_f1scores, epoch_train_learning_rate = epoch_train_list

epoch_val_list = [epoch_val_loss, epoch_val_corrects, epoch_val_precisions, epoch_val_recalls, epoch_val_f1scores, epoch_val_learning_rate]
epoch_val_loss, epoch_val_corrects, epoch_val_precisions, epoch_val_recalls, epoch_val_f1scores, epoch_val_learning_rate = epoch_val_list

epoch_test_list = [epoch_test_loss, epoch_test_corrects, epoch_test_precisions, epoch_test_recalls, epoch_test_f1scores, epoch_test_learning_rate]
epoch_test_loss, epoch_test_corrects, epoch_test_precisions, epoch_test_recalls, epoch_test_f1scores, epoch_test_learning_rate = epoch_test_list

print("---NOTICE: epoch starts with 1 not 0---")
for epoch in range(start_epoch+1,opt.epochs+1): # 从start_epoch+1开始到opt.epochs结束
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    test_metrics = initialize_metrics()
    exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=opt.learning_rate, epoch=epoch, warmup_epoch=5, max_epoch=opt.epochs)


    # Begin Training
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar: # batch_img1和batch_img2是输入图像，labels是真实标签
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        # 将数据移到GPU
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        # 清零梯度
        optimizer.zero_grad()
        # 获取模型预测结果
        cd_preds = model(batch_img1, batch_img2)
        # 计算损失
        cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        # 反向传播
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)

        # train_metrics = set_metrics(train_metrics,
                                    # cd_loss,
                                    # cd_corrects,
                                    # cd_train_report,
                                    # scheduler.get_last_lr())
                                
        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    exp_scheduler)
                                    
        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)
        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels
    #scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
    
    logging.info('updata the model')
    metadata['train_metrics'] = mean_train_metrics
    # Chart 图都保存在/chart目录下
    # epoch_loss[epoch] = mean_train_metrics['cd_losses']
    # epoch_corrects[epoch] = mean_train_metrics['cd_corrects']
    # epoch_precisions[epoch] = mean_train_metrics['cd_precisions']
    # epoch_recalls[epoch] = mean_train_metrics['cd_recalls']
    # epoch_f1scores[epoch] = mean_train_metrics['cd_f1scores']
    # epoch_learning_rate[epoch] = mean_train_metrics['learning_rate']
    epoch_train_loss[epoch] = mean_train_metrics['cd_losses']
    epoch_train_corrects[epoch] = mean_train_metrics['cd_corrects']
    epoch_train_precisions[epoch] = mean_train_metrics['cd_precisions']
    epoch_train_recalls[epoch] = mean_train_metrics['cd_recalls']
    epoch_train_f1scores[epoch] = mean_train_metrics['cd_f1scores']
    epoch_train_learning_rate[epoch] = mean_train_metrics['learning_rate']

    plt.figure(num=1)
    plt.clf() 
    # 绘图下标为[0:56]即0~55共56个点，下标0不保存，故共55轮（1~55）
    l1_1, = plt.plot(t[:epoch+1], epoch_train_f1scores[:epoch+1], label='Train f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_train_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_train_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_train_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_train_f1scores[max_indx]),xy=(t[max_indx],epoch_train_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/train/Train_F1-epoch.png', bbox_inches='tight')
   
    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_train_recalls[:epoch+1], label='Train recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('recall-epoch')
    plt.savefig('./chart/train/Train_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_train_precisions[:epoch+1], label='Train precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('precisions-epoch')
    plt.savefig('./chart/train/Train_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_train_loss[:epoch+1], label='Train loss')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('IoU-epoch')
    plt.savefig('./chart/train/Train_IoU-epoch.png', bbox_inches='tight')
    
    plt.figure(num=5)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_train_learning_rate[:epoch+1], label='Train learning_rate')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('learning_rate-epoch')
    plt.savefig('./chart/train/Train_learning_rate-epoch.png', bbox_inches='tight')
    # Save model and log 将每轮训练的参数、超参数和指标保存到/tmp/train目录下
    with open('./tmp/train/metadata_train_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata, fout)


    # Begin Validation
    model.eval()
    batch_iter1 = 0
    tbar1 = tqdm(val_loader)
    Precision_val, Recall_val, F1_val, IoU_val, ACC_val = metrics_calculation(opt, model, batch_iter1, tbar1, epoch, 'val')
    current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
    epoch_val_learning_rate[epoch] = current_lr # 记录验证阶段的学习率
        
    epoch_val_loss[epoch] = IoU_val
    epoch_val_corrects[epoch] = ACC_val
    epoch_val_precisions[epoch] = Precision_val
    epoch_val_recalls[epoch] = Recall_val
    epoch_val_f1scores[epoch] = F1_val
    metircs_val = "IoU = "+ str(IoU_val) + ", " + "ACC = "+ str(ACC_val) + ", " + "Precision = " + str(Precision_val)+ ", " +"Recall ="+ str(Recall_val)+ ", " +"F1_score ="+ str(F1_val)
    metadata_val["metrics"] = metircs_val
    logging.info("Val_EPOCH {} Val METRICS ".format(epoch)+metircs_val)
    # Save model and log 将每轮验证的参数、超参数和指标保存到/tmp/val目录下
    with open('./tmp/val/metadata_val_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata_val, fout)
    if (F1_val > best_val_f1):
        best_val_f1 = F1_val
        best_val_epoch = epoch
        torch.save(model, './tmp/best_val_checkpoint_epoch_'+str(best_val_epoch)+'.pt')
        with open('./tmp/metadata_best_val_checkpoint_epoch_'+str(best_val_epoch)+'.json', 'w') as fout:
            metadata_val['epoch'] = str(epoch)
            json.dump(metadata_val, fout)

    plt.figure(num=1)
    plt.clf() 
    l1_1, = plt.plot(t[:epoch+1], epoch_val_f1scores[:epoch+1], label='val f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_val_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_val_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_val_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_val_f1scores[max_indx]),xy=(t[max_indx],epoch_val_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/val/Val_F1-epoch.png', bbox_inches='tight')
   
    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_recalls[:epoch+1], label='val recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('recall-epoch')
    plt.savefig('./chart/val/Val_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_precisions[:epoch+1], label='val precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('precisions-epoch')
    plt.savefig('./chart/val/Val_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_loss[:epoch+1], label='val loss')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('IoU-epoch')
    plt.savefig('./chart/val/Val_IoU-epoch.png', bbox_inches='tight')
    
    plt.figure(num=5)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_val_learning_rate[:epoch+1], label='val learning_rate')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    plt.title('learning_rate-epoch')
    plt.savefig('./chart/val/Val_learning_rate-epoch.png', bbox_inches='tight')
    
    
    # Begin test
    model.eval()
    batch_iter2 = 0
    tbar2 = tqdm(test_loader)
    Precision, Recall, F1, IoU, ACC = metrics_calculation(opt, model, batch_iter2, tbar2, epoch, 'test')
    epoch_test_learning_rate[epoch] = current_lr # 记录验证阶段的学习率
    
    epoch_test_loss[epoch] = IoU
    epoch_test_corrects[epoch] = ACC
    epoch_test_precisions[epoch] = Precision
    epoch_test_recalls[epoch] = Recall
    epoch_test_f1scores[epoch] = F1
    metircs_test = "IoU = "+ str(IoU) + ", " + "ACC = "+ str(ACC) + ", " + "Precision = " + str(Precision)+ ", " +"Recall ="+ str(Recall)+ ", " +"F1_score ="+ str(F1)
    metadata_test["metrics"] = metircs_test
    logging.info("Test_EPOCH {} Test METRICS ".format(epoch)+metircs_test)
    # Save model and log 将每轮测试的参数、超参数和指标保存到/tmp/test目录下
    with open('./tmp/test/metadata_test_epoch_' + str(epoch) + '.json', 'w') as fout:
        json.dump(metadata_test, fout)
    if (F1 > best_test_f1):
        # torch.save(model, './tmp/best_test_checkpoint_epoch.pt')
        best_test_f1 = F1
        best_epoch = epoch
        torch.save(model, './tmp/best_test_checkpoint_epoch_'+str(epoch)+'.pt')
        with open('./tmp/metadata_best_test_checkpoint_epoch_'+str(epoch)+'.json', 'w') as fout:
            metadata_test['epoch'] = str(epoch)
            json.dump(metadata_test, fout)
    # comet.log_asset(upload_metadata_file_path)
    #best_metrics = mean_val_metrics 
    logging.info("The current best_val_epoch {} Test ".format(best_val_epoch)+"F1_score: "+str(best_val_f1))
    logging.info("The current best epoch {} Test ".format(best_epoch)+"F1_score: "+str(best_test_f1))
    
    plt.figure(num=1)
    plt.clf() 
    l1_1, = plt.plot(t[:epoch+1], epoch_test_f1scores[:epoch+1], label='test f1scores')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_f1scores[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_f1scores[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_f1scores[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_f1scores[max_indx]),xy=(t[max_indx],epoch_test_f1scores[max_indx]))
    plt.title('F1-epoch')
    plt.savefig('./chart/test/Test_F1-epoch.png', bbox_inches='tight')

    plt.figure(num=2)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_recalls[:epoch+1], label='test recalls')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_recalls[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_recalls[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_recalls[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_recalls[max_indx]),xy=(t[max_indx],epoch_test_recalls[max_indx]))
    plt.title('recall-epoch')
    plt.savefig('./chart/test/Test_recall-epoch.png', bbox_inches='tight')
    
    plt.figure(num=3)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_precisions[:epoch+1], label='test precisions')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_precisions[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_precisions[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_precisions[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_precisions[max_indx]),xy=(t[max_indx],epoch_test_precisions[max_indx]))
    plt.title('precisions-epoch')
    plt.savefig('./chart/test/Test_precisions-epoch.png', bbox_inches='tight')
    
    plt.figure(num=4)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_loss[:epoch+1], label='test IoU')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_loss[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_loss[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_loss[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_loss[max_indx]),xy=(t[max_indx],epoch_test_loss[max_indx]))
    plt.title('IoU-epoch')
    plt.savefig('./chart/test/Test_IoU-epoch.png', bbox_inches='tight')
    
    plt.figure(num=5)
    plt.clf()
    l1_1, = plt.plot(t[:epoch+1], epoch_test_learning_rate[:epoch+1], label='test learning_rate')
    plt.legend(handles=[l1_1])
    plt.grid()
    plt.gcf().gca().set_xlim(left = 0)
    max_indx=np.argmax(epoch_test_learning_rate[:epoch+1])#max value index
    plt.plot(t[max_indx],epoch_test_learning_rate[max_indx],'ks')
    show_max='['+str(int(t[max_indx]))+' '+str(epoch_test_learning_rate[max_indx])+']'
    plt.annotate(show_max,xytext=(t[max_indx],epoch_test_learning_rate[max_indx]),xy=(t[max_indx],epoch_test_learning_rate[max_indx]))
    plt.title('learning_rate-epoch')
    plt.savefig('./chart/test/Test_learning_rate-epoch.png', bbox_inches='tight')


    # 保存每轮模型到/tmp/train目录下，便于断点续训【/tmp/train|val|test/目录下的.json文件中只保存了每轮的其他数据，并没有保存模型】
    torch.save(model, './tmp/train/checkpoint_epoch_' + str(epoch) + '.pt')
    # 保存每个epoch的历史数据到/tmp/train目录下用于下次断点续训
    history_dict = {
        'test_loss': epoch_test_loss.tolist(),
        'test_corrects': epoch_test_corrects.tolist(),
        'test_recalls': epoch_test_recalls.tolist(),
        'test_precisions': epoch_test_precisions.tolist(),
        'test_f1scores': epoch_test_f1scores.tolist(),
        'test_learning_rate': epoch_test_learning_rate.tolist()
    }
    save_training_history(epoch, history_dict)
    print(f'Epoch {epoch} finished.')
    print()
writer.close()  # close tensor board
print('Done!')