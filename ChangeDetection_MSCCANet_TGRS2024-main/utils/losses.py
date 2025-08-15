# 论文中的混合损失函数
from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
#type(labels): <class 'torch.Tensor'>
#cd_preds.type: <class 'tuple'>
#cd_preds是长度为1的tuple,里面是一个tensor，
#当进行for循环时即将真正的预测值取出，
#for循环实际也就进行了一次。
'''组合损失函数：Loss_total = Focal Loss + Dice Loss
Dice Loss = 1 - Dice Coefficient = 1 - (2*TP)/(2*TP + FP + FN)
IoU = TP/(TP + FP + FN)
所以最小化Dice Loss等价于最大化IoU，故训练阶段降低Loss_total能在验证阶段提高IoU。
'''
def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    #print ("predictions:",len(predictions))
    for prediction in predictions:
        #print ("prediction:",prediction.size())
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss