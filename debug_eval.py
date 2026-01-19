import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
from solver import Solver
from data_loader import get_loader
from evaluation import *

# config
models_dir = './models'
pkls = [os.path.join(models_dir,f) for f in os.listdir(models_dir) if f.endswith('.pkl') and '0.3527' in f]
if not pkls:
    print('no model')
    raise SystemExit(1)
model = max(pkls, key=os.path.getmtime)
print('model', model)

class C: pass
config = C()
config.image_size = 512
config.t = 3
config.img_ch = 3
config.output_ch = 1
config.num_epochs = 100
config.num_epochs_decay = 46
config.batch_size = 1
config.num_workers = 0
config.lr = 0.0003
config.beta1 = 0.5
config.beta2 = 0.999
config.augmentation_prob = 0.3527
config.log_step = 2
config.val_step = 2
config.mode = 'test'
config.model_type = 'U_Net'
config.model_path = models_dir
config.train_path = './dataset/train/'
config.valid_path = './dataset/valid/'
config.test_path = './dataset/test/'
config.result_path = './result/'
config.cuda_idx = 0

# loader
test_loader = get_loader(image_path=config.test_path, image_size=config.image_size, batch_size=1, num_workers=0, mode='test', augmentation_prob=0.)

solver = Solver(config, None, None, test_loader)
solver.unet.load_state_dict(torch.load(model))
solver.unet.to(solver.device)
solver.unet.eval()

print('Test samples:', len(test_loader.dataset))

for i, (images, GT) in enumerate(test_loader):
    print('\n--- sample', i, '---')
    images = images.to(solver.device)
    GT = GT.to(solver.device)
    with torch.no_grad():
        SR = torch.sigmoid(solver.unet(images))
    print('SR shape', SR.shape, 'GT shape', GT.shape)
    print('SR min/max', float(SR.min()), float(SR.max()))
    print('GT min/max', float(GT.min()), float(GT.max()))
    sum_SR_probs = float(torch.sum(SR).cpu())
    sum_SR_bin = float(torch.sum((SR>0.5).float()).cpu())
    sum_GT = float(torch.sum((GT==torch.max(GT)).float()).cpu())
    print('sum SR probs, sum SR bin, sum GT:', sum_SR_probs, sum_SR_bin, sum_GT)
    # compute TP FP FN via bitwise
    SR_bin = (SR>0.5)
    GT_pos = (GT==torch.max(GT))
    TP_and = ((SR_bin)&(GT_pos)).float().sum().item()
    FP_and = ((SR_bin)&(~GT_pos)).float().sum().item()
    FN_and = ((~SR_bin)&(GT_pos)).float().sum().item()
    print('TP/FP/FN (bitwise):', TP_and, FP_and, FN_and)
    # compute with arithmetic equality method
    TP_arith = (((SR_bin==1) + (GT_pos==1)) == 2).float().sum().item()
    TP_and_again = (((SR_bin==1) & (GT_pos==1)).float().sum().item())
    print('TP_arith, TP_and_again:', TP_arith, TP_and_again)
    # compute precision via evaluation function
    pc = get_precision(SR, GT, threshold=0.5)
    f1 = get_F1(SR, GT, threshold=0.5)
    iou = get_JS(SR, GT, threshold=0.5)
    print('eval PC, F1, IoU:', pc, f1, iou)

print('\nDone diagnostics')
