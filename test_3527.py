import os
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils
from torchvision import transforms as T
from solver import Solver
from data_loader import get_loader
from evaluation import *
import csv

# 指定模型关键字（包含 0.3527）
models_dir = './models'
model_keyword = '0.3527'
pkls = [os.path.join(models_dir,f) for f in os.listdir(models_dir) if f.endswith('.pkl') and model_keyword in f]
if not pkls:
    print('No model files matching', model_keyword)
    raise SystemExit(1)
# 如果有多个，选最新
model_path = max(pkls, key=os.path.getmtime)
print('Using model:', model_path)

# 配置
image_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class C: pass
config = C()
config.image_size = image_size
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

# dataloader
test_loader = get_loader(image_path=config.test_path,
                         image_size=config.image_size,
                         batch_size=1,
                         num_workers=0,
                         mode='test',
                         augmentation_prob=0.)

# model
solver = Solver(config, None, None, test_loader)
solver.unet.load_state_dict(torch.load(model_path))
solver.unet.to(solver.device)
solver.unet.eval()

# output dirs
out_dir = os.path.join(config.result_path, config.model_type + '_3527')
vis_dir = os.path.join(out_dir, 'vis')
os.makedirs(vis_dir, exist_ok=True)

# per-image csv
per_image_csv = os.path.join(out_dir, 'per_image_counts_3527.csv')
summary_csv = os.path.join(out_dir, 'summary_3527.csv')

# iterate
rows = []
acc = SE = SP = PC = F1 = JS = DC = 0.0
length = 0
for i, (images, GT) in enumerate(test_loader):
    images = images.to(solver.device)
    GT = GT.to(solver.device)
    with torch.no_grad():
        SR = torch.sigmoid(solver.unet(images))
    # metrics
    a = get_accuracy(SR,GT)
    s = get_sensitivity(SR,GT)
    sp = get_specificity(SR,GT)
    pc = get_precision(SR,GT)
    f1 = get_F1(SR,GT)
    js = get_JS(SR,GT)
    dc = get_DC(SR,GT)

    acc += a; SE += s; SP += sp; PC += pc; F1 += f1; JS += js; DC += dc
    length += images.size(0)

    # per-image counts (binary at 0.5)
    SR_mask = (SR > 0.5).float()
    TP = int(((SR_mask==1) & (GT==1)).sum().cpu())
    FP = int(((SR_mask==1) & (GT==0)).sum().cpu())
    FN = int(((SR_mask==0) & (GT==1)).sum().cpu())
    TN = int(((SR_mask==0) & (GT==0)).sum().cpu())

    rows.append({'file': f'vis_{i:03d}', 'TP':TP,'FP':FP,'FN':FN,'TN':TN,'acc':a,'SE':s,'SP':sp,'PC':pc,'F1':f1,'IoU':js,'Dice':dc})

    # save visualizations numbered
    vutils.save_image(images.data.cpu(), os.path.join(vis_dir, f'vis_{i:03d}_image.png'))
    vutils.save_image(SR_mask.data.cpu(), os.path.join(vis_dir, f'vis_{i:03d}_SR_mask.png'))
    vutils.save_image(GT.data.cpu(), os.path.join(vis_dir, f'vis_{i:03d}_GT.png'))
    # overlay
    img_np = images.data.cpu().squeeze(0).numpy()
    img_np = np.transpose(img_np, (1,2,0))
    img_uint8 = np.clip((img_np+1)/2*255,0,255).astype('uint8')
    mask_np = SR_mask.data.cpu().squeeze(0).squeeze(0).numpy()
    mask_uint8 = (mask_np>0.5).astype('uint8')*255
    pil_img = Image.fromarray(img_uint8)
    overlay = np.zeros_like(img_uint8)
    overlay[mask_uint8==255,0] = 255
    overlay = Image.fromarray(overlay)
    blended = Image.blend(pil_img, overlay, alpha=0.4)
    blended.save(os.path.join(vis_dir, f'vis_{i:03d}_overlay.png'))

# aggregate
if length>0:
    acc /= length; SE /= length; SP /= length; PC /= length; F1 /= length; JS /= length; DC /= length

# write CSVs
with open(per_image_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file','TP','FP','FN','TN','acc','SE','SP','PC','F1','IoU','Dice'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'acc','SE','SP','PC','F1','IoU','Dice','image_size','n_samples'])
    writer.writerow([os.path.basename(model_path), acc,SE,SP,PC,F1,JS,DC,image_size,length])

print('Done. Outputs saved to', out_dir)
