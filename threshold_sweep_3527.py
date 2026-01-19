import os
import numpy as np
import torch
from solver import Solver
from data_loader import get_loader
from evaluation import *

# find 0.3527 model
models_dir='./models'
pkls=[os.path.join(models_dir,f) for f in os.listdir(models_dir) if f.endswith('.pkl') and '0.3527' in f]
if not pkls:
    print('no model')
    raise SystemExit(1)
model=max(pkls,key=os.path.getmtime)
print('model',model)

class C: pass
config=C()
config.image_size=512
config.t=3
config.img_ch=3
config.output_ch=1
config.num_epochs=100
config.num_epochs_decay=46
config.batch_size=1
config.num_workers=0
config.lr=0.0003
config.beta1=0.5
config.beta2=0.999
config.augmentation_prob=0.3527
config.log_step=2
config.val_step=2
config.mode='test'
config.model_type='U_Net'
config.model_path=models_dir
config.train_path='./dataset/train/'
config.valid_path='./dataset/valid/'
config.test_path='./dataset/test/'
config.result_path='./result/'
config.cuda_idx=0

# loader
loader=get_loader(image_path=config.test_path,image_size=config.image_size,batch_size=1,num_workers=0,mode='test',augmentation_prob=0.)

solver=Solver(config,None,None,loader)
solver.unet.load_state_dict(torch.load(model))
solver.unet.to(solver.device)
solver.unet.eval()

# collect SR and GT
all_probs=[]
for i,(images,GT) in enumerate(loader):
    images=images.to(solver.device)
    GT=GT.to(solver.device)
    with torch.no_grad():
        SR=torch.sigmoid(solver.unet(images))
    SR_np=SR.cpu().squeeze().numpy()
    GT_np=(GT.cpu().squeeze().numpy()>0.5).astype('uint8')
    all_probs.append((SR_np,GT_np))

thresholds=np.linspace(0.1,0.9,17)
print('threshold,acc,SE,SP,PC,F1,IoU,Dice')
for thr in thresholds:
    accs=[];SEs=[];SPs=[];PCs=[];F1s=[];IoUs=[];Dices=[]
    for SR_np,GT_np in all_probs:
        pred=(SR_np>thr).astype('uint8')
        TP=((pred==1)&(GT_np==1)).sum()
        FP=((pred==1)&(GT_np==0)).sum()
        FN=((pred==0)&(GT_np==1)).sum()
        TN=((pred==0)&(GT_np==0)).sum()
        total=TP+FP+FN+TN
        accs.append((TP+TN)/total if total>0 else 0)
        SEs.append(TP/(TP+FN) if (TP+FN)>0 else 0)
        SPs.append(TN/(TN+FP) if (TN+FP)>0 else 0)
        PCs.append(TP/(TP+FP) if (TP+FP)>0 else 0)
        F1s.append((2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN)>0 else 0)
        IoUs.append(TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0)
        Dices.append((2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN)>0 else 0)
    print(f'{thr:.2f},{np.mean(accs):.4f},{np.mean(SEs):.4f},{np.mean(SPs):.4f},{np.mean(PCs):.4f},{np.mean(F1s):.4f},{np.mean(IoUs):.4f},{np.mean(Dices):.4f}')
