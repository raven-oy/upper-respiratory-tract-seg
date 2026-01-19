import os
import torch
from solver import Solver
from data_loader import get_loader

models_dir='./models'
pkls=[os.path.join(models_dir,f) for f in os.listdir(models_dir) if f.endswith('.pkl') and '0.3527' in f]
model=max(pkls,key=os.path.getmtime)
print('model',model)
class C: pass
config=C()
config.image_size=512
config.t=3
config.img_ch=3
config.output_ch=1
config.batch_size=1
config.num_workers=0
config.mode='test'
config.model_type='U_Net'
config.model_path=models_dir
config.test_path='./dataset/test/'
config.result_path='./result/'
config.augmentation_prob=0.3527
config.lr=0.0003
config.beta1=0.5
config.beta2=0.999
config.num_epochs=100
config.num_epochs_decay=46
config.log_step=2
config.val_step=2
loader=get_loader(image_path=config.test_path,image_size=config.image_size,batch_size=1,num_workers=0,mode='test',augmentation_prob=0.)
solver=Solver(config,None,None,loader)
solver.unet.load_state_dict(torch.load(model))
solver.unet.to(solver.device)
solver.unet.eval()
for i,(images,GT) in enumerate(loader):
    images=images.to(solver.device); GT=GT.to(solver.device)
    with torch.no_grad(): SR=torch.sigmoid(solver.unet(images))
    SR_np=SR.cpu().squeeze().numpy(); GT_np=(GT.cpu().squeeze().numpy()>0.5).astype('uint8')
    print('sample',i,'SR_bin_sum', (SR_np>0.5).sum(), 'GT_sum', GT_np.sum())
    if i==0:
        import numpy as np
        for thr in [0.1,0.3,0.5,0.7,0.9]:
            pred=(SR_np>thr).astype('uint8')
            TP=((pred==1)&(GT_np==1)).sum(); FP=((pred==1)&(GT_np==0)).sum(); FN=((pred==0)&(GT_np==1)).sum()
            pc=TP/(TP+FP) if (TP+FP)>0 else 0
            print('thr',thr,'pred_sum',pred.sum(),'TP',TP,'FP',FP,'FN',FN,'PC',pc)
    break
