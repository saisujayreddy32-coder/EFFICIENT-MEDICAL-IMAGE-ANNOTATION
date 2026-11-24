"""Lightweight U-Net and dataset utilities for lung-mask segmentation."""
from pathlib import Path
import re, random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = 256
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def norm_stem(path):
    s = path.stem.lower()
    s = re.sub(r'(_mask|_masks|_segmentation|_label|_labels)$', '', s)
    return re.sub(r'[^a-z0-9]', '', s)


def discover_pairs(root):
    root = Path(root)
    all_files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    mask_files = [p for p in all_files if any(k in str(p.parent).lower() for k in ('mask', 'label', 'ground'))]
    image_files = [p for p in all_files if p not in mask_files and any(k in str(p.parent).lower() for k in ('cxr', 'image', 'image', 'png'))]
    if not image_files:
        image_files = [p for p in all_files if p not in mask_files]
    mask_map = {}
    for p in mask_files:
        mask_map.setdefault(norm_stem(p), p)
    pairs, used = [], set()
    for img in image_files:
        key = norm_stem(img)
        mask = mask_map.get(key)
        if mask and mask not in used:
            pairs.append((str(img), str(mask)))
            used.add(mask)
    # Fallback: pair by normalized filename across the entire tree.
    if not pairs and mask_files:
        for img in [p for p in all_files if p not in mask_files]:
            candidates = [m for m in mask_files if norm_stem(m) == norm_stem(img)]
            if candidates:
                pairs.append((str(img), str(candidates[0])))
    return pairs, [str(p) for p in image_files if all(p != Path(a) for a, _ in pairs)]


def load_gray(path, size=IMAGE_SIZE, mask=False):
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    im = Image.open(path).convert('L').resize((size, size), resample)
    arr = np.asarray(im, dtype=np.float32)
    if mask:
        return (arr > 30).astype(np.float32)
    return arr / 255.0

class SegDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        x, y = self.pairs[i]
        image = torch.from_numpy(load_gray(x)[None, ...])
        mask = torch.from_numpy(load_gray(y, mask=True)[None, ...])
        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=DoubleConv(1,16); self.e2=DoubleConv(16,32); self.e3=DoubleConv(32,64); self.pool=nn.MaxPool2d(2); self.mid=DoubleConv(64,128); self.mc_dropout=nn.Dropout2d(p=0.20); self.up3=nn.ConvTranspose2d(128,64,2,2); self.d3=DoubleConv(128,64); self.up2=nn.ConvTranspose2d(64,32,2,2); self.d2=DoubleConv(64,32); self.up1=nn.ConvTranspose2d(32,16,2,2); self.d1=DoubleConv(32,16); self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); z=self.mid(self.pool(c)); z=self.mc_dropout(z); z=self.d3(torch.cat([self.up3(z),c],1)); z=self.d2(torch.cat([self.up2(z),b],1)); z=self.d1(torch.cat([self.up1(z),a],1)); return self.out(z)

def dice_score(pred, target, eps=1e-7):
    p=(pred>0.5).float(); inter=(p*target).sum(); return float((2*inter+eps)/(p.sum()+target.sum()+eps))

def train_model(pairs, model_path, epochs=3, batch_size=4):
    if len(pairs) < 2: raise ValueError('At least 2 image-mask pairs are required to train.')
    random.seed(42); pairs=list(pairs); random.shuffle(pairs)
    split=max(1, int(len(pairs)*0.8)); train_pairs=pairs[:split]; val_pairs=pairs[split:] or pairs[:1]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=UNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); bce=nn.BCEWithLogitsLoss()
    loader=DataLoader(SegDataset(train_pairs),batch_size=min(batch_size,len(train_pairs)),shuffle=True)
    history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); probs=torch.sigmoid(logits); loss=bce(logits,y)+(1-dice_score_tensor(probs,y)); loss.backward(); opt.step(); losses.append(float(loss.item()))
        model.eval(); dices=[]
        with torch.no_grad():
            for x,y in DataLoader(SegDataset(val_pairs),batch_size=1): dices.append(dice_score(torch.sigmoid(model(x.to(device))).cpu(),y))
        history.append({'epoch':epoch+1,'loss':round(float(np.mean(losses)),4),'val_dice':round(float(np.mean(dices)),4)})
    torch.save({'state_dict':model.cpu().state_dict(),'history':history,'image_size':IMAGE_SIZE},model_path)
    return {'history':history,'device':device,'train_count':len(train_pairs),'validation_count':len(val_pairs)}

def dice_score_tensor(p,t,eps=1e-7):
    inter=(p*t).sum(); return (2*inter+eps)/(p.sum()+t.sum()+eps)

def predict(model_path, image_path, mc_passes=12):
    """Return mean mask probability and MC-dropout predictive uncertainty.

    Dropout remains active during inference to produce stochastic segmentation
    samples. The returned score is mean predictive entropy in [0, 1].
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=UNet().to(device)
    ckpt=torch.load(model_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.mc_dropout.train()  # Enable only MC dropout; keep BatchNorm in evaluation mode.
    x=torch.from_numpy(load_gray(image_path)[None,None,...]).to(device)
    samples=[]
    with torch.no_grad():
        for _ in range(max(2, int(mc_passes))):
            samples.append(torch.sigmoid(model(x))[0,0].cpu().numpy())
    stack=np.stack(samples, axis=0)
    probs=stack.mean(axis=0)
    entropy=-(probs*np.log(probs+1e-8)+(1-probs)*np.log(1-probs+1e-8))
    return probs, float(np.clip(entropy.mean()/np.log(2), 0.0, 1.0))
