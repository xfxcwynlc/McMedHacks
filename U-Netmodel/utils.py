import torch
import torchvision
from keras_preprocessing.image import dataframe_iterator

#from dataset import GlycogenDataset
from torch.utils.data import TensorDataset,DataLoader,ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from IPython import embed
import tifffile




def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def cropping(arr1,arr2,sz):
    m_x = int(arr1.shape[0]/2)
    m_y = int(arr1.shape[1]/2)
    return arr1[(m_x-sz):(m_x+sz),(m_y-sz):(m_y+sz),:], arr2[(m_x-sz):(m_x+sz),(m_y-sz):(m_y+sz),:]


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=3,
    pin_memory=True,
):
    train_ds,rawshape,sz = LoadOneDataset(val_dir)
    val_ds,maskshape,sz = LoadOneDataset(val_dir)
    #train_ds = ConcatDataset(dataset1)
    #val_ds = ConcatDataset(dataset2)

    train_loader = DataLoader(
         train_ds,
         batch_size=batch_size,
         num_workers=num_workers,
         pin_memory=pin_memory,
         shuffle=False,
     )

    val_loader = DataLoader(
         val_ds,
         batch_size=batch_size,
         num_workers=num_workers,
         pin_memory=pin_memory,
         shuffle=False,
     )
    return train_loader, val_loader,maskshape,sz



def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for (x, y)  in loader:
            x = x.unsqueeze(1).to(device)
            #y = y.to(device).unsqueeze(1)
            y = y.unsqueeze(1).to(device)
            preds =  model(x)
            #preds = torch.argmax(model(x),dim=1)
            #maxvalue = torch.max(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


#
# def dice_multi(preds,target,smooth=1e-7):
#     #changes to one-hot encoding:
#     onehot_flat = (torch.nn.functional.one_hot(target)).flatten()
#     #change logits to softmax so it sums to one at dim=1
#     preds_flat = torch.softmaspreds.


def MultiTensorDataset(train_dir):
    multi_tensorDataset = []
    #print("traindir is:")
    #print(train_dir)

    for i in os.listdir(train_dir):
        train_sub_dir = train_dir + i + '/images/'
        mask_sub_dir = train_dir + i + '/masks/'
        #print(train_sub_dir)
        t1 = np.load(train_sub_dir + os.listdir(train_sub_dir)[0])
        m1 = np.load(mask_sub_dir + os.listdir(mask_sub_dir)[0])
        arr_t, arr_m = cropping(t1,m1)
        tensor_train = torch.Tensor(arr_t.transpose())
        tensor_mask = torch.Tensor(arr_m.transpose())
        my_dataset = TensorDataset(tensor_train, tensor_mask)
        multi_tensorDataset.append(my_dataset)
    return multi_tensorDataset

def LoadOneDataset(test_dir):
    t1 = np.load(test_dir)
    if t1.shape[1]>800:  
        sz = int(t1.shape[1]/2)-int(t1.shape[1]*0.19)
    elif t1.shape[1]>600:
        sz = int(t1.shape[1]/2)-int(t1.shape[1]*0.06)
    else:
        sz = int(t1.shape[1]/2)
    arr_t, arr_m = cropping(t1,t1,sz)
    test = torch.Tensor(arr_t.transpose())
    mydataset = TensorDataset(test,test)
    return mydataset,t1.shape,sz
'''
def save_tiff(tensor):
    for i in range(tensor.shape[0]):
        

    grid = torchvision.utils.make_grid(tensor)
    index=tensor.shape[0]
    tensor[index,:,:,:]
    embed()
'''


def save_numpy_arr(predic,savel):
    for i in range(predic.shape[0]):
        temp = predic[i,:,:,:].squeeze(0).detach().cpu().numpy()
        savel.append(temp)

        
        
def restoreimage(preds, rawshape,sz):
    m_temp = np.zeros(rawshape)
    mid_x = int(rawshape[0]/2)
    mid_y = int(rawshape[1]/2)
    m_temp[mid_x-sz:mid_x+sz, mid_y-sz:mid_y+sz,:] = preds.transpose()
    print(m_temp.shape)
    return m_temp

def save_predictions_as_imgs(
    loader, model, rawshape, sz, folder="saved_images4/", device="cuda"
):
    model.eval()
    savel = []
    for idx, (x, y) in enumerate(loader):
        x1 = x.unsqueeze(1).to(device=device)
        with torch.no_grad():
            #problems with this part of code.
            temp = model(x1)
            preds = torch.sigmoid(temp)
            preds = (preds > 0.5).float()
            
            #preds = preds.argmax(temp,dim=1)
            #print("preds shape is")
            #print(preds.shape)
            #preds012 = torch.argmax(temp,dim=1).float().unsqueeze(1)

        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )
        #torchvision.utils.save_image(
        #    x, f"{folder}/raw_{idx}.png", normalize=True
        #)
        save_numpy_arr(preds,savel)
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png", normalize=True
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png", normalize=True)
    savel=np.array(savel)
    savel = restoreimage(savel,rawshape,sz)
    print(savel.shape)
    with open(os.path.join(folder,"results.npy"), 'wb') as f:
        np.save(f,savel)
   

    model.train()
