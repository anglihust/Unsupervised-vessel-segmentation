import os
import numpy as np
import cv2
from natsort import natsorted
import random
from glob import glob
from tqdm import tqdm
from keras import backend as K
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, GridDistortion,ElasticTransform,RandomScale,Rotate)
import tensorflow as tf
from bunch import Bunch
import json


transformation_train= [HorizontalFlip(p=0.5),VerticalFlip(p=0.5),GridDistortion(p=0.5),ElasticTransform(p=0.5),ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=15, p=0.5)]
transformation_val =[]

transformation_train_source= [HorizontalFlip(p=0.5),VerticalFlip(p=0.5),ShiftScaleRotate(scale_limit=0.5, shift_limit=0.1, rotate_limit=15, p=0.5)]
transformation_val_source =[]

def create_transformer(transformations, images):
    target = {}
    for i, image in enumerate(images[1:]):
        target['image' + str(i)] = 'image'
    return albu.Compose(transformations, p=1, additional_targets=target)(image=images[0],
                                                                        image0=images[1])
def fresh_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)


def paint_border(imgs,patch_size,stride):
    assert (len(imgs.shape) == 2)
    img_h = imgs.shape[0]  # height of the full image
    img_w = imgs.shape[1]  # width of the full image
    leftover_h = (img_h - patch_size) % stride  # leftover on the h dim
    leftover_w = (img_w - patch_size) % stride  # leftover on the w dim
    full_imgs=imgs
    if (leftover_h != 0):  #change dimension of img_h
        tmp_imgs = np.zeros((img_h+(stride-leftover_h),img_w))
        tmp_imgs[0:img_h,0:img_w,] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0],img_w+(stride - leftover_w)))
        tmp_imgs[0:img_h,0:img_w] =imgs
        full_imgs = tmp_imgs
    return full_imgs

def extract_patches(full_imgs, patch_size,stride):
    assert (len(full_imgs.shape)==4)  #4D arrays
    t_extract_img = paint_border(full_imgs[0,:,:,0],patch_size,stride)
    img_h = t_extract_img.shape[0]  #height of the full image
    img_w = t_extract_img.shape[1] #width of the full image
    N_patches_img = full_imgs.shape[0]*((img_h-patch_size)//stride+1)*((img_w-patch_size)//stride+1)  #// --> division between integers
    patches = np.empty([N_patches_img,patch_size,patch_size,1])
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):
        extract_img = paint_border(full_imgs[i,:,:,0],patch_size,stride)
        for h in range((img_h-patch_size)//stride+1):
            for w in range((img_w-patch_size)//stride+1):
                patch = extract_img[h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
                patches[iter_tot,:,:,0]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_img)
    return patches,img_h,img_w


def recompone_overlap(preds,patch_size,stride,img_h,img_w):
    assert (len(preds.shape)==4)  #4D arrays
    N_patches_h = (img_h-patch_size)//stride+1
    N_patches_w = (img_w-patch_size)//stride+1
    N_patches_img = N_patches_h * N_patches_w

    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h-patch_size)//stride+1):
            for w in range((img_w-patch_size)//stride+1):
                full_prob[i,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size,:]+=preds[k]
                full_sum[i,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size,:]+=1
                k+=1

    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1)  #at least one
    final_avg = full_prob/full_sum
    return final_avg

def img_process(data,rl=False):
    assert(len(data.shape)==4)
    data=data.transpose(0, 3, 1,2)
    if rl==False:#原始图片是否已经预处理过
        train_imgs=np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
            train_img[:,0,:,:]=data[:,index,:,:]
            #print("original",np.max(train_img),np.min(train_img))
            train_img = dataset_normalized(train_img)   #归一化
            train_img=np.array(train_img,dtype=int)
            #print("normal",np.max(train_img), np.min(train_img))
            train_img = clahe_equalized(train_img)      #限制性直方图归一化
            #print("clahe",np.max(train_img), np.min(train_img))
            train_img=np.array(train_img,dtype=int)
            train_img = adjust_gamma(train_img, 1.2)    #gamma校正
            #print("gamma",np.max(train_img), np.min(train_img))
            #print("reduce",np.max(train_img), np.min(train_img))
            train_imgs[:,index,:,:]=train_img[:,0,:,:]

    else:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            train_img = dataset_normalized(train_img)
    train_imgs=train_imgs.transpose(0, 2, 3, 1)
    return train_imgs

def histo_equalized(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))
    imgs_normalized=imgs_normalized*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs



def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config

def swish(x):
    return K.sigmoid(x)*x

def window_search_2D(mask,patch_size,stride):
    for i in range((mask.shape[0]-patch_size)//stride+1):
        for j in range((mask.shape[1]-patch_size)//stride+1):
            index_x = i*stride+patch_size
            index_y = j*stride+patch_size
            if index_x>= mask.shape[0]:
                index_x = mask.shape[0]
            if index_y>= mask.shape[1]:
                index_y = mask.shape[1]
            patch = mask[index_x-patch_size:index_x,index_y-patch_size:index_y]
            if i==0 and j==0:
                patch_array=np.expand_dims(patch,axis=0)
            else:
                patch_array=np.append(patch_array,np.expand_dims(patch,axis=0),axis=0)
    return patch_array


class Data_set():
    def __init__(self,config):
        main_path = os.path.join(os.getcwd(),'data')
        target_img_path=os.path.join(main_path,'target')
        target_mask_path =os.path.join(main_path,'target_label')
        source_img_path_train = os.path.join(main_path,'source')
        source_mask_path_train= os.path.join(main_path,'source_label')
        source_img_path_eval = os.path.join(main_path,'source_eval')
        source_mask_path_eval= os.path.join(main_path,'source_eval_label')
        unsupervised_img_path =os.path.join(main_path,'target_uns')

        self.data_path =os.path.join(os.getcwd(),'data')
        self.patch_diameter=config.patch_diameter
        self.stride  = config.stride
        self.normalized = config.normalize
        self.batch_size=config.batch_size
        self.class_num =config.class_num
        self.img_width=config.src_width
        self.img_height=config.src_height
        self.binary_out = config.binary_out
        self.unsupervised = config.unsupervised
        # read target images, output images patch

        self.img_patches_array,self.mask_patches_array = self.target_data_loading(target_img_path,target_mask_path,self.stride)
        self.img_patches_array=np.array(self.img_patches_array,dtype=np.float32)
        self.mask_patches_array=np.array(self.mask_patches_array,dtype=np.float32)
        index=np.random.choice(self.img_patches_array.shape[0],self.img_patches_array.shape[0],replace=False)
        self.target_train_index = index[0:int(index.shape[0]*0.1)]
        self.target_eval_index = index[int(index.shape[0]*0.1)::]
        self.target_subsample = self.target_train_index.shape[0]*15
        self.target_eval_subsample = self.target_eval_index.shape[0]

        # read source images, output whole images

        self.source_img_train,self.source_mask_train=self.source_data_loading(source_img_path_train,source_mask_path_train)
        self.source_img_eval,self.source_mask_eval=self.source_data_loading(source_img_path_eval,source_mask_path_eval)
        self.source_img_train=np.array(self.source_img_train,dtype=np.float32)
        self.source_mask_train=np.array(self.source_mask_train,dtype=np.float32)
        self.source_img_eval=np.array(self.source_img_eval,dtype=np.float32)
        self.source_mask_eval=np.array(self.source_mask_eval,dtype=np.float32)

        #self.aug=albu.Compose([HorizontalFlip(p=0.5),
                             #ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=15, p=0.5),
                             #GridDistortion(p=0.5)])

        if self.unsupervised:
            self.unsupervised_patches_array,_ = self.target_data_loading(unsupervised_img_path,unsupervised_img_path,32)
            self.unsupervised_patches_array  = np.array(self.unsupervised_patches_array,dtype=np.float32)
            self.unsupervised_subsample=self.unsupervised_patches_array.shape[0]

        self.class_weight=[1.0,0.0]
        self.train_attnlist=[np.where(self.source_mask_train[:,:,:,0]==np.max(self.source_mask_train[:,:,:,0]))]
        self.soruce_subsample=int(len(self.train_attnlist[0][0])*0.05)

        self.eval_attnlist = [np.where(self.source_mask_eval[:, :, :, 0] == np.max(self.source_mask_eval[:, :, :, 0]))]
        self.source_val_subsample=int(len(self.eval_attnlist[0][0])*0.05)

    def target_data_loading(self,ori_path,noi_path,stride,iscolor=False,inverse=False):
        oripath=[]
        noipath=[]
        for ext in('/*.tif','/*.png'):
            oripath.extend(natsorted(glob(ori_path+ext)))
            noipath.extend(natsorted(glob(noi_path+ext)))
        assert(len(oripath)==len(noipath))
        for i in tqdm(range(len(oripath))):
            if iscolor:
                ori_img =cv2.imread(oripath[i])
                ori_img=ori_img[:,:,1]*0.75+ori_img[:,:,0]*0.25
            else:
                ori_img =cv2.imread(oripath[i],0)
            mask =plt.imread(noipath[i],0)
            mask =np.array(mask,dtype=int)
            mask[mask==255]=1
            mask_patches = window_search_2D(mask,self.patch_diameter,stride)
            img_patches = window_search_2D(ori_img,self.patch_diameter,stride)
            if i ==0:
                mask_patches_array=mask_patches
                img_patches_array=img_patches
            else:
                mask_patches_array = np.append(mask_patches_array,mask_patches,axis=0)
                img_patches_array= np.append(img_patches_array,img_patches,axis=0)

        mask_patches_array=np.expand_dims(mask_patches_array,axis=3)
        img_patches_array=np.expand_dims(img_patches_array,axis=3)

        if not self.binary_out:
            back_patches_array=1-mask_patches_array
            mask_patches_array=np.append(mask_patches_array,back_patches_array,axis=3)

        if inverse:
            img_patches_array=255-img_patches_array

        if self.normalized:
            img_patches_array = img_patches_array/255*2-1

        return img_patches_array,mask_patches_array

    def source_data_loading(self,ori_path,noi_path,iscolor=False,inverse=True):
        oripath=[]
        noipath=[]
        for ext in('/*.tif','/*.png'):
            oripath.extend(natsorted(glob(ori_path+ext)))
            noipath.extend(natsorted(glob(noi_path+ext)))
        assert(len(oripath)==len(noipath))
        ori_img_array = np.zeros([len(oripath),self.img_width,self.img_height,1])
        noi_img_array = np.zeros([len(oripath),self.img_width,self.img_height,1])
        for i in tqdm(range(len(oripath))):
            if iscolor:
                ori_img =cv2.imread(oripath[i])
                width,height,channel=ori_img.shape
                ori_img_array[i,0:width,0:height,0]=np.asarray(ori_img[:,:,1]*0.75+ori_img[:,:,0]*0.25)
            else:
                ori_img =cv2.imread(oripath[i],0)
                width,height=ori_img.shape
                ori_img_array[i,0:width,0:height,0] = ori_img

            noi_img =plt.imread(noipath[i],0)
            noi_img =np.array(noi_img,dtype=int)
            noi_img[noi_img==255]=1
            noi_img_array[i,0:width,0:height,0] = (noi_img==1)

        if not self.binary_out:
            back_patches_array =1-noi_img_array
            noi_img_array=np.append(noi_img_array,back_patches_array,axis=3)

        ori_img_array=img_process(ori_img_array)

        if inverse:
            ori_img_array=255-ori_img_array
        if self.normalized:
            ori_img_array=ori_img_array/255*2-1
        return ori_img_array, noi_img_array

    def target_genDef(self,train_imgs,train_masks,transformation,mask=True,aug=False):
        while 1:
            for t in range(int(train_imgs.shape[0]/ self.batch_size)):
                index = np.random.choice(train_imgs.shape[0],self.batch_size,replace=False)
                btrain_imgs = train_imgs[index,:,:,:]
                btrain_masks = train_masks[index,:,:,:]
                imgs = np.zeros([self.batch_size,self.patch_diameter,self.patch_diameter,1],dtype=np.float32)
                masks = np.zeros([self.batch_size,self.patch_diameter,self.patch_diameter,2],dtype=np.float32)
                for i in range(self.batch_size):
                    X = btrain_imgs[i,:,:,:]
                    Y1 = np.array(btrain_masks[i,:,:,:],dtype=np.uint8)
                    if aug:
                        images = [X, Y1]
                        transformed = create_transformer(transformation, images)
                        imgs[i,:,:,:] = transformed['image']
                        masks[i,:,:,:] = transformed['image0']
                    else:
                        imgs[i,:,:,:] = X
                        masks[i,:,:,:] = Y1
                if mask:
                    yield (imgs, masks)
                else:
                    yield imgs


    def target_train_gen(self):
        return self.target_genDef(self.img_patches_array[self.target_train_index,:,:,:],self.mask_patches_array[self.target_train_index,:,:,:],transformation_train)

    def target_eval_gen(self):
        return self.target_genDef(self.img_patches_array[self.target_eval_index,:,:,:],self.mask_patches_array[self.target_eval_index,:,:,:],transformation_val)

    def target_unsurpervised_gen(self):
        return self.target_genDef(self.unsupervised_patches_array,self.unsupervised_patches_array,transformation_val,False,False)

    def source_genDef(self,train_imgs,train_masks,attnlist,class_weight,transformation,aug=True):
        while 1:
            Nimgs=train_imgs.shape[0]
            for t in range(int(len(attnlist[0][0])*0.05 / self.batch_size)):
                X = np.zeros([self.batch_size,self.patch_diameter, self.patch_diameter,1])
                if not self.binary_out:
                    Y = np.zeros([self.batch_size,self.patch_diameter, self.patch_diameter,2])
                else:
                    Y = np.zeros([self.batch_size,self.patch_diameter, self.patch_diameter,1])
                for j in range(self.batch_size):
                    [i_center, x_center, y_center] = self.CenterSampler(attnlist,class_weight,Nimgs)
                    patch = train_imgs[i_center, int(y_center - self.patch_diameter / 2):int(y_center + self.patch_diameter / 2),int(x_center - self.patch_diameter / 2):int(x_center + self.patch_diameter / 2),:]
                    patch_mask = train_masks[i_center, int(y_center - self.patch_diameter / 2):int(y_center + self.patch_diameter / 2),int(x_center - self.patch_diameter / 2):int(x_center + self.patch_diameter / 2),:]
                    patch_mask = np.array(patch_mask,dtype=np.uint8)
                    if aug:
                        images = [patch, patch_mask]
                        transformed = create_transformer(transformation, images)
                        X[j, :, :, :] = transformed['image']
                        Y[j, :, :, :] = transformed['image0']
                    else:
                        X[j, :, :, :] = np.array(patch,dtype=np.float32)
                        Y[j, :, :, :] = np.array(patch_mask,dtype=np.float32)
                yield (X, Y)

    def train_gen(self):
        return self.source_genDef(self.source_img_train,self.source_mask_train,self.train_attnlist,self.class_weight,transformation_train_source,True)

    def val_gen(self):
        return self.source_genDef(self.source_img_eval, self.source_mask_eval,self.eval_attnlist,self.class_weight,transformation_val_source,False)

    def CenterSampler(self,attnlist,class_weight,Nimgs):
        t = attnlist[0]
        cid = random.randint(0, t[0].shape[0] - 1)
        i_center = t[0][cid]
        y_center = t[1][cid] + random.randint(0 - int(self.patch_diameter / 2), 0 + int(self.patch_diameter / 2))
        x_center = t[2][cid] + random.randint(0 - int(self.patch_diameter / 2), 0 + int(self.patch_diameter / 2))

        if y_center < self.patch_diameter / 2:
            y_center = self.patch_diameter / 2
        elif y_center > self.img_width - self.patch_diameter / 2:
            y_center = self.img_width - self.patch_diameter / 2

        if x_center < self.patch_diameter / 2:
            x_center = self.patch_diameter / 2
        elif x_center > self.img_height - self.patch_diameter / 2:
            x_center = self.img_height - self.patch_diameter / 2

        return i_center, x_center, y_center
