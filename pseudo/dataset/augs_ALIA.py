import numpy as np
import random
import torch
import scipy.stats as stats
import torchvision.transforms as T
import cv2


# # # # # # # # # # # # # # # # # # # # # 
# # 0 random box
# # # # # # # # # # # # # # # # # # # # # 
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1 cutmix label-adaptive 
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits, 
        labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 2) get box
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 3) labeled adaptive
    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
        
            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            
            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    # 4) copy and paste
    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            
        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits
    
    return unlabeled_image, unlabeled_mask, unlabeled_logits 


# # # # # # # # # # # # # # # # # # # # # 
# # 2 cutmix  
# # # # # # # # # # # # # # # # # # # # #
def cut_mix(image_u_aug, label_u_aug, logits_u_aug, image_u_intense=None):
    image_cutmix = image_u_aug.clone()
    if image_u_intense is not None:
        image_intense_cutmix = image_u_intense.clone()
    label_cutmix = label_u_aug.clone()
    logits_cutmix = logits_u_aug.clone()
    u_rand_index = torch.randperm(image_u_aug.size()[0])[:image_u_aug.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(image_u_aug.size(), lam=np.random.beta(4, 4))

    for i in range(0, image_cutmix.shape[0]):
        image_cutmix[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            image_u_aug[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        if image_u_intense is not None:
            image_intense_cutmix[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                image_u_intense[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        label_cutmix[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            label_u_aug[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        logits_cutmix[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            logits_u_aug[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    if image_u_intense is not None:
            return image_cutmix, label_cutmix, logits_cutmix, image_intense_cutmix

    return image_cutmix, label_cutmix, logits_cutmix


# # # # # # # # # # # # # # # # # # # # # 
# # 3 cutout  
# # # # # # # # # # # # # # # # # # # # #
def cut_out(image_u_aug=None, label_u_aug=None, logits_u_aug=None):
    image_cutmix = image_u_aug.clone()
    label_cutmix = label_u_aug.clone()
    logits_cutmix = logits_u_aug.clone()
    u_rand_index = torch.randperm(image_u_aug.size()[0])[:image_u_aug.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(image_u_aug.size(), lam=np.random.beta(4, 4))

    for i in range(0, image_cutmix.shape[0]):
        image_cutmix[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0

        label_cutmix[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0

        logits_cutmix[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 1.0

    return image_cutmix, label_cutmix, logits_cutmix


# # # # # # # # # # # # # # # # # # # # # 
# # 4 cutmix_with_label_inject  
# # # # # # # # # # # # # # # # # # # # #
def cut_mix_label_inject(unlabeled_image, unlabeled_mask, unlabeled_logits, 
        labeled_image, labeled_mask):
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 2) get box
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 3) labeled inject
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
        labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
        mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
        
        mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits 

# # # # # # # # # # # # # # # # # # # # # 
# # 5 cutmix_with_label_inject  
# # # # # # # # # # # # # # # # # # # # #
def cut_mix_slabel_inject(unlabeled_image, unlabeled_mask, unlabeled_logits, 
        labeled_image, labeled_mask):
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 2) get box
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))

    s_bbx1, s_bby1, s_bbx2, s_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 3) labeled inject
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]] = \
            labeled_image[u_rand_index[i], :, s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]]
    
        mix_unlabeled_target[i, s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]] = \
            labeled_mask[u_rand_index[i], s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]]
        
        mix_unlabeled_logits[i, s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]] = \
            labeled_logits[u_rand_index[i], s_bbx1[i]:s_bbx2[i], s_bby1[i]:s_bby2[i]]
        
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
        mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            unlabeled_image[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
        
        mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            unlabeled_image[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits 


def rotate(image, label, confidence):

    rotater = T.RandomRotation(degrees=(0, 180))
    # imgs = [rotater(img) for img in image]
    imgs = rotater(image)
    label = rotater(label)
    rotater_con = T.RandomRotation(degrees=(0, 180), fill=1.0)
    confidence = rotater_con(confidence)

    return imgs, label, confidence

def Perspective(image, label, confidence):

    perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
    # imgs = [rotater(img) for img in image]
    imgs = perspective_transformer(image)
    label = perspective_transformer(label)
    perspective_transformer_conf = T.RandomPerspective(distortion_scale=0.6, p=1.0, fill=1.0)
    confidence = perspective_transformer_conf(confidence)

    return imgs, label, confidence

def Affine(image, label, confidence):

    affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    # imgs = [rotater(img) for img in image]
    imgs = affine_transfomer(image)
    label = affine_transfomer(label)
    affine_transfomer_conf = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75), fill=1.0)
    confidence = affine_transfomer_conf(confidence)

    return imgs, label, confidence

def random_strong_aug(image_u_aug, pseudo_label, pseudo_confid, image_l,label_l, confidence, ramdom_num=1):
    ramdom_list =  random.choices([1, 3], k=ramdom_num)
    for i in ramdom_list:
        if i == 1:
            image_u_aug, pseudo_label, pseudo_confid = cut_mix_label_adaptive(
                            image_u_aug,
                            pseudo_label,
                            pseudo_confid, 
                            image_l,
                            label_l,
                            confidence
                        )
        if i == 2:
            image_u_aug, pseudo_label, pseudo_confid = cut_mix_label_inject(
                            image_u_aug,
                            pseudo_label,
                            pseudo_confid, 
                            image_l,
                            label_l
                        )
        if i == 3:
            image_u_aug, pseudo_label, pseudo_confid = cut_mix(image_u_aug, pseudo_label, pseudo_confid)
        if i == 4:
            image_u_aug, pseudo_label, pseudo_confid = cut_out(image_u_aug, pseudo_label, pseudo_confid)
        if i == 5:
            image_u_aug, pseudo_label, pseudo_confid = Affine(image_u_aug, pseudo_label, pseudo_confid)
        if i == 6:
            image_u_aug, pseudo_label, pseudo_confid = rotate(image_u_aug, pseudo_label, pseudo_confid)
        if i == 7:
            image_u_aug, pseudo_label, pseudo_confid = Perspective(image_u_aug, pseudo_label, pseudo_confid)

        return image_u_aug, pseudo_label, pseudo_confid