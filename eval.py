import torch
import numpy as np
from skimage.measure import compare_psnr,compare_ssim
import cv2


class ImageEval(object):
    def __init__(self,out_img,gt_img):
        # tensor cpu
        out_img[out_img>1]=1.0
        out_img[out_img<0]=0.0
        gt_img[gt_img>1]=1.0
        gt_img[gt_img<0]=0.0

        self.out_img=out_img*255
        self.gt_img=gt_img*255

        self.out_img_np=self.out_img.detach().numpy()
        self.gt_img_np=self.gt_img.detach().numpy()

    def compute_PSNR(self):
        self.psnr=compare_psnr(self.gt_img_np,self.out_img_np,data_range=255)


    def compute_SSIM(self):

        gray_out=cv2.cvtColor(self.out_img_np.transpose(1,2,0),cv2.COLOR_RGB2GRAY)
        gray_gt=cv2.cvtColor(self.gt_img_np.transpose(1,2,0),cv2.COLOR_RGB2GRAY)

        (score,diff)=compare_ssim(gray_gt,gray_out,full=True,data_range=255)
        self.ssim=score
        # return self.ssim

    def eval(self):
        self.compute_PSNR()
        self.compute_SSIM()
        self.print_eval()
        return self.psnr,self.ssim

    def print_eval(self):
        print('PSNR:%.2f'%(self.psnr))
        print('SSIM:%.2f'%(self.ssim))
