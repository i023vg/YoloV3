import os
import cv2
import numpy as np
from PIL import Image
from natsort import natsorted

class ImagePaste(object):

    def __init__(self,OriginalImg_dir,PasteImg_dir,Output_dir):
        #書き込み用のディレクトリ
        self.output_dir = Output_dir
        #pasteの元画像のディレクトリ
        self.originalimg_dir = OriginalImg_dir
        #paste画像のディレクトリ
        self.pasteimg_dir = PasteImg_dir

    def Blend(self,blend_rate=0.5):
        self.blend_rate = blend_rate
        blend_filename = natsorted(os.listdir(self.pasteimg_dir))
        for i,filename in enumerate(blend_filename):
            l_imgname = natsorted(os.listdir(self.originalimg_dir + filename))
            m_imgname = natsorted(os.listdir(self.pasteimg_dir + filename))
            for j, f in enumerate(l_imgname):
                print(i,'\t',j,'\t',f)
                im1 = Image.open(self.originalimg_dir + filename + '/' + f)
                im1 = im1.convert('RGBA')
                im2 = Image.open(self.pasteimg_dir + filename + '/' + m_imgname[j])
                im2 = im2.convert('RGBA')
                img_blend = Image.blend(im1,im2,self.blend_rate)
                img_blend = img_blend.convert('RGB')
                img_blend.save(self.output_dir + f,quality=95)


    def Paste(self,Load_FileName):
        self.load_filename = Load_FileName
        back_file = natsorted(os.listdir(self.originalimg_dir)) 
        paste_file = natsorted(os.listdir(self.pasteimg_dir))
        crop_coordinate = np.load(self.load_filename, allow_pickle = True)
        b=0
        p=0
        mark = 0
        while True:
            back = back_file[b]
            paste = paste_file[p]
    
            if back[:-4]==paste[:-11]:
                if mark >= 1:
                    im1 = Image.open(self.output_dir + back)
                    back_im = im1.copy()
                    im2 = Image.open(self.pasteimg_dir + paste)
                    back_im.paste(im2,(int(crop_coordinate[p][0]),int(crop_coordinate[p][1])))
                    back_im.save(self.output_dir + back,quality=95)
                else:
                    im1 = Image.open(self.originalimg_dir + back)
                    back_im = im1.copy()
                    im2 = Image.open(self.pasteimg_dir + paste)
                    back_im.paste(im2,(int(crop_coordinate[p][0]),int(crop_coordinate[p][1])))
                    back_im.save(self.output_dir + back,quality=95)
        
                p += 1
                mark += 1
            else:
                b += 1
                mark = 0

            if p==len(paste_file):
                break





            
        
