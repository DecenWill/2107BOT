#coding=utf8

import os
from libtiff import TIFF  
from scipy import misc  
  
##tiff文件解析成图像序列  
##tiff_image_name: tiff文件名；  
##out_folder：保存图像序列的文件夹  
##out_type：保存图像的类型，如.jpg、.png、.bmp等  
def tiff_to_image_array(tiff_image_name, image_name, out_folder, out_type):   
            
    tif = TIFF.open(tiff_image_name, mode = "r")  
    idx = 0  
    for im in list(tif.iter_images()):  
        #  
        im_name = out_folder + image_name[:-5] + out_type  
        misc.imsave(im_name, im)  
        print im_name, 'successfully saved!!!'  
        idx = idx + 1  
    return  
  
##图像序列保存成tiff文件  
##image_dir：图像序列所在文件夹  
##file_name：要保存的tiff文件名  
##image_type:图像序列的类型  
##image_num:要保存的图像数目  
def image_array_to_tiff(image_dir, file_name, image_type, image_num):  
  
    out_tiff = TIFF.open(file_name, mode = 'w')  
      
    #这里假定图像名按序号排列  
    for i in range(0, image_num):  
        image_name = image_dir + str(i) + image_type  
        image_array = Image.open(image_name)  
        #缩放成统一尺寸  
        img = image_array.resize((480, 480), Image.ANTIALIAS)  
        out_tiff.write_image(img, compression = None, write_rgb = True)  
          
    out_tiff.close()  
    return

#source_path = '../data/cancer_subset00'
#source_path = '../data/non_cancer_subset00'
source_path = '../../2017final_data/positive'

for filename in os.listdir(source_path):
    tiff_to_image_array(source_path + '/' + filename, filename, '../../2017final_data/positive_png/', '.png')
