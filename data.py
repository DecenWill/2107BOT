#coding=utf8
from __future__ import print_function

import os
import numpy as np

#from skimage.io import imsave, imread, imshow 
import cv2
from cv2 import imwrite, imread, imshow
from libtiff import TIFF  
from scipy import misc 

data_path = '../../2017final_data/'

image_rows = 2048
image_cols = 2048

tiff_path = '../../2017final_data/negative'


def create_train_data():
    train_data_path = os.path.join(data_path, 'train_png_pre')
    train_mask_path = os.path.join(data_path, 'maskall_png_pre')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        image_mask_name = image_name.split('.')[0] + '.png'
        img = imread(os.path.join(train_data_path, image_name), 0)
        img_mask = imread(os.path.join(train_mask_path, image_mask_name), 0)
        #print(os.path.join(train_data_path, image_name))
        #print(np.max(img_mask))
        #print(img.shape)
        #assert 1==2
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])

        img = imread(os.path.join(train_data_path, image_name), 0)
        #print(np.min(img))
        #assert 1==2
        img = np.array([img])
        imgs[i] = img
        imgs_id[i] = img_id
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

def create_negative_mask(source_path, mask_path):
    images = os.listdir(source_path)
    i = 0
    print('-'*30)
    print('Creating negative mask images...')
    print('-'*30)
    total = len(images)
    for image_name in images:
        img = imread(os.path.join(source_path, image_name), 0)
        #print(image_name)
        #assert 1==2
        img[:,:] = 0
        imwrite(os.path.join(mask_path, image_name),img)       
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('create done.')

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
        print(im_name, 'successfully saved!!!')  
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

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def NegaAugmentation(path):
    datagen = ImageDataGenerator(
        rotation_range=360,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='wrap')
    for pic in os.listdir(path):
        img = load_img(path + '/' + pic)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,save_to_dir=path, 
            save_prefix=pic[:-4], save_format='png'):
            i += 1
            if i > 2:
                break

def PosiAugmentation(source_image_path, image_path, source_mask_path, mask_path):
    images = os.listdir(source_image_path)
    
    angle = 90
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_90 = cv2.warpAffine(image_ori,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name), image_ori) 
        imwrite(os.path.join(image_path, image_name[0:-4] + '_' + str(angle) + '.png'), image_90)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_90 = cv2.warpAffine(mask_ori,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name), mask_ori) 
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_' + str(angle) + '.png'), mask_90)   

    angle = 180
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_180 = cv2.warpAffine(image_ori,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name[0:-4] + '_' + str(angle) + '.png'), image_180)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_180 = cv2.warpAffine(mask_ori,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_' + str(angle) + '.png'), mask_180)

    angle = 270
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_270 = cv2.warpAffine(image_ori,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name[0:-4] + '_' + str(angle) + '.png'), image_270)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_270 = cv2.warpAffine(mask_ori,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_' + str(angle) + '.png'), mask_270)

    angle = 90
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_flipped = cv2.flip(image_ori, 1)
        image_flipped_90 = cv2.warpAffine(image_flipped,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name[0:-4] + '_flip.png'), image_flipped) 
        imwrite(os.path.join(image_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), image_flipped_90)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_flipped = cv2.flip(mask_ori, 1)
        mask_flipped_90 = cv2.warpAffine(mask_flipped,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_flip.png'), mask_flipped) 
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), mask_flipped_90)   

    angle = 180
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_flipped = cv2.flip(image_ori, 1)
        image_flipped_180 = cv2.warpAffine(image_flipped,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), image_flipped_180)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_flipped = cv2.flip(mask_ori, 1)
        mask_flipped_180 = cv2.warpAffine(mask_flipped,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), mask_flipped_180)  

    angle = 270
    M = cv2.getRotationMatrix2D((2048/2,2048/2),angle,1)
    for image_name in images:
        image_ori = imread(os.path.join(source_image_path,image_name))
        image_flipped = cv2.flip(image_ori, 1)
        image_flipped_270 = cv2.warpAffine(image_flipped,M,(2048,2048))
        imwrite(os.path.join(image_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), image_flipped_270)   
        
        mask_ori = imread(os.path.join(source_mask_path,image_name))
        mask_flipped = cv2.flip(mask_ori, 1)
        mask_flipped_270 = cv2.warpAffine(mask_flipped,M,(2048,2048))
        imwrite(os.path.join(mask_path, image_name[0:-4] + '_flip_' + str(angle) + '.png'), mask_flipped_270)  

if __name__ == '__main__':
    
    #create_train_data()
    create_test_data()
    
    #source_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/temp'
    #mask_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/maskall_png_pre'
    #create_negative_mask(source_path, mask_path)
    
    #for filename in os.listdir(tiff_path):
        #tiff_to_image_array(tiff_path + '/' + filename, filename, '../../2017final_data/negative_png/', '.png')

    #NegaAugmentation('/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/temp')
    #source_image_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/positive_png'
    #image_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/positive_png_argu'
    #source_mask_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/posi_mask'
    #mask_path = '/home/ubuntu-lee/WangXD/2017BOT_TIA/final/2017final_data/posi_mask_argu'
    #PosiAugmentation(source_image_path, image_path, source_mask_path, mask_path)
