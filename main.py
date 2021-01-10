import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import lxml.etree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.misc

xml_folder = './Translation_Dataset'

#ORIG_IMG = '1530780845_visible.png'
#rot_img = cv2.imread('Rot.png', 1)
#img = cv2.imread('lena_caption.png', cv2.COLOR_BGR2RGB)


def crop_object(image_path, coords, name):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(name)
    return cropped_image

    
def get_object_coords(xml_path, img_path):
    # Parsing the XML tree
    tree = ET.parse(xml_path)
    img = Image.open(img_path)
    root = tree.getroot()

    df = pd.DataFrame({'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []})
    for country in root.findall('object/bndbox'):
        xmin = country.find('xmin').text
        ymin = country.find('ymin').text
        xmax = country.find('xmax').text
        ymax = country.find('ymax').text
        df = df.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}, ignore_index=True)
    df.astype(int)

    return df

def tf(index,item):
    #xml_path = xml_folder + '/' + item
        imname = item.split('/')[-1].split('.')[0]
        image_path = os.path.join(xml_folder, imname + '.png')
        obj_coords = get_object_coords(item, image_path)
        obj_list = obj_coords.values.tolist()
        orig_image = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

        result = orig_image
        for obj_index, obj in enumerate(obj_list):
            
            result = cv2.rectangle(result, (int(obj[1]), int(obj[3])), (int(obj[0]), int(obj[2])), (0,0,0), cv2.FILLED)
            name = os.path.join(xml_folder, 'cropped_images', 'cropped' + str(obj_index) + '.png')
            crop_img = crop_object(image_path, (int(obj[1]), int(obj[3]), int(obj[0]), int(obj[2])), name)
            crop_img = cv2.imread(name)
            rows, cols, ch = crop_img.shape
           
            orig_rows,orig_cols,orig_ch = orig_image.shape
            
            padded = cv2.copyMakeBorder(crop_img,((orig_rows-rows)/2) + 1,(orig_rows-rows)/2, (orig_cols-cols)/2,(orig_cols-cols)/2,0,0)
            #affine transformation code

            rotation_matrix = cv2.getRotationMatrix2D((rows/2, cols/2),90,1)
            img_rotation = cv2.warpAffine(crop_img, rotation_matrix, (rows, cols))
            rotation_matrix_pad = cv2.getRotationMatrix2D((rows/2+5, cols/2+5), 90,1)
            img_rotation_pad = cv2.warpAffine(crop_img, rotation_matrix, (rows, cols))

            # changed width
            del_h = img_rotation_pad.shape[0] - crop_img.shape[0]
            del_w = img_rotation_pad.shape[1] - crop_img.shape[1]
            
            
            #ymin
            pad_top = int(obj[3]) - (del_h) 
            #rows-ymax
            pad_bottom = orig_rows-int(obj[2]) 
            #cols-xmax
            pad_right= orig_cols-int(obj[0]) 
            #xmin
            pad_left= int(obj[1]) - (del_w) 
            
            
            padded_after_crop = cv2.copyMakeBorder(img_rotation_pad,pad_top, pad_bottom, pad_left, pad_right, 0, 0)
            
            
            plt.subplot(121),plt.imshow(crop_img),plt.title('Input')
            plt.subplot(121),plt.imshow(crop_img),plt.title('Input')
            plt.subplot(122),plt.imshow(img_rotation),plt.title('Output img_rotation')
            plt.show()

            plt.subplot(121),plt.imshow(padded_after_crop),plt.title('padded_after_crop')
            plt.subplot(122),plt.imshow(img_rotation_pad),plt.title('Output dst_pad')
            plt.show()
            
            
            result = result + padded_after_crop



if __name__ == '__main__':

    xml_lst = []
    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            xml_lst.append(os.path.join(xml_folder, file))


    for index,item in enumerate(xml_lst):
        tf(index,item)
        
        plt.subplot(122),plt.imshow(result),plt.title('result')
        plt.show()
        
        #save image in xml_folder
        #cv2.imwrite(xml_folder + str(index)+ '.png', result)
        scipy.misc.imsave(os.path.join(xml_folder,  imname + '_' +  'warped' + '.png'), result)
    
