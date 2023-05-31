import os
from glob import glob #extract path of each file.
import pandas as pd #data preprocessing
from xml.etree import ElementTree as et
from functools import reduce
from shutil import move
import warnings
warnings.filterwarnings("ignore")
import sys
from src.exception import ObjectDetectionException
from src.logger import logging


class DataPreprocess :
    def __init__() :
        pass
    
    def preprocess():
        "Preprocesses all data - convert files from XML to text format"
        try:
            # get path of each xml file.
            xmlfiles = glob("src/data_images/*.xml")
            # replace double backward slash(\\) with forward slash(/)
            xmlfiles = [i.replace("\\","/") for i in xmlfiles]
            
            # read xml files
            # from each xml file we need to extract
            # filename, size(width, height), object(name, xmin, xmax, ymin, ymax)
            
            def extract_text(filename):   
                tree= et.parse(filename)
                root= tree.getroot()
                
                # extract filename
                image_name = root.find('filename').text
                # extract width and height of the image.
                width = root.find("size").find("width").text
                height = root.find("size").find("height").text
                # for object information
                objs = root.findall('object')
                parser =[]
                for obj in objs:
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    xmin = bndbox.find("xmin").text
                    xmax = bndbox.find("ymin").text
                    ymin = bndbox.find("xmax").text
                    ymax = bndbox.find("ymax").text
                    parser.append([image_name,width,height,name,xmin,xmax,ymin,ymax])

                return parser
            
            parser_all = list(map(extract_text,xmlfiles))
            data = reduce(lambda x,y : x+y, parser_all)
            df = pd.DataFrame(data, columns=['filename', 'width', 'height', 'name', 'xmin','xmax', 'ymin', 'ymax'])
            
            # datatype conversion
            cols = ['width','height','xmin', 'xmax', 'ymin', 'ymax']
            df[cols]= df[cols].astype(int)
            
            # calculate for center x, center y each bounding box.
            df['centre_x']= (df['xmax'] + df['xmin'])/2/df['width']  
            df['centre_y']= (df['ymax'] + df['ymin'])/2/df['height']
            #W
            df['w']=(df['xmax'] - df['xmin'])/df['width'] 
            #H
            df['h']= (df['ymax'] - df['ymin'])/df['height']
            
            # split the data into train and test.
            # 80% train and 20% test.
            images = df['filename'].unique()
            img_df = pd.DataFrame(images, columns=['filename'])
            img_train = tuple(img_df.sample(frac=0.8)['filename']) #shuffle and pick 80% of images
            img_test = tuple(img_df.query(f'filename not in {img_train}')['filename']) #take rest 20% images

            train_df= df.query(f"filename in {img_train}")
            test_df = df.query(f"filename in {img_test}")
            
            # Assign ID number to object names 
            # Label encoding
            def label_encoding(x):
                labels = {'person':0, 'car':1, 'chair':2, 'bottle':3, 'pottedplant':4, 'bird':5, 'dog':6,
                'sofa':7, 'bicycle':8, 'horse':9, 'boat':10, 'motorbike':11, 'cat':12, 'tvmonitor':13,
                'cow':14, 'sheep':15, 'aeroplane':16, 'train':17, 'diningtable':18, 'bus':19}
                return labels[x]
            
            train_df['id']= train_df['name'].apply(label_encoding)
            test_df['id']= test_df['name'].apply(label_encoding)
            
            # Save Image and Labels in text 
            train_folder = 'src/model_training/yolov5/data_images/train'
            test_folder = 'src/model_training/yolov5/data_images/test'
            
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder,exist_ok = True)
            
            # Groupby filename, we will get information of all bounding boxes for that file/image in single file
            cols = ['filename', 'id', 'centre_x', 'centre_y', 'w', 'h']
            groupby_obj_train = train_df[cols].groupby('filename')
            groupby_obj_test = test_df[cols].groupby("filename")
            
            # Save each image in train/test folder and respective labels in .txt
            def save_data(filename, folder_path, groupby_obj):
                #how to move images
                src = os.path.join("data_images",filename)
                dst = os.path.join(folder_path,filename) #dst is basically from folder path
                move(src,dst) #move images to dst folder
                
                # save the labels
                text_filename_split = os.path.splitext(filename)[0] + ".txt" #split out the filename and the extension
                text_filename= os.path.join(folder_path,text_filename_split)
                groupby_obj.get_group(filename).set_index('filename').to_csv(text_filename,sep=" ",index=False, header=False)

            filename_series = pd.Series(groupby_obj_train.groups.keys())
            filename_series.apply(save_data,args=(train_folder,groupby_obj_train))
            
            filename_series_test = pd.Series(groupby_obj_test.groups.keys())
            filename_series_test.apply(save_data,args=(test_folder,groupby_obj_test))
            logging.info("preprocessing completed")
            
        except Exception as e:
            raise ObjectDetectionException(e,sys)
            
            
            
            
            
            
            
            
