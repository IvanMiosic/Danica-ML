from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
import os
import glob


def open_images(path):    
    return [open_image_and_table(filename) for filename in 
            glob.glob(os.path.join(path, '*.tif'))]     

def open_image_and_table(filename):   
    image = Image.open(filename)
    image.filter(ImageFilter.SHARPEN)
    
    csv_filename = filename.replace(' merge track.tif', '.csv')
    locations = read_locations(csv_filename)
    
    return {"img": image, "loc": locations}

def read_locations(filename):
    df = pd.read_csv(filename, sep = "\t")
    
    xs = np.array(df["X"].tolist())
    ys = np.array(df["Y"].tolist())
    mean = np.array(df["Mean"].tolist())
    
    return {"X" : xs, "Y": ys, "Mean": mean}

     

def slice_image(img_dict, width=85, height=64):
    img_tif = img_dict["img"]
    img_loc = img_dict["loc"]
    
    noWidth = int(img_tif.size[0]/width)
    noHeight = int(img_tif.size[1]/height)
    
    subimages_with_count = [ 
                {"img": img_tif.crop(box), "count": count(img_loc, box)}
                for box in 
                [ [i*width, j*height, (i+1)*width, (j+1)*height] 
                  for i in range(noWidth) for j in range(noHeight) ] 
            ]
              
    return subimages_with_count


def count(locations, box = [0, 0, 1360, 1024]):
    xs = locations["X"]
    ys = locations["Y"]
    
    count = 0
    for x, y in zip(xs, ys):
        if(box[0] <= x < box[2] and box[1] <= y < box[3]):
            count+=1
            
    return count

def convert_image(img):
    """Converts given PIL image object to numpy array of type height x width x rgb"""
    data = np.asarray(img)
    data = data/255
    #np.delete(data, 3, axis=1)
    return data

def vectorized_number(count):
    e = np.zeros((6, 1))
    e[count] = 1.0
    return e

#if __name__ == "__main__":
    #imgs = open_images("../Danica slike/ED skupina/ED4.14/")
    #regions = slice_image(imgs[6])
    
    #data = convert_image(regions[543]["tif"])
    #read_locations('../Danica slike/ED skupina/ED4.2/ED4.2 1.1.xls')
    




        