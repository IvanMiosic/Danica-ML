from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
import os
import glob


def open_images(path):      #vise api za _treniranje_
    arr_images = [open_image(filename) for filename in 
                  glob.glob(os.path.join(path, '*.tif'))]        
    return arr_images

def open_image(filename):   #vise api za testiranje
    tif_filename = filename
    xls_filename = filename.replace(' merge track.tif', '.csv')
    return {"tif": Image.open(tif_filename), "loc": read_locations(xls_filename)}

def read_locations(filename):
    df = pd.read_csv(filename)
    xs = np.array(df['X'].tolist())
    ys = np.array(df["Y"].tolist())
    mean = np.array(df["Mean"].tolist())
    return {"X" : xs, "Y": ys, "Mean": mean}

     
def slice_image(img_dict, width=40, height=64):
    img_tif = img_dict["tif"]
    img_loc = img_dict["loc"]
    
    noWidth = int(img_tif.size[0]/width)
    noHeight = int(img_tif.size[1]/height)
    regions = [ {"tif": img_tif.crop(box), "count": count(img_loc, box)} for box in 
              [ [i*width, j*height, (i+1)*width, (j+1)*height] 
                  for i in range(noWidth) for j in range(noHeight) ] 
              ]
    return regions


def count(locations, box):
    xs = locations["X"]
    ys = locations["Y"]
    
    count = 0
    for x, y in zip(xs, ys):
        if(box[0] <= x < box[2] and box[1] <= y < box[3]):
            count+=1
    return count

def convert_image(img):
    img.filter(ImageFilter.SHARPEN)
    data = np.array(img)
    data = data/255
    np.delete(data, 3, axis=1)
    return data

if __name__ == "__main__":
    imgs = open_images("../Danica slike/ED skupina/ED4.14/")
    regions = slice_image(imgs[6])
    
    #data = convert_image(regions[543]["tif"])
    #read_locations('../Danica slike/ED skupina/ED4.2/ED4.2 1.1.xls')
    




        