from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
import os
import glob


def open_images(path):
    """Opens all images and associated tables from given folder."""    
    return [open_image_and_table(filename) for filename in 
            glob.glob(os.path.join(path, '*.tif'))]     

def open_image_and_table(filename): 
    """Opens file containing image and associated file containing locations of
    infected dots."""
    image = Image.open(filename)
    #image.filter(ImageFilter.SHARPEN)
    
    csv_filename = filename.replace(' merge track.tif', '.csv')
    locations = read_locations(csv_filename)
    
    return {"img": image, "loc": locations, "name" : filename}

def read_locations(filename):
    """Reads csv file containing locations of infected dots."""
    df = pd.read_csv(filename, sep = "\t")
    if "X" not in df: df = pd.read_csv(filename, sep = ",")
    
    xs = np.array(df["X"].tolist())
    ys = np.array(df["Y"].tolist())
    mean = np.array(df["Mean"].tolist())
    
    return {"X" : xs, "Y": ys, "Mean": mean}

     

def slice_image(img_dict, width=85, height=64):
    """Slices given image to subimages of given width and size. Returns array 
    of dictionaries: first element of dict is subimage, second is number of
    infected dots in image"""
    img_tif = img_dict["img"]
    img_loc = img_dict["loc"]
    name = img_dict["name"]
    
    noWidth = img_tif.size[0]//width
    noHeight = img_tif.size[1]//height
    
    subimages_with_count = [ 
                {"img": img_tif.crop(box), "count": count(img_loc, box)}
                for box in 
                [ [i*width, j*height, (i+1)*width, (j+1)*height] 
                  for i in range(noWidth) for j in range(noHeight) ] 
            ]
              
    return subimages_with_count


def count(locations, box = [0, 0, 1360, 1024]):
    """Counts number of infected dots from given coordinates of dots and 
    box (part of image, specified with [blx, bly, urx, ury])"""
    xs = locations["X"]
    ys = locations["Y"]
    
    count = 0
    for x, y in zip(xs, ys):
        if(box[0] <= x < box[2] and box[1] <= y < box[3]):
            count+=1
            
    return count

def convert_image(img):
    """Converts given PIL image object to numpy array of type height x width x rgb"""
    data = np.asarray(img)#; print(data.shape)
    data = data/255
    #np.delete(data, 3, axis=1)
    return data

def vectorized_number(count):
    """Produces vector having zeros everywhere, except at
    given index (parameter 'count' \in [0, 5]), where it is 1"""
    e = np.zeros((6, 1))
    e[count] = 1.0
    return e

#if __name__ == "__main__":
    #imgs = open_images("../Danica slike/ED skupina/ED4.14/")
    #regions = slice_image(imgs[6])
    
    #data = convert_image(regions[543]["tif"])
    #read_locations('../Danica slike/ED skupina/ED4.2/ED4.2 1.1.xls')
    




        
