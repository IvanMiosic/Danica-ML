from PIL import Image

def open_image(filename):
    return Image.open(filename)
     
def slice_image(img, X=5, Y=5):
    size = [img.size[0]/X, img.size[1]/Y]
    
    regions = [ img.crop(box) for box in 
               [ [i*size[0], j*size[1], (i+1)*size[0], (j+1)*size[1]] 
                  for i,j in zip(range(X), range(Y)) ] 
              ]
    return regions


    




        