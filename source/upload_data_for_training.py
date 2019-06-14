from api_for_data_upload import *

def prepare_data_for_training(folder_name):
    starting_images = open_images(folder_name)
    
    sliced_images = [slice_image(image) for image in starting_images]
    sliced_images = [image for subimages in sliced_images for image in subimages]
    
    converted_images = [ {
                          "img_data": convert_image(image["img"]),
                          "count": vectorized_number(image["count"]) #mozda vectorized
                         } for image in sliced_images ]
    
    return converted_images

if __name__ == "__main__":
    #input_set = prepare_data_for_training("../images/training_images/")
    #train_cnn(input_set)
    
    #print(a[0])
    #imgs = open_images("../images/training_images/")
    #img_dict = open_image_and_table("../images/training_images/ED4.3 1.6 merge track.tif")
    #subimages = slice_image(img_dict)
    #print(convert_image(subimages[5]["img"]))
    #counts = [img["count"] for img in subimages]
    #print(max(counts))
    #regions = slice_image(imgs[2])
    
    
    
    




        