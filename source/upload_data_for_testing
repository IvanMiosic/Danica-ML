def prepare_data_for_testing(folder_name):
    starting_images = open_images(folder_name)
    expected_results = [count(image["loc"]) for image in starting_images]
    
    sliced_images = [slice_image(image) for image in starting_images]
    
    converted_images = [ [ 
                         {
                          "img_data": convert_image(image["img"]),
                          "count": vectorized_number(image["count"]) #mozda vectorized
                         } for image in subimages ]
                        for subimages in sliced_images ]

    return expected_results, converted_images


def compare_results():
    expected_results, input_set = prepare_data_for_testing("../images/testing_images/")
    
    cnn_results = [sum([apply_cnn(input_data) for input_data in image]) 
                   for image in input_set]
        
    for cnn_result, expected_result in zip(cnn_results, expected_results):
        print({1} + " vs " + {2}).format(cnn_result, expected_result) 
        
#if __name__ == "__main__":
    