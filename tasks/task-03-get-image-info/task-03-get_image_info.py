import numpy as np
import cv2 as cv

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###
    ### TODO
    
    
    height = image.shape[0]
    width = image.shape[1]
    dtype = image.dtype
    
    depth = int(''.join(filter(str.isdigit, str(dtype))))

    if len(image.shape) == 3:
        #profundidade = número de bits * número de canais (rgb tem 3 canais..)
        depth *= image.shape[2]
    
    min_val = image.min()
    max_val = image.max()
    mean_val = image.mean()
    std_val = image.std()

    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")

#if sample_image is not None:
    #cv.imshow("Imagem", sample_image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()