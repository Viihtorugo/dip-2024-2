import argparse
import numpy as np
import cv2 as cv
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    ### TODO

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, stream=True)
    
    #print(response) #verificando o codigo do resquest
    
    if response.status_code == 200:
        
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        flags = kwargs.get("flags", cv.IMREAD_COLOR)
        image = cv.imdecode(image_array, flags)
        
        #print(type(image)) # verificando o tipo se é numpy

        return image
    else:
        return None

    ### END CODE HERE ###
    
def main():

    parser = argparse.ArgumentParser(description="Carregar a imagem com url")

    parser.add_argument("url", type=str, help="URL of the image to load")
    parser.add_argument("--grayscale", action="store_true", help="Load the image in grayscale mode")
    args = parser.parse_args()

    args = parser.parse_args()

    flags = cv.IMREAD_GRAYSCALE if args.grayscale else cv.IMREAD_COLOR
    image = load_image_from_url(args.url, flags=flags)
    
    if image is not None:
        print("Imagem carregada com sucesso!")
        #cv.imshow("Imagem", image) # para mostrar imagem na máquina local
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    else:
        print("Erro ao tentar acessar a url!")

if __name__ == "__main__":
    main()