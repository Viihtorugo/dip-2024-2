import cv2 as cv
import numpy as np

def equalize_histogram(image_path: str) -> np.ndarray:
    """
    Realiza equalização de histograma apenas no canal Y de uma imagem RGB convertida para YCrCb.

    Parâmetros:
        image_path (str): Caminho para a imagem RGB.

    Retorno:
        np.ndarray: Imagem RGB com o canal Y equalizado.
    """
    # TODO: Implemente sua solução aqui

    img = cv.imread(image_path, cv.COLOR_BGR2RGB)
    
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_ycrcb)
    y_equalized = cv.equalizeHist(y)
    img_ycrcb_equalized = cv.merge([y_equalized, cr, cb])
    img_rgb_ycrcb_equalized = cv.cvtColor(img_ycrcb_equalized, cv.COLOR_YCrCb2RGB)

    return img_rgb_ycrcb_equalized


student_result = equalize_histogram('unequal_lighting_color_image.png')
expected_result = cv.imread('expected_result.png')

if (expected_result == student_result).all():
    print('Success!')