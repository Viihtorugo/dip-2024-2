
import cv2 as cv
import numpy as np

def rotation_with_angle(img, angle = 45):

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    m = cv.getRotationMatrix2D(center, angle, 1.0)

    img_rotation = cv.warpAffine(img, m, (w, h))

    return img_rotation

def verify_laplacian_isotropy(image_path, angle=45):
    """
    Aplica rotação + Laplaciano e Laplaciano + rotação na imagem de entrada
    e retorna o coeficiente de correlação de Pearson entre os dois resultados.

    Parâmetros:
        image_path (str): Caminho para a imagem em tons de cinza.
        angle (float): Ângulo de rotação em graus.

    Retorno:
        float: Coeficiente de correlação de Pearson entre as duas imagens.
    """
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # TODO: Implementar solução aqui    
    img1 = rotation_with_angle(image, angle)
    img2 = cv.Laplacian(image, cv.CV_64F, ksize=3)

    result_1 = cv.Laplacian(img1, cv.CV_64F, ksize=3)
    result_2 = rotation_with_angle(img2, angle)

    # Flatten and compute correlation
    corr = np.corrcoef(result_1.flatten(), result_2.flatten())[0, 1]
    print(f'Correlation coefficient: {corr}')
    return corr

    # Flatten and compute correlation
    corr = np.corrcoef(result_1.flatten(), result_2.flatten())[0, 1]
    print(f'Correlation coefficient: {corr}')
    return corr

verify_laplacian_isotropy('example_image.png', angle=45)
# Expected: Correlation coefficient: 0.9621492663463063

verify_laplacian_isotropy('checkerboard_image.png', angle=45)
# Expected: Correlation coefficient: 0.9383739474511829