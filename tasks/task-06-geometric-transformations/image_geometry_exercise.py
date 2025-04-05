# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale w by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import cv2

def translated(img, dist = (100, 100)):
    
    h, w = img.shape[:2]
    x_dist = dist[0]
    y_dist = dist[1]

    ts_mat = np.array([[1, 0, x_dist], [0, 1, y_dist]])

    out_img = np.zeros(img.shape, dtype='u1')

    for i in range(h):
        for j in range(w):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x, origin_y, 1])

            new_xy = np.dot(ts_mat, origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < w and 0 < new_y < h:
                out_img[new_y, new_x] = img[i, j]

    return out_img   

def rotate_image_90_clockwise(img):
    h = len(img)
    w = len(img[0])
    
    rotated = [[0] * h for _ in range(w)]

    for i in range(h):
        for j in range(w):
            rotated[j][h - 1 - i] = img[i][j]

    return np.array(rotated)

def stretch_image_horizontally(img, scale=1.5):
    stretched_img = []

    for row in img:
        new_w = int(len(row) * scale)
        new_row = []

        for i in range(new_w):
            # Mapeia a nova posição para a original (inverso da escala)
            orig_x = int(i / scale)
            new_row.append(row[orig_x])

        stretched_img.append(new_row)

    stretched_img = np.array(stretched_img)
    #print(stretched_img.shape)

    return stretched_img

def mirror_image_horizontally(img):    
    return np.array([row[::-1] for row in img])

def barrel_distortion(img, k=0.00001):
    h = len(img)
    w = len(img[0])
    cx = w / 2
    cy = h / 2

    distorted = []

    print(len(img.shape))

    

    return np.array(distorted)

def barrel_distortion_rgb(img, k=0.00001):
    h = len(img)
    w = len(img[0])
    cx = w / 2
    cy = h / 2

    if (len(img.shape) == 3):
        distorted = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]

        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                r = (dx**2 + dy**2)**0.5

                factor = 1 + k * (r**2)

                src_x = int(cx + dx / factor)
                src_y = int(cy + dy / factor)

                if 0 <= src_x < w and 0 <= src_y < h:
                    distorted[y][x] = img[src_y][src_x]
                else:
                    distorted[y][x] = [0, 0, 0]

        return np.array(distorted)
    
    else:
        #print("teste")
        distorted = [[0 for _ in range(w)] for _ in range(h)]

        for y in range(h):
            for x in range(w):
                
                dx = x - cx
                dy = y - cy
                r = (dx**2 + dy**2)**0.5

                factor = 1 + k * (r**2)

                src_x = int(cx + dx / factor)
                src_y = int(cy + dy / factor)

                if 0 <= src_x < w and 0 <= src_y < h:
                    distorted[y][x] = img[src_y][src_x]
                else:
                    distorted[y][x] = 0 

        return np.array(distorted)


def apply_geometric_transformations(img: np.ndarray) -> dict:
    
    result = {
        "translated": translated(img, (50, 100)),
        "rotated": rotate_image_90_clockwise(img),
        "stretched": stretch_image_horizontally(img),
        "mirrored": mirror_image_horizontally(img),
        "distorted": barrel_distortion_rgb(img)
    }

    #for key, value in result.items():
    #    cv2.imshow(key, value)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result

    pass

#img = cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)
#apply_geometric_transformations(img)