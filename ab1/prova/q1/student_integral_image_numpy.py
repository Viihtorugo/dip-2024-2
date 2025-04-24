# student_integral_image_numpy.py

import numpy as np

def sum_s_image(n, m, image):
    total = 0

    for i in range (0, n + 1):
        for j in range(0, m + 1):
            total += image[i,j]

    #print(total)

    return total

def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """
    Computes the integral image using NumPy.

    Parameters:
        image (np.ndarray): 2D grayscale image.

    Returns:
        np.ndarray: Integral image.
    """
    # TODO: Implement your solution here
    img_res = np.zeros((3,3), dtype=np.float64)

    for y in range(len(image)):
        for x in range (len(image[0])):
            img_res[y, x] = sum_s_image(y, x, image)

    return img_res

# Define test image
image = np.array([
    [0.32285394, 0.95322289, 0.31806831],
    [0.12936134, 0.45275244, 0.60094833],
    [0.71811803, 0.49059312, 0.38843348],
], dtype=np.float64)

expected_result = np.array([
    [0.32285394, 1.27607683, 1.59414514],
    [0.45221528, 1.85819061, 2.77720725],
    [1.17033331, 3.06690176, 4.37435188],
], dtype=np.float64)

integral_image = compute_integral_image(image)

if (expected_result == integral_image).all():
    print(f'Success!')

print(f'Original image: \n{image}\n')
print(f'Expected result: \n{expected_result}\n')
print(f'Your result: \n{integral_image}\n')