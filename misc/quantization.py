import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# numpy is one of the most useful libraries for ML.
import numpy as np
def get_scale(x_min, x_max, b):
 # Compute scale as discussed.
 return (x_max - x_min ) * 1.0 / (2**b)
"""Quantizing the given vector x."""
def quantize(x, x_min, x_max, b):
 # Clamp x to lie in [x_min, x_max].
 x = np.minimum(x, x_max)
 x = np.maximum(x, x_min)
 # Compute scale as discussed.
 scale = get_scale(x_min, x_max, b)
 x_q = np.floor((x - x_min) / scale)
 # Clamping the quantized value to be less than (2^b - 1).
 x_q = np.minimum(x_q, 2**b - 1)
 # Return x_q as an unsigned integer.
 # uint8 is the smallest data type supported by numpy.
 return x_q.astype(np.uint8)

def dequantize(x_q, x_min, x_max, b):
 # Compute the value of scale the same way.
 s = get_scale(x_min, x_max, b)
 x = x_min + (s * x_q)
 return x

x = np.arange(-10.0, 10.0 + 1e-6, 2.5)
print("original array")
print(x)


# Quantize the entire array in one go.
x_q = quantize(x, -10.0, 10.0, 3)

print("quantized array")
print(x_q)

print("dequantized array")
de_x_q = dequantize(x_q, -10.0, 10.0, 3)

os.system("wget https://github.com/reddragon/book-codelabs/raw/main/pia23378-16.jpeg -O image.jpg")
img = (mpimg.imread('image.jpg') / 255.0)
plt.imshow(img)
def simulate_transmission(img, b):
 transmitted_image = quantize(img, 0.0, 1.0, b)
 decoded_image = dequantize(transmitted_image, 0.0, 1.0, b)
 plt.axis('off')
 plt.imshow(decoded_image)

for b in [ 2, 4, 8 ]:
 simulate_transmission(img, b)
