# Improves our style transfer image.

import cv2
from argparse import ArgumentParser

from model import resolve_single
from model.edsr import edsr
from model.wdsr import wdsr_b
from model.srgan import generator

from utils import plot_sample


# MARK: 1. Parse arguments
parser = ArgumentParser()
parser.add_argument("--method", type=str, default="SRGAN")    # EDSR, WDSR, SRGAN
parser.add_argument("--imageName", type=str, default="../demo/1.png")
arguments = parser.parse_args()
print("Arguments:", arguments)

superResolutionMethod = arguments.method
imageName = arguments.imageName


# MARK: 2. Denoise
image = cv2.imread(imageName)    # This is also a numpy ndarray
# print(type(image), image.shape)

denoisedImage = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 10, 15)


# MARK: 3. Super resolution
if (superResolutionMethod == "EDSR"):
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights("../weights/edsr-16-x4/weights.h5")
elif (superResolutionMethod == "WDSR"):
    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights('../weights/wdsr-b-32-x4/weights.h5')
elif (superResolutionMethod == "SRGAN"):
    model = generator()
    model.load_weights('../weights/srgan/gan_generator.h5')

superResolutionImage = resolve_single(model, denoisedImage).numpy()


# MARK: 4. Plot images
# cv2.imshow("Source", image)
# cv2.imshow("Denoise", denoisedImage)
# cv2.imshow("Super", superResolutionImage)
# cv2.waitKey(0)
cv2.imwrite("output.png", superResolutionImage)
