#Jehad Hamayel
#1200348
import random
from PIL import Image
from IPython.display import display
import cv2
import numpy as np
from scipy.ndimage import uniform_filter , median_filter
from matplotlib import pyplot as plt
#Open the image
path = 'tank.jpg'
image = Image.open(path)
# Convert the image to 8-bit gray-level
GrayLevelImage = image.convert("L")
# Save and display the 8-bit gray-level image
GrayLevelImage.save("tankBW.jpg")
pathGrayLevelImage = 'tankBW.jpg'
imageGrayLevelImage_original = Image.open(pathGrayLevelImage)
imageGrayLevelImage_original.show()
display(imageGrayLevelImage_original)
#Applying a power law transform with gamma = 0.4 to the image
image=cv2.imread(pathGrayLevelImage)
r=image/255
c=255
gamma=0.4
#power law transformation
s=np.array(c*(r**gamma),dtype='uint8')
# Save or display the resulting image with power law transform
cv2.imwrite('imageWithGamma0.4.jpg',s)
imageWithGamma = Image.open('imageWithGamma0.4.jpg')
imageWithGamma.show()
display(imageWithGamma)

# Add zero-mean Gaussian noise with variance = 40 gray-levels
imageGrayLevelImageArray = np.array(imageGrayLevelImage_original, dtype=np.float32)
mean=0
std = 40
segma= std ** 0.5
h,w=imageGrayLevelImageArray.shape
# Add noise to the image
noisy_image_array = imageGrayLevelImageArray + np.random.normal(mean, segma,(h,w))
# Clip pixel values to the valid range [0, 255]
noisyImageArrayClipped = np.clip(noisy_image_array, 0, 255)
# Convert the NumPy array back to an image
noisyImageWithZeroMeanGaussianNoise = Image.fromarray(np.uint8(noisyImageArrayClipped))
# Save or display the resulting image with Gaussian noise
noisyImageWithZeroMeanGaussianNoise.show()
noisyImageWithZeroMeanGaussianNoise.save("noisyImageWithZeroMeanGaussianNoise.jpg")

#Applying a 5 by 5 mean filter to the Gaussian noisy-image above and show the result.
imageGrayLevelImageArray_MeanFilter = np.array(noisyImageWithZeroMeanGaussianNoise,dtype=np.uint8)
MeanFilterArray = uniform_filter(imageGrayLevelImageArray_MeanFilter,size=5)# 5 by 5 mean filter
MeanFilterImage = Image.fromarray(np.uint8(MeanFilterArray))
MeanFilterImage.show()
display(MeanFilterImage)
MeanFilterImage.save("MeanFilterImage.jpg")

#Adding salt and pepper noise (noise-density=0.1) to the original image and then apply a 7 by 7 median filter to the noisy-image and show both images
imageGrayLevelImage_originalArray=np.array(imageGrayLevelImage_original, dtype=np.uint8)
density=0.1
salt_and_pepper_noisy_imageArray = np.copy(imageGrayLevelImage_originalArray)
rows , colums = salt_and_pepper_noisy_imageArray.shape
numberOfPixels = int(((salt_and_pepper_noisy_imageArray.size * density) + 1)/2)
#Adding salt and pepper noise (noise-density=0.1) to the original image
for i in range(numberOfPixels):
    # Pick a random y coordinate from the original image
    y_Coordinate = random.randint(0, rows - 1)
    # Pick a random x coordinate from the original image
    x_Coordinate = random.randint(0, colums - 1)
    # Color that pixel to white(salt)
    if salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] != 0 and \
            salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] != 255:
        salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] = 255
    else:
        i -= 1

for i in range(numberOfPixels):
    # Pick a random y coordinate from the original image
    y_Coordinate = random.randint(0, rows - 1)

    # Pick a random x coordinate from the original image
    x_Coordinate = random.randint(0, colums - 1)

    # Color that pixel to black(pepper)
    if salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] != 0 and \
            salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] != 255:
        salt_and_pepper_noisy_imageArray[y_Coordinate][x_Coordinate] = 0
    else:
        i -= 1

salt_and_pepper_noisy_image = Image.fromarray(np.uint8(salt_and_pepper_noisy_imageArray))
salt_and_pepper_noisy_image.save("salt_and_pepper_noisy_image.jpg")
#Applying a 7 by 7 median filter to the noisy-image and show both images
median_filtered_salt_and_pepper_noisy_imageArray = median_filter(salt_and_pepper_noisy_imageArray, size=7)
median_filtered_salt_and_pepper_noisy_image = Image.fromarray(np.uint8(median_filtered_salt_and_pepper_noisy_imageArray))
plt.subplot(1, 2, 1), plt.imshow(salt_and_pepper_noisy_image, cmap='gray'), plt.title('Salt and Pepper Noise Image')
plt.subplot(1, 2, 2), plt.imshow(median_filtered_salt_and_pepper_noisy_image, cmap='gray'), plt.title('Median Filtered Image')
median_filtered_salt_and_pepper_noisy_image.save("median_filtered_salt_and_pepper_noisy_image.jpg")
plt.show()
#Applying a 7 by 7 Mean filter to the noisy-image
Mean_filtered_salt_and_pepper_noisy_imageArray = uniform_filter(salt_and_pepper_noisy_imageArray,size=7)
Mean_filtered_salt_and_pepper_noisy_image = Image.fromarray(np.uint8(Mean_filtered_salt_and_pepper_noisy_imageArray))
# show both images
plt.subplot(1, 2, 1), plt.imshow(salt_and_pepper_noisy_image, cmap='gray'), plt.title('Salt and Pepper Noise Image')
plt.subplot(1, 2, 2), plt.imshow(Mean_filtered_salt_and_pepper_noisy_image, cmap='gray'), plt.title('Mean Filtered Image')
Mean_filtered_salt_and_pepper_noisy_image.save("Mean_filtered_salt_and_pepper_noisy_image.jpg")
plt.show()

#Convoluation Function
def Convoluation2Dimensions(originalImage, SobelKernel):

    sizeOfKernal = len(SobelKernel)
    padingSize = sizeOfKernal // 2
    PaddingTheOriginalImage=np.pad(originalImage, padingSize, mode='edge')
    resultOfConvoluation=np.zeros_like(originalImage)
    for x in range(originalImage.shape[0]):
        for y in range(originalImage.shape[1]):
            resultOfConvoluation[x,y] = np.sum(PaddingTheOriginalImage[x:x+sizeOfKernal,y:y+sizeOfKernal]*SobelKernel)

    return resultOfConvoluation

SobelKernel_X =([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
SobelKernel_Y =([[-1,0,1],[ -2, 0, 2],[ -1, 0, 1]])

imageGrayLevelImage_originalArray=np.array(imageGrayLevelImage_original, dtype=np.float32)
Gx = Convoluation2Dimensions(imageGrayLevelImage_originalArray, SobelKernel_X)
Gy = Convoluation2Dimensions(imageGrayLevelImage_originalArray, SobelKernel_Y)
SobelResponse = np.sqrt(Gx**2 + Gy**2)
SobelResponseImageArrayClipped = cv2.normalize(SobelResponse,None,0,255,cv2.NORM_MINMAX)
SobelResponse_image = Image.fromarray(np.uint8(SobelResponseImageArrayClipped))
plt.subplot(1, 2, 1), plt.imshow(imageGrayLevelImage_original, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(SobelResponse_image, cmap='gray'), plt.title('Sobel Response Image')
SobelResponse_image.save("SobelResponse_image.jpg")
plt.show()

#############################Q2
def myImageFilter(input_image, filter):
    sizeOfKernal = len(filter)
    padingSize = sizeOfKernal // 2

    PaddingTheOriginalImage = np.pad(input_image, padingSize, mode='edge')
    resultOfConvoluation = np.zeros_like(input_image)
    for x in range(input_image.shape[0]):
        for y in range(input_image.shape[1]):
            resultOfConvoluation[x, y] = np.sum(PaddingTheOriginalImage[x:x + sizeOfKernal, y:y + sizeOfKernal] * filter)

    return resultOfConvoluation
def AveragingKernelGenerater(size):
    kernal = np.ones((size,size),dtype=np.float32)
    return kernal / (size * size)

def GaussianKernelGenerater(str):
    size =(2*std + 1)
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 +1))
    kernel = (1/(2*np.pi*np.square(str)))*np.exp(-(x ** 2 + y ** 2) / (2 * str ** 2))
    return kernel / np.sum(kernel)

def SobelKernelGenerater():
    SobelKernel_X = ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    SobelKernel_Y = ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return SobelKernel_X,SobelKernel_Y
def PrewittKernelGenerater():
    PrewittKernel_X = ([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    PrewittKernel_Y = ([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return PrewittKernel_X,PrewittKernel_Y

imageHouse1 = cv2.imread('House1.jpg', cv2.IMREAD_GRAYSCALE)
imageHouse2 = cv2.imread('House2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply filters and display results

averaging_kernel1 = AveragingKernelGenerater(3)
averaging_kernel2 = AveragingKernelGenerater(5)
filtered_image3x3 = myImageFilter(imageHouse1, averaging_kernel1)
filtered_image5x5 = myImageFilter(imageHouse1, averaging_kernel2)
filtered_image3x3 = Image.fromarray(np.uint8(filtered_image3x3))
filtered_image3x3.show()
display(filtered_image3x3)
filtered_image5x5 = Image.fromarray(np.uint8(filtered_image5x5))
filtered_image5x5.show()
display(filtered_image5x5)

filtered_image3x3 = myImageFilter(imageHouse2, averaging_kernel1)
filtered_image5x5 = myImageFilter(imageHouse2, averaging_kernel2)
filtered_image3x3 = Image.fromarray(np.uint8(filtered_image3x3))
filtered_image3x3.show()
display(filtered_image3x3)
filtered_image5x5 = Image.fromarray(np.uint8(filtered_image5x5))
filtered_image5x5.show()
display(filtered_image5x5)


gaussian_kernel = GaussianKernelGenerater(1)
filtered_image1 = myImageFilter(imageHouse1, gaussian_kernel)
gaussian_kernel = GaussianKernelGenerater(2)
filtered_image2 = myImageFilter(imageHouse1, gaussian_kernel)
gaussian_kernel = GaussianKernelGenerater(3)
filtered_image3 = myImageFilter(imageHouse1, gaussian_kernel)
filtered_image1 = Image.fromarray(np.uint8(filtered_image1))
filtered_image1.show()
display(filtered_image1)
filtered_image2 = Image.fromarray(np.uint8(filtered_image2))
filtered_image2.show()
display(filtered_image2)
filtered_image3 = Image.fromarray(np.uint8(filtered_image3))
filtered_image3.show()
display(filtered_image3)


gaussian_kernel = GaussianKernelGenerater(1)
filtered_image1 = myImageFilter(imageHouse2, gaussian_kernel)
gaussian_kernel = GaussianKernelGenerater(2)
filtered_image2 = myImageFilter(imageHouse2, gaussian_kernel)
gaussian_kernel = GaussianKernelGenerater(3)
filtered_image3 = myImageFilter(imageHouse2, gaussian_kernel)
filtered_image1 = Image.fromarray(np.uint8(filtered_image1))
filtered_image1.show()
display(filtered_image1)
filtered_image2 = Image.fromarray(np.uint8(filtered_image2))
filtered_image2.show()
display(filtered_image2)
filtered_image3 = Image.fromarray(np.uint8(filtered_image3))
filtered_image3.show()
display(filtered_image3)

def SobelFilter(imageHouse):
    SobelKernel_X,SobelKernel_Y = SobelKernelGenerater()
    imageHouse = np.array(imageHouse,dtype=np.float32)
    GX = myImageFilter(imageHouse, SobelKernel_X)
    GY = myImageFilter(imageHouse, SobelKernel_Y)

    magnitudeSoble = np.sqrt(GX**2 + GY**2)
    magnitudeSobleArray = cv2.normalize(magnitudeSoble,None,0,255,cv2.NORM_MINMAX)
    magnitudeSobleIemage = Image.fromarray(np.uint8(magnitudeSobleArray))
    magnitudeSobleIemage.show()
    display(magnitudeSobleIemage)
    return magnitudeSobleIemage
def PrewittFilter(imageHouse):
    PrewittKernel_X,PrewittKernel_Y = PrewittKernelGenerater()
    imageHouse = np.array(imageHouse,dtype=np.float32)
    GX = myImageFilter(imageHouse, PrewittKernel_X)
    GY = myImageFilter(imageHouse, PrewittKernel_Y)

    magnitudePrewitt = np.sqrt(GX**2 + GY**2)
    magnitudePrewittArray = cv2.normalize(magnitudePrewitt,None,0,255,cv2.NORM_MINMAX)
    magnitudePrewittIemage = Image.fromarray(np.uint8(magnitudePrewittArray))
    magnitudePrewittIemage.show()
    display(magnitudePrewittIemage)
    return magnitudePrewittIemage


magnitudeSobleIemage1 = SobelFilter(imageHouse1)
magnitudeSobleIemage2 = SobelFilter(imageHouse2)
magnitudePrewittIemage1 = PrewittFilter(imageHouse1)
magnitudePrewittIemage2 = PrewittFilter(imageHouse2)

# Subtract magnitudePrewittIemage from magnitudeSobleIemage
result = cv2.subtract(magnitudeSobleIemage1, magnitudePrewittIemage1)

# Display the original and subtracted images
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(magnitudeSobleIemage1, cmap='gray'), plt.title('Soble Image')
plt.subplot(132), plt.imshow(magnitudePrewittIemage1, cmap='gray'), plt.title('Prewitt Image')
plt.subplot(133), plt.imshow(result, cmap='gray'), plt.title('Subtraction Result')
plt.show()

# Subtract magnitudePrewittIemage from magnitudeSobleIemage
result = cv2.subtract(magnitudeSobleIemage2, magnitudePrewittIemage2)

# Display the original and subtracted images
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(magnitudeSobleIemage2, cmap='gray'), plt.title('Soble Image')
plt.subplot(132), plt.imshow(magnitudePrewittIemage2, cmap='gray'), plt.title('Prewitt Image')
plt.subplot(133), plt.imshow(result, cmap='gray'), plt.title('Subtraction Result')
plt.show()

#############################Q3
Noisyimage1 = Image.open('Noisyimage1.jpg')
Noisyimage1Array = np.array(Noisyimage1,dtype=np.uint8)
MeanFilterArray = uniform_filter(Noisyimage1Array,size=5)
MeanFilterimage = Image.fromarray(np.uint8(MeanFilterArray))
MeanFilterimage.show()
display(MeanFilterimage)

MedianFilterArray = median_filter(Noisyimage1Array, size=5)
MedianFilterimage = Image.fromarray(np.uint8(MedianFilterArray))
MedianFilterimage.show()
display(MedianFilterimage)

Noisyimage2 = Image.open('Noisyimage2.jpg')
Noisyimage2Array = np.array(Noisyimage2,dtype=np.uint8)
MeanFilterArray = uniform_filter(Noisyimage2Array,size=5)
MeanFilterimage = Image.fromarray(np.uint8(MeanFilterArray))
MeanFilterimage.show()
display(MeanFilterimage)

MedianFilterArray = median_filter(Noisyimage2Array, size=5)
MedianFilterimage = Image.fromarray(np.uint8(MedianFilterArray))
MedianFilterimage.show()
display(MedianFilterimage)
#############################Q4
Q4Image = cv2.imread('Q_4.jpg', cv2.IMREAD_GRAYSCALE)

Gx= cv2.Sobel(Q4Image,cv2.CV_64F,1,0,ksize=3)
Gy= cv2.Sobel(Q4Image,cv2.CV_64F,0,1,ksize=3)

GMagnitude = np.sqrt(Gx**2 + Gy**2)
StretchedMagnitude = cv2.normalize(GMagnitude, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(10, 6))
plt.imshow(StretchedMagnitude, cmap='gray'), plt.title('Stretched Gradient Magnitude')
plt.show()

# Compute and plot the histogram of gradient magnitude
plt.figure(figsize=(10, 6))
plt.hist(GMagnitude.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Gradient Magnitude')
plt.xlabel('Gradient Magnitude')
plt.ylabel('Frequency')
plt.show()

# Compute gradient orientation
gradientOrientation = np.arctan2(Gy, Gx)
gradientOrientationNormalized = cv2.normalize(gradientOrientation, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(10, 6))
plt.imshow(gradientOrientationNormalized, cmap='gray'), plt.title('The Angle of Gradient Vector')
plt.show()

# Compute and plot the histogram of gradient orientation
plt.figure(figsize=(8, 4))
plt.hist(gradientOrientation.ravel(), bins=36, range=[0, 2 * np.pi])
plt.title('Histogram of Gradient Orientation')
plt.xlabel('Gradient Orientation (Radians)')
plt.ylabel('Frequency')
plt.show()
#############################Q5
walk_1Image = cv2.imread('walk_1.jpg', cv2.IMREAD_GRAYSCALE)
walk_2Image = cv2.imread('walk_2.jpg', cv2.IMREAD_GRAYSCALE)
# Subtract walk_2 from walk_1
result = cv2.subtract(walk_1Image, walk_2Image)

# Display the original and subtracted images
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(walk_1Image, cmap='gray'), plt.title('Walk 1')
plt.subplot(132), plt.imshow(walk_2Image, cmap='gray'), plt.title('Walk 2')
plt.subplot(133), plt.imshow(result, cmap='gray'), plt.title('Subtraction Result')
plt.show()

#############################Q6
# Load the image
Q4image = cv2.imread('Q_4.jpg', cv2.IMREAD_GRAYSCALE)
# Apply Canny edge detector with different threshold values
thresholds = [50, 100, 150]  # You can experiment with different threshold values

plt.subplot(2, 2, 1), plt.imshow(Q4image, cmap='gray'), plt.title('Original Image')
# Apply Canny for different threshold values
edges = cv2.Canny(Q4image, thresholds[0], thresholds[0] * 2)
plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray'), plt.title(f'Canny (Threshold={thresholds[0]})')
edges = cv2.Canny(Q4image, thresholds[1], thresholds[1] * 2)
plt.subplot(2, 2, 3), plt.imshow(edges, cmap='gray'), plt.title(f'Canny (Threshold={thresholds[1]})')
edges = cv2.Canny(Q4image, thresholds[2], thresholds[2] * 2)
plt.subplot(2, 2, 4), plt.imshow(edges, cmap='gray'), plt.title(f'Canny (Threshold={thresholds[2]})')
plt.tight_layout()
plt.show()