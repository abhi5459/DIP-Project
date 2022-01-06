import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import time
import functools
import heapq
from scipy.ndimage import morphology as morph
import sys
from skimage.morphology import disk
from skimage.metrics import structural_similarity as ssim

def getImg(path, flag):  #flag = 0 for greyscale images, 1 for coloured

    # Reads image from given file path and resizes it if needed(testing purposes)
    img=cv2.imread(path, flag)
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    for i in range(len(powers)):
        if width<=powers[i]:
            width=powers[i]
            break
            
    for i in range(len(powers)):
        if height<=powers[i]:
            height=powers[i]
            break
            
    print(width,height)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #Converts image from BGR to RGB format
    if flag == 1:
        b,g,r=cv2.split(img)
        img=cv2.merge([r,g,b])

    return img

def genDepthImage(imgL, imgR): #Takes two images in stereoscopic alignment to generate a depth image
    l=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    r=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
    image = stereo.compute(l,r)

    return image

def downsample(img, scale_percent): #Downsamples an image based on the scaling factor
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return img

def gaussian(sigma, x):
    #Returns the gaussian distribution at a location x for the given sigma
    # print("gauss got ", sigma, x)
    return float(math.exp(-(x**2/(2*(sigma**2)))))

def stepUp(img):
    #Converts an NxM image to an 2Nx2M image by adding alternating empty rows and columns to the image
    newimage = np.zeros((img.shape[0]*2, img.shape[1]*2))
    newimage[::2,::2] = img
    newimage = newimage.astype('uint8')
    return newimage

def genLambda(img):
    #Uses the dimensions of image to generate the lambda array needed during depth propagation
    arr = np.ones(img.shape)
    arr = stepUp(arr)
    return arr

def genGuidanceImg(rgb, depth): # generates downsampled grey scale image using rgb original image using 
                                # dimensions of intermediate depth image

    grey = cv2.cvtColor(np.copy(rgb), cv2.COLOR_RGB2GRAY)
    grey = downsample(grey, (depth.shape[0]/grey.shape[0])*100)
    ans = (grey - np.min(grey))/(np.max(grey)-np.min(grey))
    return ans

def propagate(depth, guidance, k, lam, sigS, sigR): # fills the empty pixels using the neighbour pixels  

    templam = np.copy(lam)
#     fills in cells with four neighbours on the diagonals
    depth = depth.astype('int')
    for i in range(1,depth.shape[0]-1, 2):
        for j in range(1, depth.shape[1]-1, 2): #iterating only over the required pixels in the first step
            # total is the average of pixel values of 4 neighbours
            total = depth[i-1][j-1]+depth[i+1][j-1]+depth[i-1][j+1]+depth[i+1][j+1]
            total /= 4
            # creating an array containing all the 4 neighbour pixel values and their average
            dx = [depth[i-1][j-1],depth[i+1][j-1],depth[i-1][j+1],depth[i+1][j+1], total]
            val = sys.float_info.max
            for d in dx:
                final = 0
                # iterating over neighbourhood of filter size
                for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                    for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                        if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                            continue
                        elif lam[p][q] == 0:
                            continue
                        else:
                            euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                            final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))
                # taking the d with minimum cost value and assigning it to the empty pixel
                if final < val:
                    val = final
                    depth[i][j] = d
                    templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells with four neighbours along the edges
    for i in range(1,depth.shape[0]-2):
        for j in range(1, depth.shape[1]-2):
            if lam[i][j] == 1:
                continue
            total = depth[i][j-1]+depth[i][j+1]+depth[i-1][j]+depth[i+1][j]
            total /= 4
            dx = [depth[i][j-1], depth[i][j+1], depth[i-1][j], depth[i+1][j], total]
            val = sys.float_info.max
            for d in dx:
                final = 0
                for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                    for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                        if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                            continue
                        elif lam[p][q] == 0:
                            continue
                        else:
                            euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                            final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

                if final < val:
                    val = final
                    depth[i][j] = d
                    templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells on the leftmost column with three neighbours
    for i in range(1,depth.shape[0]-2):
        if lam[i][0] == 1:
                continue
        j=0
        total = depth[i][1]+depth[i-1][0]+depth[i+1][0]
        total /= 3
        dx = [depth[i][1], depth[i-1][0], depth[i+1][0], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells on the second rightmost column with three neighbours
    for i in range(1,depth.shape[0]-2):
        j=depth.shape[1]-2
        if lam[i][j] == 1:
            continue
        total = depth[i][j-1]+depth[i-1][j]+depth[i+1][j]
        total /= 3
        dx = [depth[i][j-1], depth[i-1][j], depth[i+1][j], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells on the topmost row with three neighbours    
    for j in range(1,depth.shape[1]-2):
        i=0
        if lam[i][j] == 1:
            continue
        total = depth[i+1][j]+depth[i][j-1]+depth[i][j+1]
        total /= 3
        dx = [depth[i+1][j], depth[i][j-1], depth[i][j+1], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells on the second bottommost row with three neighbours
    for j in range(1,depth.shape[1]-2):
        i=depth.shape[0]-2
        if lam[i][j] == 1:
            continue
        total = depth[i-1][j]+depth[i][j-1]+depth[i][j+1]
        total /= 3
        dx = [depth[i-1][j], depth[i][j-1], depth[i][j+1], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
     
#     fills in cells on the bottommost row with three neighbours from previous row
    for j in range(0,depth.shape[1]-1):
        
        i=depth.shape[0]-1
        if lam[i][j] == 1:
            continue
        dx=[]
        if j==0:
            total = depth[i-1][j]+depth[i-1][j+1]
            total /= 2
            dx = [depth[i-1][j], depth[i-1][j+1], total]
        else:
            total = depth[i-1][j]+depth[i-1][j-1]+depth[i][j+1]
            total /= 3
            dx = [depth[i-1][j], depth[i][j-1], depth[i][j+1], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills in cells on the rightmost column with three neighbours from previous column
    for i in range(0,depth.shape[0]-1):
        j=depth.shape[1]-1
        if lam[i][j] == 1:
            continue
        if i==0:
            total = depth[i][j-1]+depth[i+1][j-1]
            total /= 2
            dx = [depth[i][j-1], depth[i+1][j-1], total]
        else:
            total = depth[i-1][j-1]+depth[i][j-1]+depth[i+1][j-1]
            total /= 3
            dx = [depth[i-1][j-1], depth[i][j-1], depth[i+1][j-1], total]
        val = sys.float_info.max
        for d in dx:
            final = 0
            for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
                for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                    if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                        continue
                    elif lam[p][q] == 0:
                        continue
                    else:
                        euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                        final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

            if final < val:
                val = final
                depth[i][j] = d
                templam[i][j] = 1
    lam = np.copy(templam)
    
#     fills the bottom right corner pixel using the three neighbours
    i = depth.shape[0]-1
    j = depth.shape[1]-1
    total = depth[i-1][j-1]+depth[i][j-1]+depth[i-1][j]
    total /= 3
    dx = [depth[i-1][j-1], depth[i][j-1], depth[i-1][j], total]
    val = sys.float_info.max
    for d in dx:
        final = 0
        for p in range(i-math.floor(k/2), k+i-math.floor(k/2)):
            for q in range(j-math.floor(k/2), k+j-math.floor(k/2)):
                if p < 0 or q < 0 or p >= guidance.shape[0] or q >= guidance.shape[1]:
                    continue
                elif lam[p][q] == 0:
                    continue
                else:
                    euc = math.sqrt((((i-p)/depth.shape[0])**2)+((((j-q)/depth.shape[1])**2)))
                    final += (gaussian(sigS, euc) * gaussian(sigR, guidance[i][j]-guidance[p][q]) * min(0.5, abs(d - depth[p][q])))

        if final < val:
            val = final
            depth[i][j] = d
            templam[i][j] = 1
    depth = depth.astype('uint8')
    return depth

# calls the propagate function the no. of times upsampling is needed till we reach the ground_truth image resolution
def call_back(depth, rgb, k, sigS, sigR, levels): 
    result = depth
    for i in range(levels):
        result = stepUp(np.copy(result))
        norm = genGuidanceImg(rgb, result)
        lam = genLambda(result)
        result = propagate(np.copy(result), norm, k, lam, sigS, sigR)
    return result

def error(ground_truth, output): # returns similarity on a scale of 0 to 1(1 being the identical images)
    return ssim(ground_truth, output)

# global variables

downsampling_percentange = 100
iterations =2
filter_size = 7
sigma_s = 1
sigma_r = 0.2

# making an array for powers of 2 upto 64 and then multiples of 64

powers=[]
temp = 1
powers.append(temp)
for i in range(1,7):
    temp=temp*2
    powers.append(temp)
temp_a=0
for i in range(100):
    temp_a+=temp
    powers.append(temp_a)

# def prin(x):
# 	return x

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# cv2.createTrackbar('Downsampling_level', 'image',0,6, prin)
# # switch1 = '0 : BLUE n1 : GREEN n2 : RED'
# cv2.createTrackbar('Filter size', 'image', 0, 2, prin)
# # switch2 = '0 : BLUE n1 : GREEN n2 : RED n3 : NONE'
# cv2.createTrackbar('SigmaR', 'image', 3, 3, prin)
# # switch3 = '0 : DARK n1 : LIGHT'
# cv2.createTrackbar("SigmaS", 'image', 0, 1, prin)


def centre(scale,k,sigmaS,sigmaR,rgb,ground_truth):

    output=0
    rgb=getImg(rgb,1)
    ground_truth=getImg(ground_truth,0)
    fig, index = plt.figure(figsize=(30, 12)), 1
    rows,columns=1,3

    # scale,k,sigmaS,sigmaR=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
    depth = downsample(np.copy(ground_truth),100/(2**scale))
    output = call_back(depth,rgb,k,sigmaS,sigmaR,scale)

    print("hello")

    

    # depthimg = gen_depth_image(imgL, imgR)
    errp = error(ground_truth, output)*100
    fig.add_subplot(rows, columns, index)
    index += 1
    plt.imshow(ground_truth,cmap="gray")
    plt.title("Ground Truth depth image")
    fig.add_subplot(rows, columns, index)
    index += 1
    plt.imshow(output, cmap="gray")
    plt.title("The Upscaled output is "+ str(errp) + "% Similar")
    plt.show()
    print("yo")

    # cv2.imshow('Hi',output)
    # cv2.waitKey(0)
# cv2.imwrite('output.png',output)
# time.sleep(20)
# prevScale = 0
# prevk = -1
# prev_sigmaR = -1
# prev_sigmaS = -1
# while True:
#     # time.sleep(3)
#     scale = cv2.getTrackbarPos('Downsampling_level', 'image')
#     if(scale == 0):
#         scale = 1
#     k = cv2.getTrackbarPos('Filter size', 'image')
#     sigmaR = cv2.getTrackbarPos('SigmaR', 'image')
#     sigmaS = cv2.getTrackbarPos('SigmaS', 'image')
#     # print(exp)
#     if scale != prevScale or prevk!=k or prev_sigmaR!=sigmaR or prev_sigmaS!=sigmaS:
#         depth = downsample(np.copy(ground_truth),100/(2**scale))
#         output = call_back(depth,rgb,k,sigmaS,sigmaR,scale)
#         prevScale = scale
#         prevk = k
#         prev_sigmaR = sigmaR
#         prev_sigmaS = sigmaS
#         cv2.imshow('Original', output)

#     # cv2.imshow('image', ground_truth)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()