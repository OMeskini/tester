import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import cv2 as cv
import os
from random import randint, uniform, choice,gauss
from os.path import basename
from PIL import Image, ImageDraw
import math

#load path
path='/home/oumaima/Pictures/'


    # load images
for pic in os.listdir(path):
   #read image
   img = cv.imread(path+pic)

   #get shape
   height, weight, channels = img.shape

   # Create a white image
   img_white = np.zeros((height, weight, 3), np.uint8)
   img_white[:] = (255, 255, 255)

   # list of fonts
   list=[cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_PLAIN, cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX,cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_TRIPLEX, cv.FONT_HERSHEY_COMPLEX_SMALL,cv.FONT_HERSHEY_SCRIPT_SIMPLEX,  cv.FONT_HERSHEY_SCRIPT_COMPLEX]

   #Localization of text
   bottom=randint(50,height-50)
   left=randint(0,weight-50)

   #text caracteristics
   font = choice(list)
   fontScale =uniform(0.3, 6.5)
   lineType = randint(1, 5)

   #random color
   u = gauss(0, 1)
   v = gauss(0, 1)
   z = gauss(0, 1)

   R=abs(int(30*u))
   G=abs(int(20*v))
   B=abs(int(30*z))

   fontColor = (R, G, B)


   #number of lines
   nb_lines=randint(1,5)

   #in case of little image apply an average fontscale
   if height < 200:
       fontScale = uniform(0.3, 3.5)

   #foreach line
   for j in range(1, nb_lines):
    #choose a text
    words = open('/etc/dictionaries-common/words').read().splitlines()

    line=""
    while line=="":
     for k in range(1,randint(1,4)):
       x=choice(words)
       line += x+" "


    #for visibility of text
    if (fontScale < 1):
        lineType = 1

    bottomLeftCornerOfText = (left, bottom)

    #write the text
    cv.putText(img, line,
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)

    #crop region of text
    if(fontScale<2):
        diff=40
    else:
        diff=90

    d=2*diff
    bd = bottom + diff
    if (bd > height):
        bd = height
        d=height-(bottom-diff)

    cropped_img = img[bottom-diff:bd, left:weight]

   # M = cv.getRotationMatrix2D((bottom/2,left/2), 3, 1.0)
    #rotated = cv.warpAffine(cropped_img, M, (weight-left,d))
    #img[bottom-diff:bd, left:weight]=rotated

    blur = cv.GaussianBlur(cropped_img, (5, 5), 0)
    h =bd-(bottom-diff)
    w=weight-left


    # worley image


    image = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(image)
    pixels = image.load()
    n = 100  # of seed points
    m = 0  # random.randint(0, n - 1) # degree (?)
    seedsX = [randint(0, w- 1) for i in range(n)]
    seedsY = [randint(0, h - 1) for i in range(n)]

    # find max distance
    maxDist = 0.0
    for ky in range(h):
        for kx in range(w):
              # create a sorted list of distances to all seed points
              dists = [math.hypot(seedsX[i] - kx, seedsY[i] - ky) for i in range(n)]
              dists.sort()
              if dists[m] > maxDist: maxDist = dists[m]

    # paint
    for ky in range(h):
        for kx in range(w):
            # create a sorted list of distances to all seed points
            dists = [math.hypot(seedsX[i] - kx, seedsY[i] - ky) for i in range(n)]
            dists.sort()
            c = int(round(255 * dists[m] / maxDist))
            pixels[kx, ky] = (0, 0, c)

    image.save("WorleyNoise.png", "PNG")

    img_worley = cv.imread("WorleyNoise.png")

    dst = cv.addWeighted(blur, 0.6, img_worley, 0.4, 0)
    #kernel = np.ones((3, 3), np.uint8)
    #img_dilated = cv.dilate(blur, kernel)
    img[bottom-diff:bd, left:weight]=dst


    cv.putText(img_white, line,
               bottomLeftCornerOfText,
               font,
               fontScale,
               fontColor,
               lineType)

    #create a directory to hold data_set
    if not os.path.exists("data_set_directory"):
        os.makedirs("data_set_directory")

    #name of image without the extension
    img_name = basename("/a/b/c"+os.path.splitext(path+pic)[0])

    #save image
    cv.imwrite("data_set_directory/"+ img_name +".jpg", img)
    cv.imwrite("data_set_directory/"+ img_name + "_out_white.jpg", img_white)

    #for next line
    if(nb_lines>1) :
       # load image to write next line
       img = cv.imread("data_set_directory/" + img_name + ".jpg")
       img_white = cv.imread("data_set_directory/" + img_name + "_out_white.jpg")

       #make space between lines
       if(fontScale<4):
         bottom=bottom+90

       else:
           bottom=bottom+120


       if (bottom>height-90):
            break


    #cv.imshow('dilated', img_dilated)

    pixelSize = 9
    resized_image = cv.resize(img,(height/pixelSize,weight/pixelSize))
    resized_image2 = cv.resize(resized_image, (height*pixelSize,weight*pixelSize))


    #s&p noise
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
    out[coords] = 0

    #cv.imwrite("data_set_directory/" + img_name + "new.jpg", img)


cv.waitKey(0)
