import cv2 as cv
from cv2 import blur
from cv2 import dilate
from matplotlib.pyplot import contour
from numpy import number
import numpy as np

path ="/home/imman/Imman Codings/Deep-Face/resources/coin2.jpg"
image = cv.imread(path)
#image = cv.VideoCapture(0)
#half = cv.resize(image,(0,0),fx=0.2,fy=0.2)



while True:

    #ret,frame = image.read()
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    #kernal = np.ones((2,2), np.uint8)

    

    blur = cv.GaussianBlur(gray,(11,11),1)
    _, thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)
    edges = cv.Canny(blur,30,200,3)
    dilated = cv.dilate(edges,(2,2),iterations=1)


    (cnt,hierarchy) = cv.findContours(dilated.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    cv.drawContours(rgb,cnt,-1,(0,255,0),2)

    
    cv.imshow('Thresh',thresh)
    cv.imshow('Dilated',dilated)
    #cv.imshow('Black & White',gray)
    #cv.imshow('Blur',blur)




    number_objects = len(cnt)
    text = "Objects:"+str(number_objects)
    cv.putText(rgb,text,(10,25),cv.FONT_HERSHEY_COMPLEX,0.4,(240,0,159),1)
    key=cv.waitKey(5)

    cv.imshow('Final Image',rgb)

    if key==ord('q'):
        break

cv.destroyAllWindows()








    

    
 

print("The number of objects in this image",(number_objects))

