#!C:\\Users\\magnu\\anaconda3\\envs\\camEnv\\python.exe

import cv2
import imutils
import time , datetime

def addTimeStamps(img,timestamp,pos=['right','bottom'],pixShift=[0,0], monoChrom=True, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255,255,255), show=False):
        
        height , width , channels = img.shape

        if pos[0] == 'right':
            yPos = width - 12
        else:
            yPos = 15 # Must subtract 100 to make all of text fit in frame

        if pos[1] == 'bottom':
            xPos = 5
        else:
            xPos = height - 220

        org = (xPos + pixShift[1], yPos + pixShift[0])
        
        rotIm = imutils.rotate_bound(img, 90)

        localTime = datetime.datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d %H:%M:%S:%f")[:-3]
        img_t = cv2.putText(rotIm, localTime ,org,fontFace,fontScale,color,thickness,cv2.LINE_AA, False)

        if monoChrom:
            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        
        unRot = imutils.rotate_bound(img_t, -90)
        
        if show:
            cv2.imshow("img",unRot)
            cv2.waitKey()
        
        return unRot

pic = r"C:\\Users\\magnu\\OneDrive\\Dokument\\KI\\KI2020\\test_out3.avi"

cap = cv2.VideoCapture(pic)
succ,im = cap.read()
im_t = addTimeStamps(im, time.time(),pos=['left','top'],color=(0,0,0))