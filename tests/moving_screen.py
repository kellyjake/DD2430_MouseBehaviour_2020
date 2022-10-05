import cv2
import screeninfo
import numpy as np

scrns = screeninfo.get_monitors()
w = scrns[0].width
h = scrns[0].height
img = np.random.random((h,w))
print(img)
w_name = 'test'
win = cv2.namedWindow(w_name,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(w_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cv2.moveWindow(w_name,4000,0)

cv2.imshow(w_name,img) 

#cv2.moveWindow(w_name,12000,0)

cv2.getWindowImageRect(win)
cv2.waitKey(0)
cv2.destroyAllWindows()