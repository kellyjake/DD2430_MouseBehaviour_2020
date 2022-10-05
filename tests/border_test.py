import cv2 as cv

im = cv.imread(r'c:\Users\magnu\OneDrive\Bilder\friday_report.png')

im_b = cv.copyMakeBorder(im, 30, 0, 30 , 30, cv.BORDER_CONSTANT, value=(0,255,0))
im_b = cv.copyMakeBorder(im_b, 0, 30, 0 , 0, cv.BORDER_CONSTANT, value=(0,0,0))
im_b = cv.cvtColor(im_b, cv.COLOR_BGR2RGB)
wname = 'test'
w = cv.namedWindow(wname)

cv.imshow(wname,im_b)
cv.waitKey(0)