import cv2 as cv

vid_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\20201202_behaviour2020_v_6287_distractionduring_task_vol2_1_recording.avi'

cap = cv.VideoCapture(vid_file)

ok , img = cap.read()

img = img[:370,:]

_ = cv.namedWindow('test')

cv.imshow('test',img)
cv.waitKey(0)