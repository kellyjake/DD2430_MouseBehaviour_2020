import cv2
import datetime
import time
from threading import Event
import imutils 


def recUntilEventStop(cam, writer, event, channels=1):
    timestamps = []

    cam.BeginAcquisition()

    while not(event.isSet()):
        im = cam.GetNextImage()
        timestamps.append(time.time())
        im_cv2_conv = im.GetData().reshape(cam.Height(),cam.Width(),channels)
        writer.write(im_cv2_conv)
    
    print("-- Saving video to file, please wait... --")
    writer.release()
    cam.EndAcquisition()

    return timestamps


def recForSecs(cam, writer, sec, channels=1):
    timestamps = []

    cam.BeginAcquisition()
    begin_t = time.time()
    
    t = 0

    while (t - begin_t) < sec:
        im = cam.GetNextImage()
        t = time.time()
        timestamps.append(t)
        im_cv2_conv = im.GetData().reshape(cam.Height(),cam.Width(),channels)
        writer.write(im_cv2_conv)
    
    print("-- Timestamps added, saving video to file. Please wait... --")
    writer.release()
    cam.EndAcquisition()

    return timestamps


def recordCameraTest(main_event_cam_test,restart_event_cam,restart_event_ard,tracker,isColor=False):
    print("-- [VIDEO] RECORDING --")

    while not(main_event_cam_test.isSet()):
        vidName = tracker.get_current_savepath(append_str='recording',save_format='avi')
        csvName = tracker.get_current_savepath(append_str='timestamps',save_format='csv')
        
        csvFile = open(csvName,'w')
        videoFile = open(vidName,'w')

        tWriter = csv.writer(csvFile)
        tWriter.writerow('Camera timestamps')

        while (restart_event_cam.isSet() & (not(main_event_cam_test.isSet()))):
            timestamp = time.time()
            tWriter.writerow(timestamp)

        print("-- Saving video to file, please wait... --")
        csvFile.close()
        videoFile.close()

        restart_event_cam.set()
        restart_event_ard.wait()

    return
