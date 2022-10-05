import cv2
import numpy as np
from mark_LED import LEDMarker
import sys
import matplotlib.pyplot as plt
import tqdm

class Syncer:
    def __init__(self,videoFileName, timeStampFileName, lims=None, bufferSize=100):
        self.__videoName = videoFileName
        self.__timeStampFile = timeStampFileName
        self.__bufferSize = bufferSize

        if lims is None:
            self.__marker = LEDMarker(self.__videoName,self.__bufferSize)
            self.__marker.mark_LED()
            self.__xmin , self.__xmax , self.__ymin , self.__ymax = self.__marker.get_bounding_box_pixels()   
        else:
            self.__xmin , self.__xmax , self.__ymin , self.__ymax = lims
        
        self.__limits = {"x_min":self.__xmin, "x_max":self.__xmax, "y_min":self.__ymin, "y_max":self.__ymax}

        self.__load_timestamps()

    def compute_intensities(self):
        self.__intensities = []

        self.__cap = cv2.VideoCapture(self.__videoName)
        success = True
        n_frames = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm.tqdm(total=n_frames)
        
        with pbar:
            while success:
                success , frame = self.__cap.read()
                if success:
                    scaled_frame = frame[self.__limits['x_min']:self.__limits['x_max'],self.__limits['y_min']:self.__limits['y_max']]
                    intensity = np.mean(scaled_frame)
                    self.__intensities.append(intensity)
                    pbar.update(1)

    def __load_timestamps(self):
        print("Loading timestamps")
        with open(self.__timeStampFile,'r') as csv_file:
            csv_file.readline() # Read headers
            self.__timestamps = np.array(list(filter(None,csv_file.read().split('\n'))),dtype='float64') # Removing newlines
            self.__relative_time = self.__timestamps - self.__timestamps[0]

    def get_intensities(self):
        return self.__intensities

    def get_timestamps(self,relative=True):
        if relative: 
            return self.__relative_time
        else:
            return self.__timestamps

    def make_plot(self,show=True):
        plt.clf()
        relative_time = self.__timestamps - self.__timestamps[0]
        plt.plot(relative_time,self.__intensities)

        if show: plt.show()