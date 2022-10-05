import cv2
import numpy as np
from tkinter import messagebox
import tkinter as tk
import sys

class LEDMarker:

    def __init__(self,fileName,bufferSize):
        self.__cap = cv2.VideoCapture(fileName)
        self.__height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__bufferSize = bufferSize

        self.__init_buffer()
        
        self.__init_tk()

        self.__curr_frame_idx = 0
        #self.__img = None
        self.__update_frame()

    def __init_tk(self):
        self.__root = tk.Tk()
        self.__root.withdraw()

    def __next_frame(self):
        if self.__curr_frame_idx < self.__bufferSize - 1:
            self.__curr_frame_idx += 1
        else:
            self.__new_frame()
        
        self.__update_frame()

    def __prev_frame(self):
        if self.__curr_frame_idx > 0:
            self.__curr_frame_idx -= 1

            self.__update_frame()
        else:
            messagebox.showerror(title='End of buffer',message=F"Reached beggining of buffer! Can't go back any further. (Buffersize {self.__bufferSize})")

    def __update_frame(self):
        self.__curr_frame = self.__buffer[self.__curr_frame_idx]
        self.__clone = self.__curr_frame.copy()    
        self.__img = self.__clone.copy()
        cv2.imshow('Frame',self.__img)

    def __new_frame(self):
        success , frame = self.__cap.read()
        if success:
            del self.__buffer[0]
            self.__buffer.append(frame)
        else:
            messagebox.showerror(title='End of video',message='Reached last frame!')

    def __init_buffer(self):
        self.__buffer = []
        for _ in range(self.__bufferSize):
            success , frame = self.__cap.read()
            if success:
                self.__buffer.append(frame)

    def __show_frame(self):
        while(1):
            cv2.imshow('Frame',self.__img)
            k = cv2.waitKey(1) & 0xFF
            if k == 13:  
                break # break upon pressing Enter
            elif k == 110:
                self.__next_frame()
            elif k == 112: 
                self.__prev_frame()

        cv2.destroyAllWindows()

    def __check_inbound(self,x,y):
        if not(x > 0):
            x = 0
        elif not(x < self.__width):
            x = self.__width - 1

        if not(y > 0):
            y = 0
        elif not(y < self.__height):
            y = self.__height - 1

        return x , y

    # mouse callback function
    def __draw_rect(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if bool(self.__ref_point): 
                self.__ref_point.clear()

            x_bounded , y_bounded = self.__check_inbound(x,y)
            
            self.__ref_point = [(x_bounded,y_bounded)]

        elif event == cv2.EVENT_LBUTTONUP:
            
            x_bounded , y_bounded = self.__check_inbound(x,y)

            self.__ref_point.append((x_bounded,y_bounded))

            cv2.rectangle(self.__img,self.__ref_point[0],self.__ref_point[1],(255,0,0),1)
        
            
    def mark_LED(self):
        self.__ref_point = []

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame',self.__draw_rect)
            
        self.__clone = self.__curr_frame.copy()    
        self.__img = self.__clone.copy()

        messagebox.showinfo(title='Bounding box',message='Please draw a box around the location of the LED. Be as precise as possible. \nYou have as many tries as you like. \nSee next frame by pressing "n"\nSee previous frame by pressing "p"\n Press ENTER when finished.')

        self.__show_frame()

    def get_bounding_box_pixels(self):
        [xmin , ymin] , [xmax , ymax] = np.sort(self.__ref_point,0)
        
        return xmin , xmax , ymin , ymax

def main(args):
    try:
        
        fileName , bufferSize = args
        test = LEDMarker(fileName,int(bufferSize))
        test.mark_LED()
        print(test.get_bounding_box_pixels())

        #f = open(fileName,'rb')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])