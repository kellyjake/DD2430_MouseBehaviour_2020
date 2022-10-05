import PySpin
import cv2
import datetime
import time
import faulthandler

class BlackFlyCam:

    def __init__(self, fileName='test', serialNo='18475994', isColor=False):
        faulthandler.enable()
        print("-- Initializing camera --")
        self.isColor = isColor

        self.serial = serialNo
        self.fileName = fileName

        self.timestamps = []

        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        print("In initCamera")
        system = PySpin.System.GetInstance()

        camList = system.GetCameras()
        print(camList)
        #self.blackFly = camList.GetBySerial(self.serial)

        #self.blackFly.Init()
        #self.blackFly.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        #self.height = self.blackFly.Height()
        #self.width = self.blackFly.Width()
        #self.channels = 1 # Black and white only
        #self.fps = self.blackFly.AcquisitionFrameRate()
        print(self.__dict__)
        print("Camera initialized")

        self.initCamera()
        print("Made it out of initCamera")
                
        #vidWriter = cv2.VideoWriter(self.fileName + '.avi',self.fourcc, self.fps, (self.width,self.height), self.isColor)

    def initCamera(self):
        pass
    
    def startCamera(self):
        print("-- Starting camera --")
        self.blackFly.BeginAcquisition()
    
    def stopCamera(self):
        print("-- Stopping camera --")
        self.blackFly.EndAcquisition()

    def initWriter(self,fourcc,fps,width,height):
        pass
        """
        print("In initWriter")
        
        
        try:
            self.vidWriter = cv2.VideoWriter(self.fileName + '.avi',fourcc, fps, (width,height), self.isColor)
        except Exception as e:
            print(e)
        """
    
    def recUntilEventStop(cam, event):
        pass
        timestamps = []
        while not(event.isSet()):
            im = cam.GetNextImage()
            timestamps.append(time.time())
            im_cv2_conv = im.GetData().reshape(self.height,self.width,self.channels)
            vidWriter.write(im_cv2_conv)
        
        print("-- Saving video to file, please wait... --")
        vidWriter.release()
        

    def recForSecs(self,sec):
        print("In recForSecs")
        nFrames = int(self.fps*sec)

        for _ in range(nFrames):
            im = self.blackFly.GetNextImage()
            self.timestamps.append(time.time())
            im_cv2_conv = im.GetData().reshape(self.height,self.width,self.channels)
            self.vidWriter.write(im_cv2_conv)
        
        print("-- Saving video to file, please wait... --")
        self.vidWriter.release()

    def addTimeStamps(self,pos=['left','bottom'],pixShift=[10,10]):
        
        if pos[0] == 'right':
            xPos = self.width - 100 # Must subtract 100 to make all of text fit in frame
        else:
            xPos = 0

        if pos[1] == 'top':
            yPos = self.height
        else:
            yPos = 0

        org = (xPos + pixShift[0], yPos + pixShift[1])

        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        thickness = 1
        color = 0

        cap = cv2.VideoCapture(self.fileName + '.avi')
        vidWriter_timed = cv2.VideoWriter(self.fileName + '_timed.avi',self.fourcc, self.blackFly.AcquisitionFrameRate() , (self.width,self.height), self.isColor)

        i = 0
        success = True

        while success:
            success , frame = cap.read()
            if success:
                localTime = datetime.datetime.fromtimestamp(self.timestamps[i]).strftime("%Y:%m:%d:%H:%M:%S:%f")
                img_t = cv2.putText(frame, localTime ,org,fontFace,fontScale,color,thickness, cv2.LINE_AA, True)
                grayImage = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
                vidWriter_timed.write(grayImage)

            i += 1

        cap.release()
        vidWriter_timed.release()

