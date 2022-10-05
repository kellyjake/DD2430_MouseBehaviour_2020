import simple_pyspin
import cv2


fourcc = cv2.VideoWriter_fourcc(*'XVID')
blackFly = simple_pyspin.Camera()
blackFly.init()

blackFly.AcquisitionFrameRate

blackFly.get_info('AcquisitionFrameRate')

blackFly.AcquisitionFrameRate
#blackFly.ExposureAuto = False
#blackFly.ExposureTime = 40000

#blackFly.AcquisitionFrameRateAuto = 'Off'
blackFly.AcquisitionFrameRateEnable = True
blackFly.AcquisitionFrameRate = 10.
