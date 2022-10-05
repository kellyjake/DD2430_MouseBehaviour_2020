import numpy as np
import cv2 as cv
import screeninfo
from skimage import draw
import matplotlib.pyplot as plt
import math

class Mouse():
    def __init__(self,size,view_angle,view_range):
        self.size = size
        self.angular_view = view_angle
        self.angular_range = view_range
        print(F"Input angular range: {self.angular_range}")
        self.rotation = lambda theta : np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

        screen = screeninfo.get_monitors()[-1]
        w = screen.width
        h = screen.height

        self.angle = 0
        self.pos = np.array([w/2,h/2])
    
        self.gaze_vec = self.pos
        print(F"Initial gaze_vec: {self.gaze_vec}")
        self.lims = [w,h]


    def random_step(self,stepsize):
        delta = (np.random.random(2) - 0.5)*2*stepsize
        #print(F"Lims: {self.lims}")
        #print(F"Old pos: {self.pos}")
        #print(F"New pos: {self.pos + delta}")

        for i,p in enumerate(self.pos):
            if (0 <= (self.pos[i] + delta[i]) < self.lims[i]):
                self.pos[i] += delta[i]
            else:
                print("Lims breached!")
                self.pos[i] = self.lims[i] - 1
        
        print(F"Pos: {self.pos}")
        return self.pos

    def random_gaze(self,stepsize):
        delta_angle = 2*(np.random.random() - 0.5)*stepsize
        
        self.angle += delta_angle
        self.angle %= 360

        rot = self.rotation(np.radians(self.angle))

        print(F"Rotating by angle {delta_angle} to angle {self.angle}")

        self.gaze_vec = self.pos + np.array([0,self.angular_range]) @ rot
        print(F"New gaze vec: {self.gaze_vec}")

    def step(self,stepx,stepy):
        delta = [stepx,stepy]

        for i,p in enumerate(self.pos):
            if (0 <= (self.pos[i] + delta[i]) < self.lims[i]):
                self.pos[i] += delta[i]
            else:
                self.pos[i] = self.lims[i] - 1
        
        return self.pos

    def get_gaze(self):
        #lp_x = self.gaze_vec[0]*np.arccos(np.radians(-self.angular_view/2))
        #lp_y = self.gaze_vec[1]*np.arcsin(np.radians(-self.angular_view/2))
        lp = self.rotation(-np.radians(self.angular_view/2)).T @ (self.gaze_vec - self.pos) + self.pos
        rp = self.rotation(np.radians(self.angular_view/2)).T @ (self.gaze_vec - self.pos) + self.pos

        #rp_x = self.gaze_vec[0]*np.arccos(np.radians(self.angular_view/2))
        #rp_y = self.gaze_vec[1]*np.arcsin(np.radians(self.angular_view/2))
        
        #lp = (lp_x,lp_y)
        #rp = (rp_x,rp_y)
        #lp = self.gaze_vec @ self.rotation(np.radians(-self.angular_view/2))
        #rp = self.gaze_vec @ self.rotation(np.radians(self.angular_view/2))
        print(F"lp: {lp} , rp:{rp}")

        return self.gaze_vec , lp , rp

    def update_pos(self):
        self.gaze_vec = self.pos + self.angular_range


def looming(speed1,speed2,max_t,max_rad):
    
    x = np.linspace(2,5,max_t)

    sigmoid = (x**speed1)*(speed2**x)
    sigmoid /= sigmoid[-1]

    return sigmoid

def draw_circle(h,w,rad,pos):
    arr = np.ones((h,w))
    rr , cc = draw.disk(pos,radius=rad,shape=arr.shape)
    
    arr[rr, cc] = 0

    return arr

def draw_mouse(mouse,frame):

    rr_mouse , cc_mouse = draw.disk((mouse.pos[1],mouse.pos[0]),mouse.size,shape=frame.shape)
    
    frame[rr_mouse, cc_mouse] = 1

    cp , lp , rp = mouse.get_gaze()
    
    #rr_view , cc_view = draw.polygon([mouse.pos[1], lp[1], rp[1]],[mouse.pos[0],lp[0],rp[0]],shape=frame.shape)
    
    #frame[rr_view, cc_view] = 1
    lims = frame.shape[:-1]
    print(lims)
    for i in range(2):    
        if (lp[i] >= lims[i]):
            lp[i] = lims[i] - 1
        if (rp[i] >= lims[i]):
            rp[i] = lims[i] - 1

    rr_lp , cc_lp = draw.line(int(mouse.pos[1]), int(mouse.pos[0]), int(lp[1]), int(lp[0]))
    rr_rp , cc_rp = draw.line(int(mouse.pos[1]), int(mouse.pos[0]), int(rp[1]), int(rp[0]))
    rr_cp , cc_cp = draw.line(int(mouse.pos[1]), int(mouse.pos[0]),int(cp[1]),int(cp[0]))

    frame[rr_cp, cc_cp, 0] = 1
    frame[rr_lp,cc_lp, 1] = 1
    frame[rr_rp,cc_rp, 1] = 1
    
    return frame

#def random_walk(h,w,stepsize):


if __name__ == '__main__':
    screen_id = 1

    screen = screeninfo.get_monitors()[-1]
    x = screen.x
    y = screen.y
    w = screen.width
    h = screen.height

    print(F"{w},{h}")

    img1 = np.ones((h,w,3), dtype=np.float32)
    
    n = 200
    speed1 = 8
    speed2 = 100
    max_radius = min([x,y])
    loom = looming(speed1,speed2,n,max_radius)

    shadows = []
    for i in range(n):        
        shadows.append(draw_circle(h,w,loom[i]*max_radius,(w/2,h/2)))

    w_name = 'projector'
    wind = cv.namedWindow(w_name,cv.WND_PROP_FULLSCREEN)
    cv.moveWindow(w_name, x - 1, y - 1)
    cv.setWindowProperty(w_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    mouse = Mouse(1,view_angle=90,view_range=200)

    for shad in shadows:
        img2 = np.zeros((h,w,3), dtype=np.float32)

        im = draw_mouse(mouse, img2)

        cv.imshow(w_name, im)
        cv.waitKey(100)

        mouse.random_gaze(15)
        mouse.random_step(50)
        #mouse.step(0,10)
    
    cv.destroyAllWindows()

    
