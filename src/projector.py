import cv2 , time , screeninfo
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

class Mouse():
    def __init__(self,n,r):
        self.r = r
        self.trigger_positions = np.zeros((n,n),dtype=object)
        self.stimuli_rad = np.array([r,r])

    def add_stimuli_positions(self,r):
        for i in range(-1,len(self.trigger_positions)-1):
            for j in range(-1,len(self.trigger_positions[0])):
                delta_r = np.array([i,j])
                self.trigger_positions[i,j] = lambda pos : pos - r*delta_r

    def get_stimuli_position(self,idx):
        """
        Returns location function w.r.t. mouse coordinates
        (For stimuli presentation at specific location)
        """
        return self.trigger_positions[idx]

class TriggerEllipse():
    """
    Adds an elliptic trigger with position pos and radii rad1 and rad2
    """
    def __init__(self,pos,rad1,rad2,trigger_dict,stimuli):
        """
        pos = center
        rad1 = minor semi-axis radius
        rad2 = major semi-axis radius
        rad1 = rad2 gives a circle with radius rad1

        proj = Projector object
        """
        self.__center = pos
        self.__rad = np.array([rad1,rad2])
        self.trigger_dict = trigger_dict
        self.__stimuli = stimuli
        self.__idx = self.trigger_dict['trigger_index']
        print(f"Created ellipse with center {self.__center} and radius {self.__rad}")

    def get_stimuli(self):
        return self.__stimuli

    def check(self,point):
        """
        Checks if given coordinate lies within the ellipse
        and returns boolean
        """
        
        #print(f'trigger {self.__idx}: {np.sum(( (point - self.__center)/self.__rad)**2 )}')
        return np.sum(( (point - self.__center)/self.__rad)**2 ) <= 1

    def draw(self,shape):
        rr , cc = draw.ellipse_perimeter(int(self.__center[1]),int(self.__center[0]),r_radius=int(self.__rad[1]),c_radius=int(self.__rad[0]),shape=shape)
        return [rr , cc]


class Projector():
    def __init__(self,bgr_col=1,use_fullscreen=True,push_window=4000,rotate=False):
        print("Getting monitors")
        self.screens = screeninfo.get_monitors()
        self.rotate = rotate

        self.__font = cv2.FONT_HERSHEY_SIMPLEX
        self.__proj_x_offset = push_window
    
        for i,screen in enumerate(self.screens):
            if screen.name == 'DP-3': # Name of projector
                self.__screen_id = i
                break
    
        if self.__screen_id is None: # If projector not found, use first display
            self.__screen_id = 0
        
        self.screen = self.screens[self.__screen_id]

        print(f"Using screen {self.screen}")
        
        self.__fullscreen = use_fullscreen
    
        self.__w_name = 'Projector'

        if self.rotate:
            # [SIC]!
            self.w = self.screen.height
            self.h = self.screen.width
        else:
            self.w = self.screen.width
            self.h = self.screen.height

        print(f"Height and width determined: h: {self.h}, w: {self.w}")
        # ymax = 900 when using tb init
        self.set_lims(xmin=480,xmax=1470,ymin=50,ymax=1046) # Values found manually, don't change!

        self.origin = [None,None]
        self.set_origin()

        self.set_bgr_col(bgr_col)

        self.frame = self.default_frame

        self.triggers = {}
        print("Projector setup completed")


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PUBLIC METHODS *#*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    
    def start(self):
        self.__create_window()
        self.show()


    def set_origin(self,x=None,y=None):
        """
        Defines origin of projector coords
        """

        if x is None:
            self.origin[0] = int((self.xmax + self.xmin) / 2.)
        else:
            print(f"set origin x: {x}")
            ver_x = self.__check_lims(x,self.xmin,self.xmax)
            self.origin[0] = ver_x

        if y is None:
            self.origin[1] = int((self.ymax + self.ymin) / 2.)
        else:
            print(f"set origin y: {y}")
            ver_y = self.__check_lims(y,self.ymin,self.ymax)
            self.origin[1] = ver_y
        


    def set_lims(self,xmin,xmax,ymin,ymax):
        """
        Sets max and min coords of projection on screen
        """

        try:
            assert (xmin < xmax)
            assert (ymin < ymax)
        except AssertionError:
            print("Not updating limits, min value larger than max for x or y!")
            print(f"xmin < xmax : {xmin} < {xmax} = {xmin <= xmax}")
            print(f"ymin < ymax : {ymin} < {ymax} = {ymin <= ymax}")
            return

        self.xmin = self.__check_lims(xmin,0,self.w)
        self.xmax = self.__check_lims(xmax,0,self.w)
        self.ymin = self.__check_lims(ymin,0,self.h)
        self.ymax = self.__check_lims(ymax,0,self.h)

    def set_bgr_col(self,gray_val=1):
        """
        Set default color of background (0 = black, 1 = white)
        Defaults to white
        """
        self.bgr_col = gray_val
        self.default_frame = np.zeros((self.h,self.w)) + self.bgr_col
        self.frame = self.default_frame

    def show(self,img=None,ms=1):
        """
        Show image img on projector for ms milliseconds
        """

        if img is None:
            img = self.default_frame
        
        """
        if self.rotate:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        """

        cv2.imshow(self.__w_name,img)
        cv2.waitKey(ms)
    

    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# STIMULUS METHODS *#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


    def create_looming_shadow(self,rad,pos,speed,max_t):
        """
        Creates a sequence of images with growing shadow with
        parameters speed1 and speed2 in max_t steps
        """

        print(f"Making shadow with speed {speed} of length {max_t}")
        shadow_seq = self._create_shadow(speed,max_t)

        img_seq = []
        for shadow in shadow_seq:
            img_seq.append(self._draw_circle(rad*shadow,pos))

        return img_seq


    def create_moving_dot(self,rad,start_pos,end_pos,max_t,n=500,random=True):
        """
        Creates a moving dot relative to projector origin for time max_t*2 ms (?)
        """
        n = 1000.
        img_seq = []

        try:
            x_seq = np.linspace(int(n*start_pos[0]),int(n*end_pos[0]),int(max_t)) / n
            y_seq = np.linspace(int(n*start_pos[1]),int(n*end_pos[1]),int(max_t)) / n

            if random:
                random_noise = np.random.normal(0,0.015,(2,int(max_t)))
                x_seq += random_noise[0,:]
                y_seq += random_noise[1,:]

            print(f"Making moving dot with rad {rad} of length {max_t} going from {start_pos} to {end_pos}")
            for x,y in zip(x_seq,y_seq):
                img_seq.append(self._draw_circle(rad,(x,y)))

        except (ValueError , IndexError, TypeError) as e:
            print("Error in creating moving dot!")
            print(e)
        
        return img_seq

    def create_fixed_dot(self,rad,start_pos,max_t):
        """
        Creates a dot relative to projector origin for time max_t*2 ms (?)
        """
        img_seq = []

        print(f"Making fixed dot with rad {rad} of length {max_t} at {start_pos}")

        for _ in range(int(max_t)):
            img_seq.append(self._draw_circle(rad,start_pos))

        return img_seq
        
    def _create_shadow(self,speed,max_t):
        """
        Creates a sequence of intensities between 0 and 1 
        for shadow growth
        """
        x = np.linspace(3,5,int(max_t))

        shadow_seq = (speed**x)
        shadow_seq /= shadow_seq[-1]

        return shadow_seq


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*# FUNCTIONAL METHODS *#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


    def get_lims(self):
        return self.xmin, self.xmax , self.ymin , self.ymax
    
    def show_lims(self,ms=0,col=0):
        """
        Shows currently set limits as bounding box
        """
        frame = self.__make_bb((self.ymax,self.xmin),(self.ymin,self.xmax),col)
        
        xmin_ymin_str = f"(x-,y-){self.xmin,self.ymin}"
        xmin_ymax_str = f"(x-,y+){self.xmin,self.ymax}"
        xmax_ymin_str = f"(x+,y-){self.xmax,self.ymin}"
        xmax_ymax_str = f"(x+,y+){self.xmax,self.ymax}"

        cv2.putText(frame,xmin_ymin_str,(self.xmin + 10,self.ymin + 30), self.__font, 0.5,(0,0,0),1)
        cv2.putText(frame,xmin_ymax_str,(self.xmin + 10,self.ymax - 10), self.__font, 0.5,(0,0,0),1)
        cv2.putText(frame,xmax_ymin_str,(self.xmax - 200,self.ymin + 30), self.__font, 0.5,(0,0,0),1)
        cv2.putText(frame,xmax_ymax_str,(self.xmax - 200,self.ymax - 10), self.__font, 0.5,(0,0,0),1)


        self.show(frame,ms)

    def show_origin(self,ms=0,col=0):
        """
        Shows location of current origin
        """
        print(self.origin)
        origin_mask = self._draw_circle(20,(0,0))
        frame = self.__draw(origin_mask,col)
        cv2.putText(frame,f"(x_o,y_o)={self.origin}",(self.origin[0],self.origin[1]), self.__font, 0.7,(0,0,0),1)

        self.show(frame,ms)

    def show_sequence(self,img_seq,ms,col):
        """
        Shows a sequence of images in seq, each for ms milliseconds.
        img_seq must be preproccessed (transposed) to be correctly displayed!
        """
        
        for rr , cc in img_seq:
            self.frame[rr,cc] = col
            self.show(self.frame,ms)
            self.frame[rr,cc] = self.bgr_col
            

    def stop(self):
        cv2.destroyWindow(self.__w_name)

    def add_trigger(self,trigger_dict,stim,return_frame=False):
        center = trigger_dict['trigger_pos_unit']
        idx = trigger_dict['trigger_index']
        
        trigger_rad1_scaled , trigger_rad2_scaled = self._translate_radii(trigger_dict)

        transl_center = self._translate_coords(center)
        ver_center = self._verify_pos(transl_center)


        self.triggers[idx] = TriggerEllipse(ver_center,trigger_rad1_scaled,trigger_rad2_scaled,trigger_dict,stim)
        
        rr , cc = self.triggers[idx].draw(self.frame.shape)

        self.triggers[idx].trigger_dict['frame'] = [rr , cc]

    def remove_trigger(self,idx):
        try:
            del self.triggers[idx]

        except KeyError:
            pass

    def get_triggers_perimiter(self,col=0):
        print("Showing trigger perimiter")
        frame = self.frame.copy()
        for trigger in list(self.triggers.values()):
            rr , cc = trigger.trigger_dict['frame']
            frame[rr,cc] = col

        return frame

    def check_triggers(self,pos):
        # Present stimulus at position relative to mouse
        # TODO : create interface
        # TODO : how create stimuli ahead? 
        for trigger in self.triggers.values():
            if trigger.check(pos):
                print("Trigger triggered!")
                return trigger
        
        return None

            
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#* EXT CLASS METHODS *#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

    def _translate_radii(self,trigger_dict):
        """
        Scales trigger ellipse radii from [0,1] range to projector coordinates
        """

        rad1 = trigger_dict['trigger_rad1_unit']
        rad2 = trigger_dict['trigger_rad2_unit']

        xmin , xmax , ymin , ymax = self.get_lims()

        x_range = abs(xmax - xmin)
        y_range = abs(ymax - ymin)

        rad1_scaled = rad1*(x_range)/2.
        rad2_scaled = rad2*(y_range)/2.

        return rad1_scaled , rad2_scaled

    def _translate_coords(self,pos):
        """
        Translates from [-1,1]x[-1,1] range
        to [0,w]x[0,h].
        This is for easy of use for lab guys.

        Examples:
        pos = (x,y)
        (x,y) = (0,0) corresponds to projector origin
        (x,y) = (1,0) corresponds to x_max and y at origin
        (x,y) = (0,-0.5) corresponds to x at origin and y
                halfway to bottom from origin
        """

        x , y = pos
        
        xmin_o = np.abs(self.xmin - self.origin[0])
        xmax_o = np.abs(self.xmax - self.origin[0])
        ymin_o = np.abs(self.ymin - self.origin[1])
        ymax_o = np.abs(self.ymax - self.origin[1])

        if x < 0:
            delta_x = xmin_o
        else:
            delta_x = xmax_o
        
        transl_x = self.origin[0] + x*delta_x

        if y < 0:
            delta_y = ymin_o
        else:
            delta_y = ymax_o

        transl_y = self.origin[1] - y*delta_y

        return (transl_x , transl_y)


    def _verify_pos(self,pos):
        x , y = pos
        ver_x = self.__check_lims(x,self.xmin,self.xmax)
        ver_y = self.__check_lims(y,self.ymin,self.ymax)

        return np.array([ver_x,ver_y])


    def _draw_circle(self,rad,pos,wrt_origin=True):
        """
        Returns indices (rr - rows, cc - cols) of a circle
        with radius rad at position pos with color col
        to apply to a frame with size proj.frame.shape
        """

        if wrt_origin:
            pos = self._translate_coords(pos)

        ver_pos = self._verify_pos(pos)

        rr , cc = draw.disk((ver_pos[1],ver_pos[0]),radius=rad,shape=self.default_frame.shape)
        
        return [rr , cc]


    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    #*#*#*#*#*#*# PRIVATE METHODS #*#*#*#*#*#*#
    #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
    
    def __draw(self,mask,col):
        rr , cc = mask
        frame = self.default_frame.copy()
        frame[rr,cc] = col
        
        return frame

    def __create_window(self):
        self.__window = cv2.namedWindow(self.__w_name)
        print(self.__proj_x_offset)
        cv2.moveWindow(self.__w_name, self.__proj_x_offset ,0)
        cv2.setWindowProperty(self.__w_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            

    
    def __make_bb(self,start_coord,end_coord,col):
        """
        Creates bounding box to show limits of start_coord and end_coord
        xs = (xmin, xmax) -> coords for upper and lower bounds of bb x coords
        ys = (ymin, ymax) -> coords for upper and lower bounds of bb y coords
        """
        frame = self.default_frame.copy()

        rr , cc = draw.rectangle_perimeter(start_coord,end=end_coord,shape=self.default_frame.shape)

        frame[rr,cc] = col

        return frame


    def __check_lims(self,val,min_lim,max_lim):
        """
        Verify that min_lim <= val <= max_lim
        If true, return val, else return truncated value
        """

        if min_lim <= val:
            if val <= max_lim:
                ver_val = val
            else:
                #print(f"val exceeds max_lim: {val} > {max_lim}")
                ver_val = max_lim
        else:
            #print(f"val violates min_lim: {val} < {min_lim}")
            ver_val = min_lim
        
        return ver_val


def main():
    proj = Projector()
    proj.start()
    #print(screeninfo.get_monitors())
    #screen = screeninfo.get_monitors()[1]
    
    #h = screen.height
    #w = screen.width
    
    
    #proj.set_lims(xmin=100,xmax=400,ymin=500,ymax=1000)

    # x1 = 620
    # x2 = 1390
    #x = 770 #!!
    #y = 750 #!! 
    #y = 300 , 1050

    #mouse = Mouse(9,10)
    #mouse.add_stimuli_positions(10)
    #f = mouse.get_stimuli_position([0,0])
    #print(f(10))

    proj.set_lims(624,1386,304,1046) #!!!! don't change!!!
    proj.set_origin()
    proj.show_lims()

    """
    while True:
        try:
            inp = input(">").split(' ')
            print(inp)
            a ,  b , c , d = inp
            proj.set_lims(int(a),int(b),int(c),int(d)) #!!!! don't change!!!
            proj.set_origin()
            proj.show_lims()
        except ValueError:
            print('ValueError')
            if inp == 'q':
                break
            pass
    """
    
    dot = proj.create_moving_dot(20,(-1,-1),(1,1),200)
    proj.show_sequence(dot,1,0)

    dot = proj.create_moving_dot(200,(-1,-1),(1,1),100)
    proj.show_sequence(dot,2,0)
    
    
    for i,x in enumerate(np.linspace(-.6,.6,3)):
        for j,y in enumerate(np.linspace(-.4,.4,3)):
            proj.add_trigger((x,y),(80,30),i*3+j,{})
    

    proj.show_triggers_perimiter(0.5)
    proj.show_origin()

    seq = proj.create_looming_shadow(250,(0,0),4,100)
    proj.show_sequence(seq,1,0.5)




if __name__ == '__main__':
    main()