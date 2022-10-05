import time , sys , os, serial , logging , clipboard
import numpy as np

from threading import Event , Lock, Thread
import multiprocessing as mp
from queue import Empty , Queue , Full

from optionHandler import OptionHandler
from parsers import dirParser , fileParser
from directoryTracker import DirTracker

import tkinter as tk
from tkinter import scrolledtext , messagebox , TclError

class TkUICommunicator:
    def __init__(self,read_queue, write_queue, proj_queue, ard_ready_event, user_ready, opt_dict, dir_tracker):
    
        self.__is_stopped = False
        self.__stim_options = ['Shadow','Fixed dot','Moving dot']

        self.__ard_ready_event = ard_ready_event
        self.__user_ready = user_ready
        self.__write_queue = write_queue
        self.__read_queue = read_queue
        self.__proj_queue = proj_queue
        self.__dir_tracker = dir_tracker
        self.__opt_dict = opt_dict

        self.__initial_mouse_ID()



    ################### PUBLIC METHODS ####################

    def start(self):
        logging.debug("UI: Setting up new window")
        self.__setup_window()
        logging.debug("UI: Creating ard cmd buttons")
        self.__create_ard_cmd_buttons()
        logging.debug("UI: Creating stimulus buttons")
        self.__create_proj_cmd_buttons()
        logging.debug("UI: check_in_queue")
        self.__check_in_queue()
        logging.debug("UI: mainloop")
        self.__root.mainloop()
    
    def ask_for_start(self):
        return self.__ask_for_start()

    def query_timestamps(self,pickle_savename):
        tmp_root = tk.Tk()
        tmp_root.withdraw()
        msgbox_result = messagebox.askyesno(title="Add timestamps?", message="Do you want to add timestamps automatically?\nThis may take a while (10-40 min) depending on video size. They can also be added at a later time.")
        if not(msgbox_result):
            cmd = F"python pickle_timestamps.py {pickle_savename}"
            do_copy_path = messagebox.askyesno(title="Add timestamps later", message=F"You can add the timestamps later automatically by using the following command from the Anaconda Prompt:\n\n{cmd}\n\nCopy command to clipboard?")
            if do_copy_path:
                clipboard.copy(cmd)
        else:
            messagebox.showinfo(title="Timestamp progress", message=F"See the terminal for progress.")
        tmp_root.destroy()
        return msgbox_result

    def query_dlc(self,pickle_savename):
        tmp_root = tk.Tk()
        tmp_root.withdraw()
        msgbox_result = messagebox.askyesno(title="Apply DLC?", message="Do you want to extract DLC data automatically?\nThis may take a while depending on video size. They can also be added at a later time.")
        if not(msgbox_result):
            cmd = F"python apply_dlc.py --video_list {pickle_savename}"
            do_copy_path = messagebox.askyesno(title="Apply DLC later", message=F"You can apply DLC later automatically by using the following command from the Anaconda Prompt:\n\n{cmd}\n\nCopy command to clipboard?")
            if do_copy_path:
                clipboard.copy(cmd)
        else:
            messagebox.showinfo(title="DLC progress", message=F"See the terminal for progress.")
        tmp_root.destroy()
        return msgbox_result

    def show_info(self, msg):
        messagebox.showinfo(title="Info", message=msg)

    def show_error(self, err_msg):
        messagebox.showerror(title="Error", message=F"An error occured (see terminal)!:\n\n{err_msg}")
        logging.debug(err_msg)

    def get_mouse_ID(self):
        return self.__mouse_var.get()

    def update_mouse_ID(self):
        self.__setup_mouse_ID_query()

    def stop(self):
        self.__root.destroy()



    ################## INTERNAL METHODS ###################

    # Setting up TkInter environment

    def __restart(self):
        logging.debug("UI: __restart")

        logging.debug("UI: clearing ard_ready_event")
        self.__ard_ready_event.clear()
        logging.debug("UI: Clearing user_ready")
        self.__user_ready.clear()
        
        self.__send_to_arduino('R')

        logging.debug("UI: Put R in queue, waiting for arduino to put messages in queue")
        # Wait for last input to arrive
        time.sleep(3)

        logging.debug("UI: Emptying queue")
        self.__empty_queue()
        
        _ = messagebox.showinfo("OK","Press OK to continue")

        self.__root.withdraw()


        # Stop checking in_queue
        logging.debug("UI: Cancelling queue check")
        self.__root.after_cancel(self.__after_ID_in)
        self.__is_stopped = True
        
        logging.debug("UI: Setting up mouse query")
        self.__setup_mouse_ID_query()

        logging.debug("UI: Root quit")
        self.__root.quit()

    def __ask_for_start(self):

        tmp_root = tk.Tk()
        tmp_root.withdraw()
        ans = messagebox.askokcancel('Start' ,'Press OK to start next run.')
        tmp_root.destroy()

        return ans
        
    def __setup_window(self):
        # Setting up windows and buttons etc
        self.__root = tk.Tk()
        self.__root.title("Input window")
        self.__root.geometry('1000x1000')
        self.__root.focus()
        self.__center_window(self.__root)
        
        self.__menu_frame = tk.LabelFrame(self.__root,text='Menu')
        self.__menu_frame.grid(  row=0, columnspan=1, sticky='W',
                                padx=5,pady=3)

        self.__cmd_frame = tk.LabelFrame(self.__root,text='Commands')
        self.__cmd_frame.grid(  row=4, columnspan=1, sticky='W',
                                padx=5,pady=5)

        self.__text_frame = tk.LabelFrame(self.__root,text='Display')
        self.__text_frame.grid( row=0, column=5, columnspan=5, rowspan=9,
                                padx=5, pady=5)

        self.__stim_frame = tk.LabelFrame(self.__root,text='Stimulus Menu')
        self.__stim_frame.grid( row=0, column=11, rowspan=8, columnspan=3,
                                padx=3)

        self.__kalman_frame = tk.LabelFrame(self.__root,text='Kalman Filter')
        self.__kalman_frame.grid( row=10, column=9, rowspan=3, columnspan=3,
                                padx=3)

        self.__exit_frame = tk.LabelFrame(self.__root,text='Exit')
        self.__exit_frame.grid( row=10)

        # Restart button
        self.__restart_btn = tk.Button(self.__menu_frame, text="Start new run", command=self.__restart)
        self.__restart_btn.grid(column=1,row=1)

        # Clear text button
        self.__clrbtn = tk.Button(self.__menu_frame, text="Clear text", command=self.__clear_text)
        self.__clrbtn.grid(column=1,row=2)

        # Exit button
        self.__qt_btn = tk.Button(self.__exit_frame, text="Save and exit", command=self.__exit)
        self.__qt_btn.grid(column=4,row=1)

        # Text scroll window
        self.__scrTxt = scrolledtext.ScrolledText(self.__text_frame, width=40,height=30,wrap=tk.WORD,font=('Helvetica',10))
        self.__scrTxt.grid(column=3,row=12)

        print("TK: setup_windows done")

    def __create_ard_cmd_buttons(self):
        self.__cmd_buttons = {}
        for i,(k,v) in enumerate(self.__opt_dict.items()):
            if not((k == 'Q') | (k == 'R')):
                self.__cmd_buttons[k] = tk.Button(self.__cmd_frame, text=v, command=lambda k=k: self.__send_to_arduino(k))
                self.__cmd_buttons[k].grid(column=0,row=i)

    def __create_stim_options(self):
        self.__stim_var = tk.StringVar()
        self.__stim_var.set("Stimulus Type")
        self.__lbl_stim = tk.Label(master=self.__stim_frame, textvariable=self.__stim_var)
        self.__lbl_stim.grid(column=1,row=1)

        self.__stim_variable = tk.StringVar(self.__stim_frame)
        self.__stim_variable.set(self.__stim_options[1])
        
        self.__opt_menu = tk.OptionMenu(self.__stim_frame, self.__stim_variable, *self.__stim_options, command = self.__selection_event)
        self.__opt_menu.config(width=10,textvariable=self.__stim_variable)
        self.__opt_menu.grid(column=2,row=1,sticky='ew')

        self.__lbl_stim_rad1 = tk.Label(master=self.__trigger_frame, text='Trigger x-radius')
        self.__lbl_stim_rad1.grid(column=0,row=7)
        
        self.__stim_rad1_double_var = tk.DoubleVar(master=self.__stim_frame)
        self.__stim_rad1_slider = tk.Scale(self.__trigger_frame, from_=0., to_=100., variable=self.__stim_rad1_double_var, resolution=1., showvalue=1, orient='horizontal', command = lambda ev : self.__draw_current_oval())
        self.__stim_rad1_slider.set(10.)
        self.__stim_rad1_slider.grid(column=1,row=7)

        self.__lbl_stim_rad2 = tk.Label(master=self.__trigger_frame, text='Trigger y-radius')
        self.__lbl_stim_rad2.grid(column=0,row=8)

        self.__stim_rad2_double_var = tk.DoubleVar(master=self.__stim_frame)
        self.__stim_rad2_slider = tk.Scale(self.__trigger_frame, from_=0., to_=100., variable=self.__stim_rad2_double_var, resolution=1., showvalue=1, orient='horizontal', command = lambda ev : self.__draw_current_oval())
        self.__stim_rad2_slider.set(10.)
        self.__stim_rad2_slider.grid(column=1,row=8)

    def __color_chosen_oval(self,event):
        print("Coloring chosen oval")
        #self.__draw_all_ovals()
        if self.__trigger_var.get():
            trigger = self.__triggers[self.__trigger_var.get()]

            x , y = trigger['trigger_pos']
            xrad = trigger['trigger_rad1']
            yrad = trigger['trigger_rad2']
            
            self.__trigger_canvas.create_oval(x-xrad, y-yrad, x+xrad, y+yrad, outline="#1f1")

    def __create_proj_cmd_buttons(self):
        self.__canvas_shape = (200,200)
        
        self.__trigger_idx = 0

        self.__midpt_x = self.__canvas_shape[0] / 2 
        self.__midpt_y = self.__canvas_shape[1] / 2
        
        self.__xr = self.__canvas_shape[0]/30
        self.__yr = self.__canvas_shape[0]/18
        
        self.__x = [self.__xr,self.__xr]
        self.__y = [self.__yr,self.__yr]

        self.__trigger_x = 0
        self.__trigger_y = 0

        self.__curr_start_circ = None
        self.__curr_end_circ = None

        self.__curr_angles = [None,None]

        self.__current_trigger = None

        self.__drawn_r = [100,100]

        self.__curr_connections = [None]

        self.__triggers = {0:None}

        self.__angle = [0,0]

        self.__connected_lines = [None]

        self.__drawpoint_x = [None,None]
        self.__drawpoint_y = [None,None]


        """ 
        Options for size of stimulus
        """
        self.__lbl_size = tk.Label(master=self.__stim_frame, text='Radius')
        self.__lbl_size.grid(column=1,row=2)

        self.__size_double_var = tk.DoubleVar(master=self.__stim_frame)
        self.__size_double_var.set(30.)
        self.__size_slider = tk.Scale(self.__stim_frame, from_=0., to_=100., variable=self.__size_double_var, resolution=1., showvalue=1, orient='horizontal')
        self.__size_slider.grid(column=2,row=2)
        self.__size_textbox = tk.Entry(self.__stim_frame, textvariable=self.__size_double_var,width=4)
        self.__size_textbox.grid(column=3,row=2)

        
        """
        Options for speed of stimulus
        """
        self.__lbl_speed = tk.Label(master=self.__stim_frame, text='Speed')
        self.__lbl_speed.grid(column=1,row=3)

        self.__speed_double_var = tk.DoubleVar(master=self.__stim_frame)
        self.__speed_double_var.set(50.)
        self.__speed_slider = tk.Scale(self.__stim_frame, from_=0., to_=100., variable=self.__speed_double_var, resolution=1., showvalue=1, orient='horizontal')
        self.__speed_slider.grid(column=2,row=3)
        self.__speed_textbox = tk.Entry(self.__stim_frame, textvariable=self.__speed_double_var,width=4)
        self.__speed_textbox.grid(column=3,row=3)

        """
        Options for intensity (greylevel) of stimulus
        """

        self.__lbl_intensity = tk.Label(master=self.__stim_frame, text='Intensity')
        self.__lbl_intensity.grid(column=1,row=4)

        self.__intensity_double_var = tk.DoubleVar(master=self.__stim_frame)

        self.__intensity_slider = tk.Scale(self.__stim_frame, from_=0., to_=1., variable=self.__intensity_double_var, resolution=0.01, showvalue=1, orient='horizontal')
        self.__intensity_slider.set(0.0)
        self.__intensity_slider.grid(column=2,row=4)
        self.__intensity_textbox = tk.Entry(self.__stim_frame, textvariable=self.__intensity_double_var,width=4)
        self.__intensity_textbox.grid(column=3,row=4)

        """
        Options for position of stimulus
        """

        self.__canvas_frame = tk.LabelFrame(master=self.__stim_frame,text='Stimulus Position')
        self.__canvas_frame.grid(column=1,row=10,columnspan=3,rowspan=5,sticky='W')
        self.__pos_canvas = tk.Canvas(master=self.__canvas_frame, 
           width=self.__canvas_shape[0],height=self.__canvas_shape[1])

        self.__pos_canvas.grid(column=1,row=0)

        self.__pos_canvas.create_rectangle(0, 0, self.__canvas_shape[0], self.__canvas_shape[1], fill="white")
        self.__pos_canvas.bind('<Button-1>',lambda e : self.__mouseclick(e,True))
        self.__pos_canvas.bind('<Button-3>',lambda e : self.__mouseclick(e,False))

        self.__xlab = tk.Label(master=self.__canvas_frame, text='x')
        self.__xlab.grid(column=2,row=1)

        self.__ylab = tk.Label(master=self.__canvas_frame, text='y')
        self.__ylab.grid(column=3,row=1)

        self.__startpos_lbl = tk.Label(master=self.__canvas_frame, text='Right click = Start pos (blue)')
        self.__startpos_lbl.grid(column=1,row=2)

        self.__startpos_x_double_var = tk.DoubleVar(master=self.__canvas_frame, value=0)
        self.__startpos_y_double_var = tk.DoubleVar(master=self.__canvas_frame, value=0)

        self.__startpos_textbox_x = tk.Entry(self.__canvas_frame, textvariable=self.__startpos_x_double_var,width=4)
        self.__startpos_textbox_x.grid(column=2,row=2)

        self.__startpos_textbox_y = tk.Entry(self.__canvas_frame, textvariable=self.__startpos_y_double_var,width=4)
        self.__startpos_textbox_y.grid(column=3,row=2)

        self.__endpos_lbl = tk.Label(master=self.__canvas_frame, text='Left click = End pos (red)')
        self.__endpos_lbl.grid(column=1,row=3)

        self.__endpos_x_double_var = tk.DoubleVar(master=self.__canvas_frame, value=0)
        self.__endpos_y_double_var = tk.DoubleVar(master=self.__canvas_frame, value=0)

        self.__endpos_textbox_x = tk.Entry(self.__canvas_frame, textvariable=self.__endpos_x_double_var,width=4)
        self.__endpos_textbox_x.grid(column=2,row=3)

        self.__endpos_textbox_y = tk.Entry(self.__canvas_frame, textvariable=self.__endpos_y_double_var,width=4)
        self.__endpos_textbox_y.grid(column=3,row=3)


        """
        Options for mirroring angle
        """

        self.__mirror_int_var = tk.IntVar(master=self.__canvas_frame)
        self.__mirror_int_var.set(0)
        
        self.__mirror_check = tk.Checkbutton(self.__canvas_frame, text='Mirror stimulus', variable=self.__mirror_int_var, onvalue=1, offvalue=0, command=self.__apply_mirroring)
        self.__mirror_check.grid(column=2,row=5)

        """
        Options for angular radius
        """


        self.__lbl_angle = tk.Label(master=self.__canvas_frame, text='Angular radius')
        self.__lbl_angle.grid(column=1,row=6)

        self.__angle_double_vars = [tk.DoubleVar(master=self.__canvas_frame),tk.DoubleVar(master=self.__canvas_frame)]
        self.__angle_double_vars[0].set(-30)
        self.__angle_double_vars[1].set(30)

        self.__angle_rad_int_var = tk.IntVar(master=self.__canvas_frame)
        self.__angle_rad_int_var.set(70)

        self.__rad = [self.__angle_rad_int_var.get(),self.__angle_rad_int_var.get()]
        self.__rad_max = 100.
        self.__rad_slider = tk.Scale(self.__canvas_frame, from_=self.__yr, to_=self.__rad_max, variable=self.__angle_rad_int_var, resolution=1., showvalue=1, orient='horizontal', command= lambda ev : self.__update_lines())
        self.__rad_slider.grid(column=2,row=6)
        self.__rad_textbox = tk.Entry(self.__canvas_frame, textvariable=self.__angle_rad_int_var,width=5)
        self.__rad_textbox.grid(column=3,row=6)

        """
        Options for relative to mouse 
        """

        
        self.__rel2mouse_bool_var = tk.BooleanVar(master=self.__canvas_frame)

        self.__rel2mouse_check = tk.Checkbutton(self.__canvas_frame, text='Stimulus relative to mouse', variable=self.__rel2mouse_bool_var, onvalue=1, offvalue=0, command=self.__rel_to_mouse)
        self.__rel2mouse_check.grid(column=1,row=5)


        """
        Option for adding triggers
        """
        self.__trigger_frame = tk.LabelFrame(master=self.__stim_frame,text='Trigger Settings')
        self.__trigger_frame.grid(column=1,row=20,columnspan=3,sticky='W')

        self.__trigger_canvas = tk.Canvas(master=self.__trigger_frame,
            width=self.__canvas_shape[0],height=self.__canvas_shape[1])
        self.__trigger_canvas.grid(column=1,row=1)

        self.__trigger_canvas.create_rectangle(0, 0, self.__canvas_shape[0], self.__canvas_shape[1], fill="white")
        self.__trigger_canvas.bind('<Button-1>',lambda e : self.__place_trigger(e))


        self.__trigger_var = tk.IntVar(master=self.__trigger_frame)
        self.__trigger_var.set(0)

        self.__triggerops = [0]
        #list(self.__triggers.keys())
        self.__saved_triggers_menu = tk.OptionMenu(self.__trigger_frame, self.__trigger_var, *self.__triggerops,command = lambda ev : self.__draw_all_ovals())
        self.__saved_triggers_menu.config(width=10,textvariable=self.__trigger_var)
        self.__saved_triggers_menu.grid(column=0,row=2)
        
        
        self.__command_frame = tk.LabelFrame(master=self.__stim_frame,text='Commands')
        self.__command_frame.grid(column=1,row=25)
        
        # Trigger button!
        self.__present_trigger_btn = tk.Button(self.__command_frame, text='Present Stimulus', command = lambda : self.__send_stimulus('Present'))
        self.__present_trigger_btn.grid(column=1,row=1)

        # Kalman button
        self.__kalman_btn = tk.Button(self.__kalman_frame, text='Calibrate Kalman', command = self.__calibrate_kalman)
        self.__kalman_btn.grid(column=1,row=0)

        # Add trigger
        self.__add_trigger_btn = tk.Button(self.__command_frame, text='Add Trigger', command = self.__add_trigger)
        self.__add_trigger_btn.grid(column=2,row=1)

        self.__remove_trigger_btn = tk.Button(self.__command_frame, text='Remove Chosen Trigger', command = self.__remove_trigger)
        self.__remove_trigger_btn.grid(column=3,row=1)
        
        self.__bgr_variable = tk.DoubleVar(master=self.__command_frame)
        self.__bgr_slider = tk.Scale(self.__command_frame, from_=0., to_=1., variable=self.__bgr_variable, resolution=0.01, showvalue=1, orient='horizontal')
        self.__bgr_slider.grid(column=1,row=2)

        self.__background_col_btn = tk.Button(self.__command_frame, text='Set Background Color', command = self.__send_background_value)
        self.__background_col_btn.grid(column=1,row=3)

        self.__random_bool_var = tk.BooleanVar(master=self.__command_frame)
        self.__random_check = tk.Checkbutton(self.__command_frame, text='Add random noise', variable=self.__random_bool_var, onvalue=1, offvalue=0)
        self.__random_check.grid(column=2,row=2)

        self.__create_stim_options()

        self.__rel_to_mouse()

    def __send_background_value(self):
        bgr_col = self.__bgr_variable.get()
        self.__send_to_projector(['Background',bgr_col])

    def __draw_current_oval(self):
        if self.__trigger_x and self.__trigger_y:
            self.__trigger_xrad = self.__stim_rad1_double_var.get()
            self.__trigger_yrad = self.__stim_rad2_double_var.get()

            self.__trigger_canvas.delete(self.__current_trigger)

            self.__current_trigger = self.__trigger_canvas.create_oval( self.__trigger_x-self.__trigger_xrad, 
                                                                        self.__trigger_y-self.__trigger_yrad, 
                                                                        self.__trigger_x+self.__trigger_xrad, 
                                                                        self.__trigger_y+self.__trigger_yrad    )

    def __place_trigger(self,event):
        self.__trigger_x = self.__trigger_canvas.canvasx(event.x)
        self.__trigger_y = self.__trigger_canvas.canvasy(event.y)

        self.__draw_all_ovals()
        self.__draw_current_oval()


    def __draw_all_ovals(self):
        self.__trigger_canvas.create_rectangle(0, 0, self.__canvas_shape[0], self.__canvas_shape[1], fill="white")

        for trigger in list(self.__triggers.values()):
            if trigger:
                idx = trigger['trigger_index']
                col = "#1f1" if idx == self.__trigger_var.get() else "#f11"
                x , y = trigger['trigger_pos']
                xrad = trigger['trigger_rad1']
                yrad = trigger['trigger_rad2']
                self.__trigger_canvas.create_oval(x-xrad, y-yrad, x+xrad, y+yrad, outline=col)


    def __add_trigger(self):
        self.__trigger_idx += 1

        if self.__trigger_idx == 1:
            self.__triggers = {}

        self.__triggers[self.__trigger_idx] = self.__get_current_settings()
        self.__trigger_var.set(self.__trigger_idx)
        
        self.__update_triggers()
        
        self.__send_stimulus('Add')


    def __update_triggers(self):
        self.__triggerops = sorted(list(self.__triggers.keys()))

        self.__saved_triggers_menu['menu'].delete(0, 'end')

        for trigger in list(self.__triggers.keys()):
            self.__saved_triggers_menu['menu'].add_command(label=trigger,command=tk._setit(self.__trigger_var, trigger, lambda ev : self.__draw_all_ovals()))

        self.__draw_all_ovals()


    def __remove_trigger(self):
        idx = self.__trigger_var.get()
        
        try:
            del self.__triggers[idx]
            self.__trigger_var.set(sorted(list(self.__triggers.keys()))[-1])
            self.__saved_triggers_menu['menu'].delete(0,'end')
        except (IndexError,KeyError):
            self.__trigger_var.set(0)
            pass       

        self.__update_triggers()

        self.__remove_stimulus(idx)


    def __draw_mouse(self, **kwargs):
        arrow_size = 10
        self.__canvas_mouse = self.__pos_canvas.create_oval(self.__midpt_x-self.__xr, self.__midpt_y-self.__yr, self.__midpt_x+self.__xr, self.__midpt_y+self.__yr, **kwargs)
        self.__pos_canvas.create_line(self.__midpt_x,self.__midpt_y - arrow_size, self.__midpt_x, self.__midpt_y + arrow_size, arrow=tk.FIRST)


    def __rel_to_mouse(self):
        rel2mouse = self.__rel2mouse_bool_var.get()

        if rel2mouse:
            self.__draw_mouse()

            if self.__trigger_var.get() == 'Moving dot':
                self.__trigger_var.set('Fixed dot')
                
            self.__endpos_textbox_x.config(state=tk.DISABLED)
            self.__endpos_textbox_y.config(state=tk.DISABLED)
            self.__pos_canvas.unbind('<Button-3>')
            self.__pos_canvas.bind('<Button-3>',lambda e : self.__angle_mouseclick(e,1))

            self.__pos_canvas.delete(self.__curr_end_circ)

            self.__endpos_x_double_var.set(np.NaN)
            self.__endpos_y_double_var.set(np.NaN)
            self.__endpos_lbl.config(fg="gray")

            self.__startpos_textbox_x.config(state=tk.DISABLED)
            self.__startpos_textbox_y.config(state=tk.DISABLED)
            self.__pos_canvas.unbind('<Button-1>')
            self.__pos_canvas.bind('<Button-1>',lambda e : self.__angle_mouseclick(e,0))

            self.__pos_canvas.delete(self.__curr_start_circ)

            self.__rad_slider.config(state=tk.NORMAL)
            self.__lbl_angle.config(fg="black")
            self.__rad_textbox.config(state=tk.NORMAL)

            self.__startpos_x_double_var.set(np.NaN)
            self.__startpos_y_double_var.set(np.NaN)
            self.__startpos_lbl.config(fg="gray")

            self.__mirror_check.config(state=tk.NORMAL)
        
            self.__opt_menu['menu'].entryconfigure(self.__stim_options[-1], state = "disabled")

        else:
            self.__endpos_textbox_x.config(state=tk.NORMAL)
            self.__endpos_textbox_y.config(state=tk.NORMAL)

            self.__endpos_x_double_var.set(0)
            self.__endpos_y_double_var.set(0)

            self.__pos_canvas.bind('<Button-3>',lambda e : self.__mouseclick(e,False))
            self.__endpos_lbl.config(fg="black")

            self.__startpos_textbox_x.config(state=tk.NORMAL)
            self.__startpos_textbox_y.config(state=tk.NORMAL)

            self.__startpos_x_double_var.set(0)
            self.__startpos_y_double_var.set(0)

            self.__rad_slider.config(state=tk.DISABLED)
            self.__lbl_angle.config(fg="gray")
            self.__rad_textbox.config(state=tk.DISABLED)

            self.__pos_canvas.bind('<Button-1>',lambda e : self.__mouseclick(e,True))
            self.__startpos_lbl.config(fg="black")

            self.__pos_canvas.create_rectangle(0, 0, self.__canvas_shape[0], self.__canvas_shape[1], fill="white")

            self.__opt_menu['menu'].entryconfigure(self.__stim_options[-1], state = "normal")
            
            #self.__stim_options.append('Moving dot')
            #self.__create_stim_options()

            self.__mirror_check.config(state=tk.DISABLED)


    def __apply_mirroring(self):
        on = self.__mirror_int_var.get()

        print(f"Mirroring: {on}")
        if on:
            for i in range(2):
                new_x , new_y = self.__get_mirror_coords(i)

                self.__x += [new_x]
                self.__y += [new_y]

                self.__drawn_r += [None]
                self.__rad += [None]
                self.__angle += [None]
                self.__drawpoint_x += [None]
                self.__drawpoint_y += [None]
                self.__curr_angles += [None]
                self.__angle_double_vars += [tk.DoubleVar(master=self.__stim_frame)]

            self.__curr_connections += [None]

            for i in range(2):
                self.__update_mirror_values(i + 2)

            for i in range(2):
                self.__draw_lines(self.__x[i + 2],self.__y[i + 2],i+2)
        else:
            self.__pos_canvas.delete(self.__curr_angles[-1])
            self.__pos_canvas.delete(self.__curr_angles[-2])

            try:
                self.__pos_canvas.delete(self.__connected_lines[-1])
                self.__pos_canvas.delete(self.__curr_connections[-1])
                #self.__pos_canvas.delete(self.__curr_connections[-2])
                #del self.__connected_lines[-1]
                
            except IndexError as e:
                print("Indexerror!")
                print(e)
                pass
            finally:
                del self.__x[-2:]
                del self.__y[-2:]
                del self.__drawn_r[-2:]
                del self.__rad[-2:]
                del self.__angle[-2:]
                del self.__drawpoint_x[-2:]
                del self.__drawpoint_y[-2:]
                del self.__curr_angles[-2:]
                del self.__curr_connections[-1]
                del self.__angle_double_vars[-2:]

    def __get_mirror_coords(self,idx):
        new_x = - self.__x[idx]
        new_y = self.__y[idx]

        return new_x , new_y

    def __angle_mouseclick(self,event,idx):
        print(f"Clicked! Button {idx}")
        print(f"current ocnnections {self.__curr_connections}")

        self.__x[idx] = self.__pos_canvas.canvasx(event.x) - self.__canvas_shape[0]/2
        self.__y[idx] = -(self.__pos_canvas.canvasy(event.y) - self.__canvas_shape[1]/2)
        self.__draw_lines(self.__x[idx],self.__y[idx],idx)

        if self.__mirror_int_var.get():
            new_x , new_y = self.__get_mirror_coords(idx)
            
            self.__x[idx + 2] = new_x
            self.__y[idx + 2] = new_y

            self.__update_mirror_values(idx + 2)
            self.__draw_lines(self.__x[idx + 2],self.__y[idx + 2],idx + 2)

    def __update_lines(self):
        for i,[x,y] in enumerate(zip(self.__x,self.__y)):
            self.__draw_lines(x,y,i)

    def __remove_line(self,idx):
        try:
            coord_idx = idx // 2 # Maps 0 and 1 to 0, 2 and 3 to 1 etc.
            print(self.__curr_angles)
            self.__pos_canvas.delete(self.__curr_angles[idx])
            print(self.__curr_connections)
            self.__pos_canvas.delete(self.__curr_connections[coord_idx])
        except IndexError:
            print(f"Indexerror in remove line: {idx}")
            pass

    def __update_mirror_values(self,idx):
        print(f"Updating mirror values for index {idx}")
        x , y = self.__x[idx] , self.__y[idx]

        self.__drawn_r[idx] = np.sqrt(x**2 + y**2)

        self.__rad[idx] = self.__angle_rad_int_var.get()

        self.__angle[idx] = (np.arctan2(y,x) + 2*np.pi) % (2*np.pi)

        self.__drawpoint_x[idx] = np.cos(self.__angle[idx])*self.__rad[idx]
        self.__drawpoint_y[idx] = -np.sin(self.__angle[idx])*self.__rad[idx]

        print(f"drawpoint x : {self.__drawpoint_x}")
        print(f"drawpoint y : {self.__drawpoint_y}")

        
    def __draw_lines(self,x,y,idx):
        print(f"Drawing lines for index {idx}")
        try:
            self.__drawn_r[idx] = np.sqrt(x**2 + y**2)

            self.__rad[idx] = self.__angle_rad_int_var.get()

            self.__angle[idx] = (np.arctan2(y,x) + 2*np.pi) % (2*np.pi)
            #self.__angle[idx] = np.arccos(x/(self.__rad[idx] + 1e-5))
        
            #print(f"x: {x}")
            #print(f"y: {y}")
            #print(f"angle (rad): {self.__angle[idx]}")
            #print(f"angle (deg): {self.__angle[idx]*180/np.pi}")
            #print(f"sin({self.__angle[idx]})={np.sin(self.__angle[idx])}")
            #print(f"cos({self.__angle[idx]})={np.cos(self.__angle[idx])}")

            self.__drawpoint_x[idx] = np.cos(self.__angle[idx])*self.__rad[idx]
            self.__drawpoint_y[idx] = -np.sin(self.__angle[idx])*self.__rad[idx]

            self.__remove_line(idx)
            self.__curr_angles[idx] = self.__pos_canvas.create_line(self.__midpt_x,self.__midpt_y,self.__midpt_x + self.__drawpoint_x[idx],self.__midpt_y + self.__drawpoint_y[idx])
            self.__angle_double_vars[idx].set(self.__angle[idx])
            
            print(f"current angles: {self.__curr_angles}")
            #other_arc_idx = 1 - (idx % 2)*2 + idx # Maps 0 to 1, 1 to 0, 2 to 3 and 3 to 2
            
            coord_idx = idx // 2 # Maps 0 and 1 to 0, 2 and 3 to 1 etc.

            print(f"Mapped index {idx} to {coord_idx}")
            
            print(f"drawpoint_x = {self.__drawpoint_x}")
            print(f"drawpoint_y = {self.__drawpoint_y}")

            arc_cord1 = coord_idx*3
            arc_cord2 = int(arc_cord1 + 1 - 2*arc_cord1/3.)

            thisx = self.__midpt_x + self.__drawpoint_x[arc_cord1]
            thisy = self.__midpt_y + self.__drawpoint_y[arc_cord1]

            otherx = self.__midpt_x + self.__drawpoint_x[arc_cord2]
            othery = self.__midpt_y + self.__drawpoint_y[arc_cord2]

            print(f"Adding __curr_connections[{coord_idx}]")
            #self.__curr_connections[arc_idx] = self.__pos_canvas.create_line(thisx,thisy,otherx,othery)
            self.__curr_connections[coord_idx] = self._create_arc([thisx,thisy],[otherx,othery])
            print(self.__curr_connections)
            

        except (IndexError , TypeError):
            pass

    def _create_arc(self, p0, p1):
            extend_x = (self._distance(p0,p1) -(p1[0]-p0[0]))/2 # extend x boundary 
            extend_y = (self._distance(p0,p1) -(p1[1]-p0[1]))/2 # extend y boundary
            startAngle = np.arctan2(p0[0] - p1[0], p0[1] - p1[1]) *180 / np.pi # calculate starting angle  
            return self.__pos_canvas.create_arc(p0[0]-extend_x, p0[1]-extend_y , 
                                                p1[0]+extend_x, p1[1]+extend_y, 
                                                extent=180, start=90+startAngle, style=tk.ARC)  

    def _distance(self, p0, p1):
        '''calculate distance between 2 points'''
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)   

    def __calibrate_kalman(self):
        """
        Calibrate Kalman filter now
        """
        self.__send_to_projector(['Calibrate',None])


    def __remove_stimulus(self,idx):
        self.__send_to_projector(['Remove',idx])


    def __get_current_settings(self):
        angles = [var.get() - np.pi/2 for var in self.__angle_double_vars]
        x , y = self.__transform_coords(self.__trigger_x,self.__trigger_y)

        rad1 = 2*self.__stim_rad1_double_var.get()/self.__canvas_shape[0]
        rad2 = 2*self.__stim_rad2_double_var.get()/self.__canvas_shape[1]

        dist_unit = (self.__angle_rad_int_var.get() - self.__yr) / (self.__rad_max - self.__yr)

        args = {'stim_type':self.__stim_variable.get(),
                'size':self.__size_double_var.get(),
                'start_pos':(   self.__startpos_x_double_var.get(),
                                self.__startpos_y_double_var.get()),
                'end_pos':( self.__endpos_x_double_var.get(), 
                            self.__endpos_y_double_var.get()),
                'speed':self.__speed_double_var.get(),
                'intensity':self.__intensity_double_var.get(),
                'relative_to_mouse':self.__rel2mouse_bool_var.get(),
                'distance':self.__angle_rad_int_var.get(),
                'distance_unit':dist_unit,
                'angle_intervals':angles,
                'random':self.__random_bool_var.get(),
                'trigger_pos':[self.__trigger_x,self.__trigger_y],
                'trigger_pos_unit':[x,y],
                'trigger_rad1':self.__stim_rad1_double_var.get(),
                'trigger_rad2':self.__stim_rad2_double_var.get(),
                'trigger_rad1_unit':rad1,
                'trigger_rad2_unit':rad2,
                'trigger_index':self.__trigger_idx
                }

        return args


    def __send_stimulus(self,command):
        args = self.__get_current_settings()
        
        try:
            self.__send_to_projector([command,args])
        except Full:
            print('Queue Full, please wait until current stimulus has been presented.')


    def __selection_event(self,event):
        stim = self.__stim_variable.get().lower()
        if stim == 'shadow':
            disable_endpos = True
        elif stim == 'moving dot':
            disable_endpos = False
        elif stim == 'fixed dot':
            disable_endpos = True
        else:
            return

        if disable_endpos:
            self.__endpos_textbox_x.config(state=tk.DISABLED)
            self.__endpos_textbox_y.config(state=tk.DISABLED)
            self.__pos_canvas.unbind('<Button-3>')
            self.__pos_canvas.delete(self.__curr_end_circ)

            self.__endpos_x_double_var.set(np.NaN)
            self.__endpos_y_double_var.set(np.NaN)
            self.__endpos_lbl.config(fg="gray")
        else:
            self.__endpos_textbox_x.config(state=tk.NORMAL)
            self.__endpos_textbox_y.config(state=tk.NORMAL)

            self.__endpos_x_double_var.set(0)
            self.__endpos_y_double_var.set(0)

            self.__pos_canvas.bind('<Button-3>',lambda e : self.__mouseclick(e,False))
            self.__endpos_lbl.config(fg="black")

        startpos = (self.__startpos_x_double_var.get() , self.__startpos_y_double_var.get())
        endpos = (self.__endpos_x_double_var.get() , self.__endpos_y_double_var.get())


    def __mouseclick(self,event,start_click):
        x = self.__pos_canvas.canvasx(event.x)
        y = self.__pos_canvas.canvasy(event.y)
        t_coords = self.__transform_coords(x,y)

        if start_click:
            self.__pos_canvas.delete(self.__curr_start_circ)
            self.__curr_start_circ = self.__create_circle(self.__pos_canvas,x,y,8,fill='blue')
            self.__startpos_x_double_var.set(t_coords[0])
            self.__startpos_y_double_var.set(t_coords[1])
        else:
            self.__pos_canvas.delete(self.__curr_end_circ)
            self.__curr_end_circ = self.__create_circle(self.__pos_canvas,x,y,8,fill='red')        
            self.__endpos_x_double_var.set(t_coords[0])
            self.__endpos_y_double_var.set(t_coords[1])


    def __transform_coords(self,x,y):
        if not(x is None or y is None):
            return np.round(x / (self.__canvas_shape[0] / 2) - 1 , 3) , np.round(-(y / (self.__canvas_shape[1] / 2) - 1) , 3)
        else:
            return x , y


    def __create_circle(self, canvas, x, y, r, **kwargs):
        return canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def __deactivate_cmd_buttons(self):
        for k in self.__cmd_buttons.keys():
            self.__cmd_buttons[k].config(state=tk.DISABLED)
            
    def __check_in_queue(self):
        self.__read_from_ard()
        self.__after_ID_in = self.__root.after(1, self.__check_in_queue)

    def __clear_text(self):
        self.__scrTxt.delete('1.0', 'end')

    def __send_to_arduino(self,value):
        self.__write_queue.put(value)
    
    def __send_to_projector(self,value):
        #print("Sending to projector:")
        #print(value)
        self.__proj_queue.put(value)
        
    def __read_from_ard(self):
        try:
            line = self.__read_queue.get_nowait()
            if line[0] == 'Q':
                # Emergency shutdown initiated by memory checker (RAM full)
                self.__exit()

            self.__scrTxt.insert('1.0',line)
        except Empty:
            pass
     

    def __empty_queue(self):
        while not self.__read_queue.empty():
            self.__read_from_ard()

    def __exit(self):

        self.__send_to_arduino('Q')

        time.sleep(2)
        self.__empty_queue()

        _ = messagebox.showinfo("OK","Press OK to exit")

        # Stop checking in_queue
        self.__root.after_cancel(self.__after_ID_in)
        self.__is_stopped = True
        
        
        self.__root.withdraw()

        self.__root.quit()


    # CODE FOR MOUSE ID POPUP QUERY

    def __initial_mouse_ID(self):
        self.__pre_root = tk.Tk()
        self.__pre_root.title("Set Mouse ID")
        self.__pre_root.geometry('250x100')
        self.__pre_root.focus()
        self.__center_window(self.__pre_root)
        self.__font = lambda size=12 : tk.font.Font(family='Helvetica',size=size)

        self.__mouse_var = tk.StringVar(self.__pre_root)
        self.__lbl_mouse = tk.Label(self.__pre_root,text="Choose ID of mouse:",font=self.__font())
        self.__lbl_mouse.grid(row=0,column=0)

        mouse_dict = self.__dir_tracker.get_mouse_dict()

        
        self.__mouse_options = [k for k in mouse_dict]
        

        try:
            self.__mouse_var.set(self.__mouse_options[0])
        except IndexError:
            self.__mouse_var.set("None")
        
        self.__mouse_opts = tk.OptionMenu(self.__pre_root, self.__mouse_var, *self.__mouse_options if self.__mouse_options else ["None"])
        self.__mouse_opts.config(width=10, textvariable=self.__mouse_var, font=self.__font())
        self.__mouse_opts.grid(row=1,column=0)

        new_mouse_btn = tk.Button(master=self.__pre_root, text="Add ID", command=lambda:self.__add_mouse_popup(self.__pre_root))
        new_mouse_btn.grid(row=1,column=1)

        ok_btn = tk.Button(master=self.__pre_root, text="OK", command=lambda:self.__check_value(self.__mouse_var.get(),self.__pre_root))
        ok_btn.grid(row=2,column=0)

        self.__pre_root.mainloop()


    def __setup_mouse_ID_query(self):
        self.__mouse_top = tk.Toplevel(self.__root)
        self.__mouse_top.title("New Mouse ID")
        self.__mouse_top.geometry('250x100')
        self.__center_window(self.__mouse_top)
        self.__mouse_top.focus()
        self.__font = lambda size=12 : tk.font.Font(family='Helvetica',size=size)

        self.__mouse_var = tk.StringVar(self.__mouse_top)
        self.__lbl_mouse = tk.Label(self.__mouse_top,text="Choose ID of next mouse:",font=self.__font())
        self.__lbl_mouse.grid(row=0,column=0)

        self.__mouse_options = [k for k in self.__dir_tracker.get_mouse_dict()]

        try:
            self.__mouse_var.set(self.__mouse_options[0])
        except IndexError:
            self.__mouse_var.set("None")
            pass
        
        self.__mouse_opts = tk.OptionMenu(self.__mouse_top, self.__mouse_var, *self.__mouse_options)
        self.__mouse_opts.config(width=10, textvariable=self.__mouse_var, font=self.__font())
        self.__mouse_opts.grid(row=1,column=0)

        new_mouse_btn = tk.Button(master=self.__mouse_top, text="Add ID", command=lambda: self.__add_mouse_popup(self.__mouse_top))
        new_mouse_btn.grid(row=1,column=1)

        ok_btn = tk.Button(master=self.__mouse_top, text="OK", command=lambda:self.__check_value(self.__mouse_var.get(),self.__mouse_top))
        ok_btn.grid(row=2,column=0)

        self.__mouse_top.mainloop()

    def __add_mouse_popup(self,root):
        self.__popup = tk.Toplevel(root)
        self.__popup.focus()
        self.__center_window(self.__popup)
        tk.Label(self.__popup, text="Please enter new mouse ID:",font=self.__font()).grid(row=0,column=0)

        self.__new_mouse_var = tk.StringVar()
        self.__mouse_textbox = tk.Entry(self.__popup, textvariable=self.__new_mouse_var, width=4,font=self.__font())
        self.__mouse_textbox.grid(row=0,column=1)

        add_btn = tk.Button(master=self.__popup, text="Add", command=lambda: self.__check_value(self.__mouse_textbox.get(),self.__popup))
        add_btn.grid(row=0,column=2)

        self.__popup.mainloop()

    def __check_value(self,val,root=None):
        # Replace any eventual spaces with underscores
        new_val = '_'.join(val.split(' '))

        if new_val not in self.__mouse_options:
            self.__mouse_options.append(new_val)
            self.__mouse_opts["menu"].add_command(label=new_val, command=lambda value=new_val: self.__mouse_var.set(new_val))
            try:
                self.__mouse_opts["menu"].delete("None")
            except TclError:
                pass


        self.__mouse_var.set(new_val)           

        if not(root == None):
            root.quit()
            root.destroy()


        
    def __center_window(self,toplevel):
        toplevel.withdraw()
        toplevel.update_idletasks()
        screen_width = toplevel.winfo_screenwidth()
        screen_height = toplevel.winfo_screenheight()

        x = int(screen_width/2 - toplevel.winfo_reqwidth()/2)
        y = int(screen_height/2 - toplevel.winfo_reqheight()/2)

        toplevel.geometry(F"+{x}+{y}")
        toplevel.deiconify()


if __name__ == '__main__':

    # Set up main_event to signal to comm_thread when to stop
    main_event = Event()
    restart_event = Event()
    user_ready = Event()
    # Set up data lock to prevent threads from accessing queue at the same moment
    data_lock = Lock()

    # Set up queue to pass data from user input to Arduino
    write_queue = Queue()
    read_queue = Queue()
    proj_queue = mp.Queue()

    # Only change these if you know what you are doing. 
    # Changing csv might mess things up since we are using the csv library.
    # If fourcc or video_format is changed, the other must be changed accordingly
    # (https://www.fourcc.org/codecs.php)
    program_constants = {   'fourcc':'123',
                            'video_format':'avi',
                            'rec_string':'recording',
                            'timestamp_format':'csv',
                            'timestamp_string':'timestamps',
                            'channels':1,
                            'cam_fps':550.,
                            'vid_fps':65.}

    # Ask user which computer they are on and adjust directories accordingly
    #dirs = dirParser()
    #scriptName = <Parser(dirs['scripts'])
    
    # Get user input for various parameter settings and choices
    optHandler = OptionHandler(program_constants)
    
    # Ask which script to run
    dirTracker = DirTracker(optHandler.get_dir_dict())

    # User interface
    user_UI = TkUICommunicator(read_queue, write_queue, proj_queue,restart_event, user_ready, optHandler.get_opt_dict(), dirTracker)

    
    user_UI.start()
    logging.debug("Waitin 2 sec")
    user_UI.stop()
    ans = user_UI.ask_for_start()
    
    if ans:
        user_UI.start()
    
    d = proj_queue.get_nowait()
    print(d)