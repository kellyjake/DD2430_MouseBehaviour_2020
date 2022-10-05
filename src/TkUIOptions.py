from listPorts import serial_ports
import tkinter as tk
import tkinter.font
from tkinter import messagebox , filedialog
from platform import system
import os 

class TkUIOptions:
    def __init__(self):
        self.__root = tk.Tk()
        self.__root.title("Program options")
        self.__root.geometry('250x300')
        self.__center_window(self.__root)
        self.__root.focus()
        self.__font = lambda size=12 : tk.font.Font(family='Helvetica',size=size)
        
        self.__choice_dict = {}
        self.__spec_dict = {}
        self.__dir_dict = {}

        # Variables for option of choosing which computer the user is operating from        
        # The option is currently disabled because we only use the lab computer.
        # If another is used, uncomment self.__setup_comp_query() and option will be shown.
        comp_options = {"Windows":[1,"Office"], "Linux":[2,"Lab"]}
        
        self.__OS = system()

        self.__comp_var = tk.IntVar()
        self.__comp_var.set(comp_options[self.__OS][0])

        #self.__setup_comp_query()
        self.__setup_port_query()

        # Variables for option of choosing type of Arduino board to use.
        self.__board_options = ["arduino:avr:mega","arduino:avr:uno"]
        self.__board_variable = tk.StringVar(self.__root)
        self.__board_variable.set(self.__board_options[0])

        #Currently we only use the Mega2560 board, so no need for unneccessary choices. 
        # Uncomment to allow for other types of boards.
        #self.__setup_board_query()

        # Shows some additional options, mainly for debugging purposes
        self.__setup_debug_menu()

        # Show sliders for gain, gamma and exposure time
        self.__setup_camera_specs()

        # Online tracking checkbox
        self.__setup_online_query()

        self.__setup_save_quit_button()
        
        self.__root.mainloop()
        
    ############ PUBLIC METHODS #################

    def get_choice_dict(self):
        return self.__choice_dict

    def get_spec_dict(self):
        return self.__spec_dict

    def get_dir_dict(self):
        return self.__dir_dict

    ############ INTERNAL METHODS ################

    def __set_directory(self):
        if self.__comp_var.get() == 1:
            # Lab computer
            self.__dir_dict['script_dir'] = r"C:\Users\user\Documents\Master_Program_Magnus\Ard_Scripts"
            self.__dir_dict['ard_exec'] = r"C:\Program Files (x86)\Arduino"
            self.__dir_dict['savedir'] = r"E:\ExperimentalResults"
        elif self.__comp_var.get() == 2:
            # Office computer
            self.__dir_dict['script_dir'] = "/home/titan/KI2020/Code/Ard_Scripts"
            self.__dir_dict['ard_exec'] = "/snap/arduino/41"
            self.__dir_dict['savedir'] = "/home/titan/KI2020/ExperimentalResults"

    def __setup_port_query(self):
        self.__port_var = tk.StringVar()
        self.__lbl_port = tk.Label(master=self.__root, textvariable=self.__port_var,font=self.__font())
        self.__port_var.set("Choose serial port: ")
        self.__lbl_port.pack()

        self.__port_options = sorted(serial_ports())

        self.__port_variable = tk.StringVar(self.__root)
        try:
            self.__port_variable.set(self.__port_options[0])
        except IndexError:
            messagebox.showerror("Serial port error","No serial ports available!\n\nPlease make sure there is no serial monitor open (in the Arduino IDE) or make sure the Arduino is plugged in!")
            raise IndexError
        
        opts = self.__port_options if len(self.__port_options) > 1 else tuple(self.__port_options)

        opt = tk.OptionMenu(self.__root, self.__port_variable, *opts)
        opt.config(width=90, font=self.__font())
        opt.pack()

    def __setup_online_query(self):
        self.__online_var = tk.BooleanVar(self.__root)
        btn = tk.Checkbutton(self.__root, text='Online tracking',variable=self.__online_var,onvalue=1,offvalue=0)
        btn.pack()

    def __setup_debug_menu(self):
        self.__menubar = tk.Menu(self.__root)
        self.__debugmenu = tk.Menu(self.__menubar, tearoff=0)

        self.__menubar.add_cascade(label="Debug options", menu=self.__debugmenu)

        self.__verbose_var = tk.BooleanVar(master=self.__debugmenu,value=True)
        self.__upload_var = tk.BooleanVar(master=self.__debugmenu,value=True)
        self.__connect_var = tk.BooleanVar(master=self.__debugmenu,value=True)
        self.__blackfly_var = tk.BooleanVar(master=self.__debugmenu, value=True)
    
        self.__debugmenu.add_checkbutton(label="Verbose mode", onvalue=True, offvalue=False, variable = self.__verbose_var)
        self.__debugmenu.add_checkbutton(label="Upload script", onvalue=True, offvalue=False, variable = self.__upload_var)
        self.__debugmenu.add_checkbutton(label="Connect to Arduino", onvalue=True, offvalue=False, variable = self.__connect_var)
        self.__debugmenu.add_checkbutton(label="Use BlackFly", onvalue=True, offvalue=False, variable = self.__blackfly_var)

        self.__root.config(menu=self.__menubar)

    def __setup_camera_specs(self):
        self.__setup_gamma()
        self.__setup_expt()
        self.__setup_gain()


    def __setup_gamma(self):
        self.__gamma_tmp_var = tk.DoubleVar(value=0.25)
        
        self.__gamma_lab = tk.Label(master=self.__root, text="Gamma:",font=self.__font())
        self.__gamma_lab.pack()

        self.__gamma_double_var = tk.DoubleVar(master=self.__root)
        self.__gamma_slider = tk.Scale(self.__root, from_=0., to_=1., variable=self.__gamma_double_var, resolution=.01, showvalue=0, orient='horizontal')
        self.__gamma_slider.set(self.__gamma_tmp_var.get())
        self.__gamma_slider.pack()

        self.__gamma_textbox = tk.Entry(self.__root, textvariable=self.__gamma_double_var,width=4,font=self.__font())
        self.__gamma_textbox.bind("<FocusOut>", lambda event: self.__validate(self.__gamma_textbox, self.__gamma_double_var, self.__gamma_tmp_var, self.__gamma_slider, float, 0, 1))
        self.__gamma_textbox.pack()
        

    def __setup_expt(self):
        self.__expt_tmp_var = tk.IntVar(value=1500)

        self.__expt_lab = tk.Label(master=self.__root, text="Exposure time:", font=self.__font())
        self.__expt_lab.pack()

        self.__expt_int_var = tk.IntVar(master=self.__root)
        self.__expt_slider = tk.Scale(self.__root, from_=1, to_=15000, variable=self.__expt_int_var, resolution=1, showvalue=0, orient='horizontal')
        self.__expt_slider.set(self.__expt_tmp_var.get())
        self.__expt_slider.pack()

        self.__expt_textbox = tk.Entry(self.__root, textvariable=self.__expt_int_var,width=6,font=self.__font())
        self.__expt_textbox.bind("<FocusOut>", lambda event: self.__validate(self.__expt_textbox, self.__expt_int_var, self.__expt_tmp_var, self.__expt_slider, int, 1, 100000))
        self.__expt_textbox.pack()


    def __setup_gain(self):
        self.__gain_tmp_var = tk.DoubleVar(value=0.0)

        self.__gain_lab = tk.Label(master=self.__root, text="Gain:",font=self.__font())
        self.__gain_lab.pack()

        self.__gain_double_var = tk.DoubleVar(master=self.__root)
        self.__gain_slider = tk.Scale(self.__root, from_=0, to_=10, variable=self.__gain_double_var, resolution=.1, showvalue=0, orient='horizontal')
        self.__gain_slider.set(self.__gain_tmp_var.get())
        self.__gain_slider.pack()

        self.__gain_textbox = tk.Entry(self.__root, textvariable=self.__gain_double_var,width=4,font=self.__font())
        self.__gain_textbox.bind("<FocusOut>", lambda event: self.__validate(self.__gain_textbox, self.__gain_double_var, self.__gain_tmp_var, self.__gain_slider, float, 0, 10))
        self.__gain_textbox.pack()

    def __setup_save_quit_button(self):
        qbtn = tk.Button(master=self.__root, text='Save and continue', command=self.__save_quit, font=self.__font())
        qbtn.pack()

    def __center_window(self,toplevel):
        toplevel.withdraw()
        toplevel.update_idletasks()
        screen_width = toplevel.winfo_screenwidth()
        screen_height = toplevel.winfo_screenheight()

        x = int(screen_width/2 - toplevel.winfo_reqwidth()/2)
        y = int(screen_height/2 - toplevel.winfo_reqheight()/2)

        toplevel.geometry(F"+{x}+{y}")
        toplevel.deiconify()

    def __save_quit(self):
        self.__set_directory()
        
        ok = self.__query_script()

        if ok:
            self.__make_dicts()
            self.__root.destroy()


    def __validate(self, val, var, old_var, slider, des_type, min_val, max_val):
        new_val = val.get()

        try:
            des_type(new_val)
        except:
            var.set(old_var.get())
            slider.set(old_var.get())
            return

        if not(bool(new_val)):
            var.set(old_var.get())
            slider.set(old_var.get())
            return
        
        new_val = des_type(new_val)

        if (new_val >= min_val) & (new_val <= max_val):
            old_var.set(des_type(new_val))
            return
        
        var.set(old_var.get())
        slider.set(old_var.get())
        return

    def __make_dicts(self):
        self.__choice_dict['verbose'] = self.__verbose_var.get()
        self.__choice_dict['doUpload'] = self.__upload_var.get()
        self.__choice_dict['doConnect'] = self.__connect_var.get()
        self.__choice_dict['useBlackFly'] = self.__blackfly_var.get()
        self.__choice_dict['port'] = self.__port_variable.get()
        self.__choice_dict['boardtype'] = self.__board_variable.get()
        self.__choice_dict['online'] = self.__online_var.get()

        self.__spec_dict['gamma'] = self.__gamma_double_var.get()
        self.__spec_dict['exposure_time'] = self.__expt_int_var.get()
        self.__spec_dict['gain'] = self.__gain_double_var.get()


    def __query_script(self):
        filename = filedialog.askopenfilename(initialdir = self.__dir_dict['script_dir'], title = "Select script to run", filetypes = [("ino files","*.ino")])

        if filename:
            self.__dir_dict['full_script_path'] = filename.replace('/',os.path.sep)
            
            new_name = self.__dir_dict['full_script_path'].replace('.ino','')

            self.__dir_dict['script_name'] = new_name.split(os.path.sep)[-1]

            return True
        else:
            return False

    

if __name__ == '__main__':
    t = TkUIOptions()
    c_dict = t.get_choice_dict()
    d_dict = t.get_dir_dict()
    s_dict = t.get_spec_dict()

    for d in [c_dict, d_dict, s_dict]:
        for k,v in d.items():
            print(F"{k} - {v}")
