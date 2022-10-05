#!/home/titan/anaconda3/env/venv1 python

import serial
import  os , sys , time , datetime , csv
from pathlib import Path
import faulthandler

class ArduinoHandler:
    """
    Class for communicating with the Arduino Board. Initializes Serial port and establishes communication protocol with the board.
    Also responsible for displaying and saving data received from the Arduino.
    Holds a pointer to a dir_tracker which is shared between threads and processes. This keeps track of the current directories where to save various files.

    Disclaimer: Does not do any extensive error handling.
    """

    def __init__(self, dir_tracker, option_handler, comm_protocol=None, baudrate=115200, timeout=.1):
        faulthandler.enable()

        self.__line = None
        
        # Prints only when verbose is true (debug option)
        self.__vprint = option_handler.get_vprint()        
        
        # These values can be changed, but be careful when doing so.
        # Don't use delim, SOL or EOL chars in sent messages in the Arduino script.
        self.__comm_protocol = {'dataKey':'D', 
                                'msgKey':'M', 
                                'syncKey':'T', 
                                'delim':",", 
                                'SOL':'{', 
                                'EOL':'}'} if comm_protocol is None else comm_protocol

        self.__port = option_handler.get_dict_value('port')
        self.__baudrate = baudrate
        self.__timeout = timeout

        self.__dir_tracker = dir_tracker
        self.__option_dict = option_handler.get_opt_dict()
        self.__const_dict = option_handler.get_camera_specs()

        # These values (0 - -3) are taken from the hardcoded definitions 
        # in LabTrial.cpp. If changed there, they should be here too.
        self.__transl_dict = { 'TRIAL' : {
                                            '0':"STARTED", 
                                            '1':"SUCCESSFUL", 
                                            '-1':"FAILED", 
                                            '-2':"ABORTED"},
                        
                                'PENALTY' : {
                                            '-3':"SET ITI TO",
                                            '1':"TOGGLED PENALTY"},

                                'STATS' : {
                                            '0':"STARTED", 
                                            '1':"SUCCESSFUL", 
                                            '-1':"FAILED", 
                                            '-2':"ABORTED",
                                            '-3':"PENALTIES"} }

        self.__EOL = self.__comm_protocol['EOL']
        self.__SOL = self.__comm_protocol['SOL']

        # It is not a stric requirement that data can be split into a list of length 5. 
        # Data will be displayed and saved anyway, but formatting will be weird.
        self.__msgLen = 5

        self.__doConnect = option_handler.get_dict_value('doConnect')
        
        if self.__doConnect:
            self.read_process_input = self.__read_process_input_connected
            #self.__init_comm()
        else:
            self.read_process_input = lambda : None
        


    ############## Public methods #################

    def start(self):
        if self.__doConnect:
            print("ArduinoHandler: init_comm")
            self.__init_comm()
        
        print("ArduinoHandler: init_write")
        self.__init_writer()
        print("ArduinoHandler: send_start_signal")
        self.__send_start_signal()


    def restart(self):
        self.__init_writer()
        self.__send_start_signal()


    def quit_routine(self, in_queue, restart = False):
        """
        Empties buffer and closes files. If program is exiting then also closes serial port.
        # TODO: FIX THIS TOMORROW. NOW EMPTY_BUFFER() TAKES QUEUE AS ARG. BUT I WANT TO BE ABLE
        # TO CALL SOMETHING FROM OUTSIDE WITH QUEUE AS ARG WHEN RESTARTING OR QUITTING TO EMPTY BUFFER.
        """
        time.sleep(2)

        self.__empty_buffer(in_queue)

        self.__vprint(F"-- Closing savefile {self.__csvPath} --\n")
        self.__csvFile.close()

        self.__vprint(F"-- Closing savefile {self.__syncPath} --\n")        
        self.__syncFile.close()

        if not(restart):
            self.__vprint("-- Closing Serial port --\n")
            try:
                self.__dev.close()
            except AttributeError:
                pass

    def pass_message(self,cmd):
        """
        Only passes valid commands to Arduino.
        cmd must be a single character.
        """
        if (cmd in self.__option_dict.keys()):
            self.__send(cmd)
        else:
            self.__vprint(F"Command '{cmd}' not in option keys!")

    def get_vprint(self):
        return self.__vprint
        

    ################ Internal methods ##################

    def __send_start_signal(self):
        self.__send('1')

    def __init_comm(self):
        self.__vprint("Opening serial port")
        self.__dev = serial.Serial(port=self.__port, baudrate=self.__baudrate,timeout=self.__timeout)
   
        # Need to wait for Serial port to initialize
        time.sleep(2)
        
        self.__vprint("\n-- Serial Port established --")
        self.__vprint(F"Port: {self.__port}")
        self.__vprint(F"Baudrate: {self.__baudrate}")
        self.__vprint(F"Timeout: {self.__timeout}")
        self.__vprint("-----------------------------\n")
        self.__vprint(F"Sending comm keys")
        
        # Clear any previous information that might remain in buffer
        self.__empty_buffer_hard()

        # Sending communication protocol keys
        for k,v in self.__comm_protocol.items():
            self.__send(v)
            time.sleep(.1)
            ardKey = self.__dev.read().decode('ascii')
            assert ardKey == v , self.__vprint(F"-- Problem with decoding messages from Arduino --\n-- Please make sure ASCII encoding is used --\n{k} received from Arduino: {ardKey}\nExpected {k}: {v}")



    def __init_writer(self):
        """ 
        Get various savepaths and open files for writing. All required information need to be accessible in the DirectoryTracker (initilized in main).
        """

        self.__parentDir = self.__dir_tracker.get_dirs('savedir')

        self.__csvPath = self.__dir_tracker.get_current_savepath(append_str='data', save_format='csv')
        self.__syncPath = self.__dir_tracker.get_current_savepath(append_str='timestamps', save_format='csv')
        
        # Since the recording file is created and written to in a new process, that process has a copy of the directory tracker,
        # so when it calls to get the current savepath it is not added to the dictionary (which is needed for adding timestamps),
        # so we have to do it here as well, but the path is not used here.
        _ = self.__dir_tracker.get_current_savepath(append_str=self.__const_dict['rec_string'], save_format=self.__const_dict['video_format'])
        
        # Datafile
        self.__csvFile = open(self.__csvPath,'w',encoding='utf-8')
        # Timestampfile
        self.__syncFile = open(self.__syncPath,'w',encoding='utf-8')

        self.__csvWriter = csv.writer(self.__csvFile)
        self.__csvWriter.writerows([["Time (ms)", "ID", "State", "Value"]])

        self.__syncWriter = csv.writer(self.__syncFile)
        self.__syncWriter.writerows([["Freq", self.__const_dict['freq']]])
        
        # Flag for if receiving sync start or end time
        self.__start_time_received = False

        self.__vprint(F"-- Opening savefile {self.__csvPath} --\n")
        self.__vprint(F"-- Opening savefile {self.__syncPath} --\n")


    def __send(self, message):
        if self.__doConnect: 
            self.__dev.write(str(message).encode('ascii'))
            self.__vprint(F"Sent: {str(message).encode('ascii')}")

    def __record_sync(self,data):
        """
        Not used
        """
        try:
            t = data[1]
            t_string = "t_end" if self.__start_time_received else "t_start"
            self.__syncWriter.writerow([t_string,t])
        except IndexError:
            self.__vprint("Something wrong with sync data!")
            self.__vprint(F"Got: {data}")
    
    def __save_data(self,data):        
        try:
            _ , timeStamp , ID , state , value = data
        except (ValueError , TypeError) as e:
            self.__vprint("Didn't get enough information to save data properly:")
            self.__vprint(data)
            self.__vprint(e)
            return
        
        writeList = [timeStamp,ID,state,value]

        # According to communication protocol there shouldn't be messages with more than 4 entries,
        # but if there are, these will be caught and saved in a new column in the csv file.
        if len(data) > self.__msgLen:
            self.__vprint(F"Expected message of length {self.__msgLen}, got {len(data)}.")
            self.__vprint("Writing extra messages in new column.")
            for i in range(self.__msgLen,len(data)):
                val = data[i]
                writeList.append(val)
        
        self.__csvWriter.writerow(writeList)


    def __show_message(self,splitLine):
        """
        These values may need to be changed if changes occur in LabTrial.h macro definitions
        """
        try:
            _ , ID , idx , state , val = splitLine

            if ID == 'ITI':
                msg = F"\n[TRIAL {idx}] SET INTERTRIAL TIME TO: \t {val} MS."
            elif ID == 'TRIAL':
                msg = F"\n[TRIAL {idx}] {self.__transl_dict[ID][state]}: \t {val} MS."
            elif ID == 'STATS':
                msg = F"\n[MOUSE {self.__dir_tracker.get_curr_mouse_ID()}]\t{self.__transl_dict[ID][state]} {val}."
            elif ID == 'PENALTY':
                msg = F"\n[PENALTY {idx}] {self.__transl_dict[ID][state]}: {val}."
            else:
                msg = splitLine
        except (ValueError , KeyError) as e:
            self.__vprint("Something is wrong with the formatting")
            self.__vprint(e)
            self.__vprint(splitLine)
            msg = splitLine

        return msg

    def __empty_buffer(self, read_queue):
        """
        Empties buffer and saves/prints remaining messages.
        """
        if self.__doConnect:
            self.__vprint("-- Emptying buffer --\n")
            while(self.__dev.in_waiting):
                msg = self.read_process_input()
                if msg: 
                    read_queue.put(msg)
        
        self.__vprint("Last message read after emptying buffer:")
        self.__vprint(self.__line)

    def __empty_buffer_hard(self):
        """
        Empties buffer and discards information.
        """
        if self.__doConnect:
            self.__vprint("-- Emptying buffer hard --")
            while(self.__dev.in_waiting):
                self.__dev.read()


    def __read_process_input_connected(self):
        """
        Used when connected to Arduino.
        """
        if self.__dev.in_waiting:
            self.__curr_char = self.__dev.read().decode('ascii')

            if self.__curr_char == self.__SOL:
                self.__line = ''

            elif self.__curr_char == self.__EOL:
                msg = self.__proc_input(self.__line)
                return msg
            else:
                self.__line += self.__curr_char

    def __proc_input(self,line):
        """
        Handles input from Arduino depending on the provided key in the beginning of the message.
        Messages are printed and data/sync is saved to file.
        line must be a string, prefferably with proper delimiter and of correct length.
        """
        #self.__vprint(F"Received: {line}")

        splitLine = line.split(self.__comm_protocol['delim'])

        key = splitLine[0]

        if key == self.__comm_protocol['msgKey']:
            msg = self.__show_message(splitLine)
            return msg

        elif key == self.__comm_protocol['dataKey']:
            self.__save_data(splitLine)

        elif key == self.__comm_protocol['syncKey']:
            self.__record_sync(splitLine)
            self.__start_time_received = not(self.__start_time_received)

        else:
            self.__vprint(F"Unrecognized message key: {key}")
            self.__vprint(F"Expected {self.__comm_protocol['msgKey']}, {self.__comm_protocol['dataKey']} or {self.__comm_protocol['syncKey']}")

