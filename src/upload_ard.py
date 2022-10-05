# Simple program to compile and upload Arduino code using the Arduino command line

import os , time , datetime , sys , platform , logging , subprocess

def uploadIno(dir_tracker, option_handler):

    """ 
    Function that uploads given arduino script (fullFilePath) to Arduino board.
    On windows there is a _debug executable which does not start the Arduino IDE.
    It also seems that when not running the _debug.exe, if there is a compilation error
    then the script is not uploaded, but the old script that exists on the board is run
    (without any error message).
    
    arduinoFolder : absolute path to dir where arduino executable exists 
    boardtype should be arduino:avr:mega, arduino:avr:uno or other valid board type (see Arduino IDE for options)
    port is COM** on Windows and /dev/ttyACM* on Linux. However we use a script to get all available ports (src/listPorts.py).

    debug produces a buttload of debug text, but it is still the preffered option.
    """
    
    fullFilePath = dir_tracker.get_dirs()['full_script_path']
    arduinoFolder = dir_tracker.get_dirs()['ard_exec']
    boardType = option_handler.get_dict_value('boardtype')
    port = option_handler.get_dict_value('port')

    logging.debug(F"-- Uploading {fullFilePath} to Arduino Board, please wait... --")

    arduinoStr = "arduino"
    

    thisos = platform.system()
    if thisos == 'Windows':
        arduinoStr += "_debug"

    arduinoCommand = arduinoStr + " --" + "upload" + " --board " + boardType + " --port " + port + " " + fullFilePath

    logging.debug("\n\n-- Arduino Command --")
    logging.debug(arduinoCommand)

    logging.debug(F"\n-- Starting upload of {fullFilePath}.ino --")

    os.chdir(arduinoFolder)
    presult = subprocess.call(arduinoCommand, shell=True)

    if presult:
        logging.debug(F"\n Upload Failed - Result Code = {presult} --")
        logging.debug("Script could not be uploaded. Please see error messages.\nThis usually appears due to syntax error in the .ino script, incorrect port specified, or because the port is already opened by some other script or the Arduino IDE.")
        raise Exception
    else:
        logging.debug("\n-- Upload Successful -- \n")    