#include "LabMessenger.h"

#include "Arduino.h"

/*
    LabMessenger.cpp
    Custom Message class to send information to python script according to the established protocol
    Attaches a char before each message sent to the Serial port so that the receiving script knows
    if the sent message should be displayed for the user or only saved as data.
    Data is saved with timestamps and an identifier for the event, as well as two variable values (ints)
    to be used as one wishes.

    Messages sent by methods using the _msg_key() will be displayed to the user but not saved.
    Messages sent by methods using the _data_key() will not be displayed to the user but will 
    be saved.
    Messages sent by methods using the _sync_key() will not be displayed to the user but will
    be saved.

    Since it is a bit messy to print complex strings with various values in them in C++ we only 
    send "coded" information out via send_trial_info().

    Parameters:
    ser_ref     = A pointer to the Serial port opened in the setup(). (Passed by reference).
    iterator    = Counter goverened by the System Interrupt Routine. 
                    This is the clock for the entire experiment (except other internal timers).
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


LabMessenger::LabMessenger(){

        }

LabMessenger::LabMessenger(Stream *ser_ref, volatile uint32_t *iterator){
    _serial = ser_ref;

    LabMessenger::_init_comm_protocol();

    // Waits until the OS has emptied the serial buffer
    _serial->flush();
    
    _timestamp = iterator;
}

void LabMessenger::_init_comm_protocol(){
    /*
    Method to receive and return the communication protocol.
    This is done to make sure the sent data is received and handled
    properly.

    _delim(), _SOL() (Start Of Line) and _EOL() (End Of Line) are defined
    in the header file. This is only done for brevity.
    */

    _dataKey = LabMessenger::_get_next_message();
    _serial->print(_dataKey);

    _msgKey = LabMessenger::_get_next_message();
    _serial->print(_msgKey);

    _syncKey = LabMessenger::_get_next_message();
    _serial->print(_syncKey);

    _delim = LabMessenger::_get_next_message();
    _delim();

    _SOL = LabMessenger::_get_next_message();
    _SOL();

    _EOL = LabMessenger::_get_next_message();
    _EOL();
}

char LabMessenger::_get_next_message(){
    /*
    Wait until a char exists in the serial buffer and return it.
    */

    while(!_serial->available());
    return _serial->read();
}

void LabMessenger::wait_for_message(){
    /*
    Wait until a char exists in the serial buffer and read it but
    do nothing with it. We do this to wait until the OS is ready
    to continue, like a start signal.
    */

    while(!_serial->available());
    _serial->read();
}

void LabMessenger::send_data_state(const char id[], int state){
    /*
    Method to be used by the various classes that utilize the messenger.
    Used when we only wish to send information of time, object ID (e.g. 'Laser1')
    and its state, and no further variable information.

    This information will be saved in the *_data.csv file.
    */

    _SOL();
    _data_key();
    _delim();
    _serial->print(*_timestamp);
    _delim();
    _serial->print(id);
    _delim();
    _serial->print(state);
    _delim();
    _serial->print(0);
    _EOL();
}

void LabMessenger::send_data_state_value(const char id[], int state, int val){
    /*
    Method to be used by the various classes that utilize the messenger.
    Used when we wish to send information of time, object ID (e.g. 'Laser1'), 
    its state, and some further variable information, such as its on-time or
    an ITI etc.

    This information will be saved in the *_data.csv file.
    */

    _SOL();
    _data_key();
    _delim();
    _serial->print(*_timestamp);
    _delim();
    _serial->print(id);
    _delim();
    _serial->print(state);
    _delim();
    _serial->print(val);
    _EOL();
}

void LabMessenger::send_timestamp(){
    /*
    Method used by the LabSync object.

    This information will be saved in the *_timestamps.csv file.
    */

    _SOL();
    _sync_key();
    _delim();
    _serial->print(*_timestamp);
    _EOL();
}

void LabMessenger::send_timestamp_ln(){
      /*
    Method used by the LabSync object.

    This information will be saved in the *_timestamps.csv file.

    Used for debugging.
    */

    _SOL();
    _sync_key();
    _delim();
    _serial->print(*_timestamp);
    _EOL();
    _serial->print('\n');
}

void LabMessenger::reset_timestamp(){
    /*
    Resets the iterator globally.
    */
    *_timestamp = 0;
}

void LabMessenger::send_trial_info(const char ID[], int trial_no, int state, int val){
    /*
    Used by the LabTrial class. Can be used by other classes as well, but wont be 
    displayed properly to the user (they WILL be displayed though but as a list 
    of strings).  
    See src/arduino_class.py (proc_input and __show_message methods) for more details.

    Parameters:
    ID          = Name of object sending information. Should be "TRIAL", "ITI", 
                "STATS" or "PENALTY" to be displayed properly.
    trial_no    = Trial number
    state       = State of object. See LabTrial.cpp for more info.
    val         = Free choice of information to send (time, duration, value etc.)
                As long as you know what you are sending.
    */

    _SOL();
    _msg_key();
    _delim();
    _serial->print(ID);
    _delim();
    _serial->print(trial_no);
    _delim();
    _serial->print(state);
    _delim();
    _serial->print(val);
    _EOL();
}