#pragma once

/*
    LabMessenger.h
    Custom Message class to send information to Python script according to the established protocol.
    Attaches a char before each message sent to the Serial port so that the receiving script knows
    if the sent message should be displayed for the user or only saved as data.
    Data is saved with timestamps and an identifier for the event, as well as a variable value.
    
    This is set in the header of each .ino script by the user and is described above (for now).
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"

#include "LabTimer.h"

#define _EOL() _serial->print(_EOL);
#define _SOL() _serial->print(_SOL);
#define _delim() _serial->print(_delim);
#define _data_key() _serial->print(_dataKey);
#define _msg_key() _serial->print(_msgKey);
#define _sync_key() _serial->print(_syncKey);

class LabMessenger
{
    public:


        LabMessenger();
        LabMessenger(Stream *ser_ref, volatile uint32_t *iterator);
        void wait_for_message();
        void send_data_state(const char id[], int state);
        void send_data_state_value(const char id[], int state, int val);
        void send_timestamp();
        void send_timestamp_ln();
        void reset_timestamp();
        void send_trial_info(const char ID[], int trial_no, int state, int val);


    private:
        void _init_comm_protocol();
        char _get_next_message();

        char _dataKey;
        char _msgKey;
        char _syncKey;
        char _delim;
        char _EOL;
        char _SOL;
        volatile uint32_t *_timestamp;
        Stream *_serial;
};
