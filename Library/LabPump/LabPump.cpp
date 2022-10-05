#include "LabPump.h"

#include "Arduino.h"

/*
    LabPump.h - Library for controlling Pump in Laboratory setting
    
    Parameters:
    control_pin = Arduino pin
    pump_id     = Name of Pump
    duration    = Time in milliseconds to turn on pump
    inMsg       = Pointer to a LabMessenger class object. This is how
                the LabPump class object sends information to the OS.

    Pump states:
    0 = Off
    1 = On

    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


LabPump::LabPump(){

}

LabPump::LabPump(const int control_pin, const char pump_id[], const int duration, LabMessenger *inMsg) {
    pinMode(control_pin, OUTPUT);

    // HIGH = Off
    digitalWrite(control_pin, HIGH);
    
    _t_max = duration;
    _pin = control_pin;
    _pumpMsg = inMsg;
    _pumpID = pump_id;
}

void LabPump::set_duration(int duration){
    /*
    Not currently used but might be if implementing possibility to change parameters
    between runs.
    */
    _t_max = duration;  
}

void LabPump::update(){
    _dur = _cycle_clock.duration();
    if(_is_active != 1) return;
    
    if(_dur >= _t_max){
        stop();
    }
}

void LabPump::start(){
    digitalWrite(_pin, LOW);
    _pumpMsg->send_data_state_value(_pumpID,1,_t_max);
    _is_active = 1;
    _cycle_clock.reset();
}

void LabPump::stop(){
    digitalWrite(_pin, HIGH);
    _pumpMsg->send_data_state(_pumpID,0);
    _is_active = 0;
}

bool LabPump::is_pumping(){
    return _is_active;
}
