#include "LabPokebox.h"

#include "Arduino.h"

/*
    LabPokebox.cpp - Library for controlling Box with IR LED and Photoresistor in Laboratory setting.
    This is the box the mouse pokes its nose into to complete stages of the trial. 
    Depending on the strength of the photoresistor you might need to use digital or
    analog read. If analog read does not work, try digital.
    
    For the current setup we are using analog read in the office and digital in the lab.
    
    Parameters:
    nosepoke_pin    = Arduino pin
    nosepoke_id     = Name of pokebox
    read_digital    = Read photoresistor digitally?
    inMsg           = Pointer to a LabMessenger class object. This is how
                    the LabPokebox class object sends information to the OS.

    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


LabPokebox::LabPokebox()
{

}


LabPokebox::LabPokebox(const int nosepoke_pin, const char nosepoke_id[], bool read_digital, LabMessenger *inMsg){
    _pin = nosepoke_pin;
    _pokeID = nosepoke_id;

    _read_digital = read_digital;
    pinMode(_pin, INPUT);
    _pokeMsg = inMsg;

    _cooldown_time = 100;
    _cooldown_timer = LabTimer();
}


bool LabPokebox::_do_check(){
    return (_cooldown_timer.duration() > _cooldown_time);
}

bool LabPokebox::is_broken(){
    if(_read_digital){
        _isBroken = (digitalRead(_pin) == LOW);
    } else{
        _isBroken = (analogRead(_pin) == LOW);
    }

    if(_isBroken & _do_check()){
        _pokeMsg->send_data_state(_pokeID, 1);
        _cooldown_timer.reset();
        return 1;
    }

    return 0;
}
