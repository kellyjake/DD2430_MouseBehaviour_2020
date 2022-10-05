#include "LabCueLed.h"

#include "Arduino.h"

/*
    LabCueLed.cpp - Library for controlling Cue Leds in Laboratory setting
    
    Parameters:
    led_pin = Pin on Arduino board
    led_id  = Name for LED
    inMsg   = Pointer to a LabMessenger class object. This is how
                the LabCueLed class object sends information to the OS.

    LED states:
    0 = Off
    1 = On
    2 = Blinking

    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


LabCueLed::LabCueLed(){

}

LabCueLed::LabCueLed(int led_pin, const char led_id[], LabMessenger *inMsg){ 
      _pin = led_pin;
      _ledID = led_id;
      
      pinMode(_pin, OUTPUT);
      
      _ledMsg = inMsg;
      _is_on = 0;

      _blink_on_time = 10;
      _blink_off_time = 990;
      
      // Begin with LEDs turned off
      LabCueLed::configure(0,1);
}

void LabCueLed::update(){ 
    _t_cycle = _cycle_clock.duration();

    if(!_is_on && (_t_cycle >= _t_off)){
    _is_on = 1;
    if(_t_on > 0){
        digitalWrite(_pin, HIGH);

    } 
    _cycle_clock.reset();
    }else if(_is_on && _t_cycle >= _t_on){
    _is_on = 0;
    if(_t_off>0){
        digitalWrite(_pin, LOW);
    } 
    _cycle_clock.reset();
    }
}

void LabCueLed::configure(int time_on, int time_off){
    _t_on = time_on;
    _t_off = time_off;
    _cycle_clock.reset();
}

void LabCueLed::turn_on(){
    LabCueLed::configure(1,0);
    _ledMsg->send_data_state(_ledID,1);
}

void LabCueLed::turn_off(){
    LabCueLed::configure(0,1);
    _ledMsg->send_data_state(_ledID,0);
}

void LabCueLed::blink(){
    /*
    Does not send data of every time the LED blinks, but only when
    it starts and stops to blink.
    */

    LabCueLed::configure(_blink_on_time,_blink_off_time);
    _ledMsg->send_data_state(_ledID,2);
}