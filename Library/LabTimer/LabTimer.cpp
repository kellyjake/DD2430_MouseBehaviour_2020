#include "LabTimer.h"

#include "Arduino.h"

/*
    LabTimer.cpp - Library for controlling internal Timers for various events
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

LabTimer::LabTimer(){
    reset();
}

unsigned long LabTimer::duration(){
    return millis() - _timestamp;
}
      
void LabTimer::reset(){
    _timestamp = millis();
}