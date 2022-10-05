#pragma once

/*
    LabTimer.h - Library for controlling internal Timers for various events
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"


class LabTimer
{
    public:
        LabTimer();
        unsigned long duration();
        void reset();

    private:
        unsigned long _timestamp;
};
