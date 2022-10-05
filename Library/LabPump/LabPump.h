#pragma once

/*
    LabPump.h - Library for controlling Pump in Laboratory setting with Arduino Board
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"

#include "LabMessenger.h"


class LabPump
{
    public:
        LabPump();
        LabPump(const int control_pin, const char pump_id[], const int duration, LabMessenger *inMsg);
        void set_duration(int duration);
        void update();
        void start();
        void stop();
        void hard_start();
        void hard_stop();
        bool is_pumping();

    private:
        LabTimer _cycle_clock;
        bool _is_active;
        int _t_max = 100;
        int _pin;
        double _dur;
        LabMessenger *_pumpMsg;
        const char *_pumpID;
};

