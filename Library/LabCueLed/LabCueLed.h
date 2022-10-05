#pragma once

/*
    LabCueLed.h - Library for controlling Cue Leds in Laboratory setting with Arduino Board
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"
#include "LabMessenger.h"



class LabCueLed
{
    public:
        LabCueLed();
        LabCueLed(int led_pin, const char led_id[], LabMessenger *inMsg);
        void update();
        void turn_on();
        void turn_off();
        void blink();

    private:
        void configure(int time_on, int time_off);
        bool _is_on;
        int _pin;
        int _t_on;
        int _t_off;
        int _blink_on_time;
        int _blink_off_time;
        double _t_cycle;
        bool _do_check;

        LabTimer _cycle_clock;
        LabMessenger *_ledMsg;

        const char *_ledID;
};

