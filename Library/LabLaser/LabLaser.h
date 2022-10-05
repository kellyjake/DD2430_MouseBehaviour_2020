#pragma once

/*
    LabLaser.h - Library for controlling Laser in Laboratory setting with Arduino Board
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


#include "Arduino.h"
#include "LabMessenger.h"



class LabLaser
{
    public:
        LabLaser();
        LabLaser(const int laser_pin, const char laser_ID[], const int max_cycles, const int time_on, const int time_off, const int time_delay, const int auto_min_wait, const int auto_max_wait, LabMessenger *inMsg);
        
        void update();
        void start();
        void auto_run();
        void auto_stop();
        void couple(LabLaser *coupled_laser);
        void decouple();
        
    private:
        void auto_update();
        
        bool _is_active;
        bool _should_start;
        
        int _t_on;
        int _t_off;
        bool _is_on;

        int _max_cycles_count;
        int _cycles_count;
        
        LabTimer _cycle_clock;
        LabTimer _delay_clock;
        LabTimer _auto_clock;
        
        int _t_delay;
        bool _wait;

        bool _do_auto_run;
        int _auto_wait_time;
        int _auto_min_wait;
        int _auto_max_wait;

        int _pin;
        LabMessenger *_lasMsg;
        
        bool _coupled;
        LabLaser *_coupled_laser;

        const char *_laser_ID;
};
