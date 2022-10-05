#pragma once

/*
    LabSync.h - Library for syncing camera with arduino events. See .cpp file
    for more comments.
    
    Created by Magnus Pierrau, 22/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"
#include "LabMessenger.h"
#include <avr/io.h>
#include <avr/interrupt.h>

//Arduino clock speed
#define _freq_clk 16000000
//We try to use prescaler
#define _prescaler 8
//shortcut to Turn on square wave and interrupt ISR
#define _START_CAMERA_ACQUISITION() cli(); TIMSK1 |= (1 << OCIE1A); TCCR1A |= (1 << COM1A0); sei();
//shortcut to turn off square wave and interrupt ISR
#define _STOP_CAMERA_ACQUISITION() cli(); TIMSK1 &= (0 << OCIE1A);  TCCR1A &= (0 << COM1A0); sei();

class LabSync
{
    public:
        LabSync();
        LabSync(volatile uint32_t *iterator, int freq, LabMessenger *inMsg);
        void stop();
        void start();
        void _ISR_func();
        bool is_stopped();

        volatile uint32_t *_iterator;
        int _freq;

    private:
        LabMessenger *_syncMsg;
        bool _is_stopped;

};


