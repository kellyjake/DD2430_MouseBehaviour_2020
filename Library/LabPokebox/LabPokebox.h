#pragma once

/*
    LabPokebox.h - Library for controlling Box with IR LED and Photoresistor in Laboratory setting with Arduino Board.
    This is the box the mouse pokes its nose into to complete stages of the trial.
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"
#include "LabMessenger.h"


class LabPokebox
{
    public:
        LabPokebox();
        LabPokebox(const int nosepoke_pin, const char nosepoke_id[], bool read_digital, LabMessenger *inMsg);
        bool is_broken();

    private:
        int _pin;
        bool _isBroken;
        bool _read_digital;
        LabMessenger *_pokeMsg;
        const char *_pokeID;
        int _cooldown_time;
        LabTimer _cooldown_timer;

        bool _do_check();
};

