#pragma once

/*
    LabTrial.h - Library for tracking Trials in experimental setting
    
    Created by Magnus Pierrau, 19/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"
#include "LabMessenger.h"
#include "LabSync.h"

// If these are changed then changes are required in src/arduino_class.py
#define _trial_ID "TRIAL"
#define _penalty_ID "PENALTY"
#define _ITI_ID "ITI"
#define _stats_ID "STATS"

class LabTrial
{
    public:
        LabTrial();
        LabTrial(int min_iti, int max_iti, int max_trial_time, int penalty_cooldown, LabSync *sync, LabMessenger *inMsg);
        
        void start();
        void abort();
        void reset();
        void quit();
        void send_stats();

        void success();
        void fail();

        bool timedout();
        bool iti_completed();
        void time_penalty();

        void stage_completed();
        int get_stage();

        void penalty_test();
        void toggle_penalty();


    private:
        void _reset_iti(int min, int max, bool do_print);
        void _penalty();
        void _dummy_penalty();

        void (LabTrial::*_penalty_func)();

        int _intertrial_time;
        int _min_iti;
        int _max_iti;
        int _max_trial_time;
        int _curr_trial_stage;
        int _penalty_cooldown;
        int _using_penalty;

        int _trial_count;
        int _success_count;
        int _fail_count;
        int _run_count;
        int _penalty_count;
        int _abort_count;

        LabTimer _trial_timer;
        LabTimer _penalty_cooldown_timer;

        LabMessenger *_trialMsg;
        LabSync *_sync;

};