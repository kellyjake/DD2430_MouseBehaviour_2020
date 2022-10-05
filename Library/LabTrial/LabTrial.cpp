#include "LabTrial.h"

#include "Arduino.h"

/*
    LabTrial.cpp - Library for tracking Trial in experimental setting.
    This class does not decide what the rewards or penalties are, or what the
    requirements for success or failure etc. is, that is done in the main loop.
    This only helps keep track of what stage we are in in the trial etc.

    Created by Magnus Pierrau, 19/07/2020
    For use at Karolinska Institutet

    Parameters:
    min_iti             = Lower interval limit for random intertrial interval.
    max_iti             = Upper interval limit for random intertrial interval.
    max_trial_time      = Maximum allowed trial time before trial fails.
    penalty_cooldown    = Time between penalties being issued
    sync                = Pointer to LabSync object which keeps the experimental clock synced.
    inMsg               = Pointer to a LabMessenger class object. This is how
                        the Labtrial class object sends information to the OS.

    Trial stages:
    -1  = Paused / waiting to start
    0   = In ITI phase
    1   = Trial running at stage 1 (trial box stage)

    Trial stage may be higher if experiment has more stages.

    Trial states:
    0   = Trial started
    1   = Trial successful
    -1  = Trial failed
    -2  = Trial aborted
    -3  = Penalty issued

    If trial states are changed to new values, then arduino_class.py (__transl_dict) needs to be updated too.
*/

LabTrial::LabTrial(){
    
}

LabTrial::LabTrial(int min_iti, int max_iti, int max_trial_time, int penalty_cooldown, LabSync *sync, LabMessenger *inMsg){
    // If these are changed then they must be equally altered in arduino_class.py __show_message()

    _min_iti = min_iti;
    _max_iti = max_iti;
    _max_trial_time = max_trial_time;
    _intertrial_time = -1;
    _penalty_cooldown = penalty_cooldown;

    _trial_count = 0;
    _success_count = 0;
    _fail_count = 0;
    _penalty_count = 0;
    _abort_count = 0;
    _run_count = 1;
    
    _curr_trial_stage = -1;
    
    _trialMsg = inMsg;
    _sync = sync;
    
    _using_penalty = false;
    _penalty_func = &_dummy_penalty;
    

    _trial_timer = LabTimer();
    _penalty_cooldown_timer = LabTimer();
}

void LabTrial::start(){
    /* 
    Start a new trial
    If a current trial is running we abort it and start a new 
    */

    if(_curr_trial_stage > 0){
        abort();
    }
    _trial_count++;

    _curr_trial_stage = 1;

    _trial_timer.reset();
    _trialMsg->send_trial_info(_trial_ID, _trial_count, 0, _intertrial_time);

    if (_sync->is_stopped()){
        _sync->start();
    }
}

void LabTrial::abort(){
    /* 
    Aborts the current trial. Effectively pauses the experiment 
    */

    _curr_trial_stage = -1;
    _abort_count++;
    _reset_iti(-1,-1,false);
    
    _trialMsg->send_trial_info(_trial_ID, _trial_count, -2, _trial_timer.duration());
}

void LabTrial::reset(){
    /* 
    Starts a new run. 
    This is used when switching mice.
    */

    quit();

    _run_count++;
    _trial_count = 0;
    _success_count = 0;
    _fail_count = 0;
    _penalty_count = 0;

    _trialMsg->wait_for_message();

    _sync->start();
}

void LabTrial::quit(){
    /*
    Stops timers, camera acquisition and experimental setup.
    */

    if(_curr_trial_stage > -1){
        abort();
    }

    _sync->stop();

    send_stats();

}

bool LabTrial::timedout(){
    /*
    Used to check if mouse is taking too long to complete the task.
    */

    if(_trial_timer.duration() > _max_trial_time){
        return 1;
    } else {
        return 0;
    }
}

void LabTrial::success(){
    /*
    When mouse successfully completes a task we register the success and start
    a new trial after the ITI.
    */

    _trialMsg->send_trial_info(_trial_ID, _trial_count, 1, (int) _trial_timer.duration());
    
    _success_count++;
    _curr_trial_stage = 0;

    _reset_iti(_min_iti, _max_iti + 1, true);
}

void LabTrial::fail(){
    /*
    In case the mouse takes too long the trial fails and we start a new trial
    after the ITI.
    */
    _trialMsg->send_trial_info(_trial_ID, _trial_count, -1, (int) _trial_timer.duration());
    
    _fail_count++;
    _curr_trial_stage = 0;

    _reset_iti(_min_iti, _max_iti + 1, true);
}


void LabTrial::_reset_iti(int min, int max, bool do_print){
    /*
    Resets the intertrial time randomly and uniformly between the specified min and max.
    */

    _trial_timer.reset();
    _intertrial_time = random(min, max);
    if (do_print) { _trialMsg->send_trial_info(_ITI_ID, _trial_count, _curr_trial_stage, _intertrial_time); }
}

bool LabTrial::iti_completed(){
    return ((_trial_timer.duration() > _intertrial_time) && (_curr_trial_stage == 0));
}

void LabTrial::time_penalty(){
    /*
    If the mouse pokes the rewardbox before it is ready we can restart the ITI by
    using this method. 
    The time penalty has a cool down so that it doesn't go off too often (often the
    mouse pokes around the reward box multiple times in short succession).
    Also we disable the time penalty just after the trial timer is restarted, so that
    we don't incur a penalty when the mouse is claiming its reward.
    */

    if ((_penalty_cooldown_timer.duration() > _penalty_cooldown) && _trial_timer.duration() > 2000){
        _reset_iti(_intertrial_time, _intertrial_time,false);
        _trialMsg->send_data_state_value(_penalty_ID, 1, _penalty_count);
        _penalty_cooldown_timer.reset();
        _penalty_count++;
        _trialMsg->send_trial_info(_penalty_ID, _penalty_count, -3, _intertrial_time);
    }
}

void LabTrial::_penalty(){
     if ((_penalty_cooldown_timer.duration() > _penalty_cooldown) && _trial_timer.duration() > 2000){
        _reset_iti(_intertrial_time, _intertrial_time,false);
        _trialMsg->send_data_state_value(_penalty_ID, 1, _penalty_count);
        _penalty_cooldown_timer.reset();
        _penalty_count++;
        _trialMsg->send_trial_info(_penalty_ID, _penalty_count, -3, _intertrial_time);
    }
}

void LabTrial::penalty_test(){
    (this->*_penalty_func)();
}

void LabTrial::_dummy_penalty(){}

void LabTrial::toggle_penalty(){
    if (_using_penalty){
        _penalty_func = &_dummy_penalty;
    } else {
        _penalty_func = &_penalty;
    }

    _using_penalty = !_using_penalty;
    _trialMsg->send_trial_info(_penalty_ID, _penalty_count, 0, _using_penalty);
}

void LabTrial::stage_completed(){
    /*
    Used after succesful completion of previous stage, unless it is the last stage.
    */
    _curr_trial_stage++;
}

int LabTrial::get_stage(){
    /*
    If the experimental setup has multiple stages we can use this method to
    keep track of which stage we are in.
    */
    return _curr_trial_stage;
}

void LabTrial::send_stats(){
    /*
    Print some statistics at the end of the run.
    */
    _trialMsg->send_trial_info(_stats_ID, _run_count, 0, _trial_count);
    _trialMsg->send_trial_info(_stats_ID, _run_count, 1, _success_count);
    _trialMsg->send_trial_info(_stats_ID, _run_count, -1, _fail_count);
    _trialMsg->send_trial_info(_stats_ID, _run_count, -2, _abort_count);
    _trialMsg->send_trial_info(_stats_ID, _run_count, -3 ,_penalty_count);
}