#include "LabLaser.h"

#include "Arduino.h"

/*
    LabLaser.cpp - Library for controlling Lasers in experimental setup.
    
    Parameters:
    laser_pin     = Pin on arduino board
    laser_ID      = Name for laser. Set in main script.
    max_cycles    = Number of pulse cycles
    time_on       = Time on for each pulse
    time_off      = Time off between pulses
    time_delay    = Add delay until firing laser
    auto_min_wait = Lower interval limit for random laser firings wait time
    auto_max_wait = Upper interval limit for random laser firings wait time
    inMsg         = Pointer to a LabMessenger class object. This is how
                    the LabLaser class object sends information to the OS.
    
    Laser states:
    0 = Off
    1 = On
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/


LabLaser::LabLaser()
{

}

LabLaser::LabLaser(const int laser_pin, const char laser_ID[], const int max_cycles, const int time_on, const int time_off, const int time_delay, const int auto_min_wait, const int auto_max_wait, LabMessenger *inMsg){
  _pin = laser_pin;

  _max_cycles_count = max_cycles;
  _t_on = time_on;
  _t_off = time_off;
  _t_delay = time_delay;

  _auto_min_wait = auto_min_wait;
  _auto_max_wait = auto_max_wait;

  _do_auto_run = 0;
  
  pinMode(_pin, OUTPUT);
  _lasMsg = inMsg;
  
  _delay_clock = LabTimer();
  _cycle_clock = LabTimer();
  _auto_clock = LabTimer();

  _coupled = 0;

  _lasMsg = inMsg;

  _laser_ID = laser_ID;
}

void LabLaser::update(){
  /*
  At the end of each loop cycle we update the laser to see if it is time to 
  change the state of the laser (turn on/off).
  */

  if(_do_auto_run){
    LabLaser::auto_update();
  }
  
  if(!_is_active) return;

  if(_cycles_count >= _max_cycles_count) {
    _is_active=0;
    return;
  }

  double t_cycle = _cycle_clock.duration();

  if(_wait){
    if(_delay_clock.duration() < _t_delay) return;
    _wait = 0;
  }

  if((!_is_on && t_cycle >=_t_off) || (_should_start)){
    _lasMsg->send_data_state_value(_laser_ID, 1, _t_on);
    digitalWrite(_pin, HIGH);
    _is_on=1;
    _should_start=0;
    _cycle_clock.reset();

  }else if(_is_on && (t_cycle >= _t_on)){
    _lasMsg->send_data_state_value(_laser_ID, 0, _t_off);
    digitalWrite(_pin, LOW);
    _is_on=0;
    _cycles_count++;
    _cycle_clock.reset();
  }
}

void LabLaser::start(){
  /*
  Set variables so that the laser fires at the next update.
  */

  // Start the coupled laser if coupling is activated
  if(_coupled) _coupled_laser->start();
  
  _is_active = 1;
  _should_start = 1;
  _cycles_count = 0;

  _is_on = 0;
  _cycle_clock.reset();

  if(_t_delay>0){
    _wait=1;
  }
  _delay_clock.reset();
}

void LabLaser::auto_run(){
  /*
  Makes the laser fire randomly at times between _auto_min_wait and _auto_max_wait
  */

  _do_auto_run = 1;
  _auto_clock.reset(); 
  _auto_wait_time = random(_auto_min_wait, _auto_max_wait + 1);
}

void LabLaser::auto_stop(){
  /*
  Stop autorunning laser.
  */
  _do_auto_run = 0;
}

void LabLaser::auto_update(){
  /*
  This is run at laser.update() if _do_auto_run == 1
  */
  if((_auto_clock.duration() > _auto_wait_time) && _do_auto_run){
    LabLaser::start();
    _auto_wait_time = random(_auto_min_wait, _auto_max_wait + 1);
    _auto_clock.reset();
  }
}

void LabLaser::couple(LabLaser *coupled_laser){
  /*
  This method couples the current laser with the other given laser.
  This makes the coupled laser fire at the same time as the current laser.
  The coupled laser will fire according to its parameters, but the start 
  time will be simultaneous for both lasers.
  */

  _coupled = 1;
  _coupled_laser = coupled_laser;
}

void LabLaser::decouple(){
  /*
  The current laser will retain the pointer to the coupled laser, but they will 
  not longer fire together.
  */
  _coupled = 0;
}