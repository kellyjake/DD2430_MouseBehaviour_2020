// Behavioural script IV - For conditioning
// (13/07/2020) 
// Magnus Pierrau

// WHAT DOES THIS SCRIPT DO?
// This script includes both the behavioural routine and the lasers.
// GO ON...

// HOW TO CONFIGURE
// Add all valid keyboard options below followed by -- and a one line description.
// Follow this by another -- and enter the variable name you wish to save the command as.
// The first value will show up as an executable option when running the trial and the 
// second will indicate the ID of the event in the saved csv file.

// Description may not include "--". This will cause incomplete descriptions.
// Option keys must char type in code as commands sent from Python script are strings.
// Do not remove any of this text. Only add options below --BEGIN_OPTION_SETUP--.

// Example: 
// a--Turn LED on
// b--Turn LED off

// ############## DO NOT EDIT BELOW ##############

// --BEGIN_OPTION_SETUP--
// a--Start new trial
// b--Abort current trial
// c--Activate lasers 1 & 2
// d--Activate laser 1
// e--Activate laser 2
// f--Auto run laser 1 & 2
// g--Auto run laser 1
// h--Auto run laser 2
// i--Stop auto run
// j--Toggle penalty On/Off
// --END_OPTION_SETUP--



// Add any variable you want to be able to update during the run below.
// --BEGIN_CONFIGURABLE_VARIABLES--

//
// Trial variables
//

int min_intertrial_time = 10000;   // Max & min durations
int max_intertrial_time = 10000;
int max_trial_duration  = 10000;   // Longest trial time permissible
int penalty_cooldown = 1000;

//
// LabLaser 1 parameters (635)
//

int laser1_start_delay = 0;     // Time to start (ms)
int laser1_time_on  = 2000;     // Duration of the on pulse (ms)
int laser1_time_off = 0;        // Duration of the off pulse (ms)
int laser1_pulse_count = 1;     // Number of pulses per sequence


int laser1_auto_min_wait = 30000;
int laser1_auto_max_wait = 60000;

//
// LabLaser 2 parameters (473)
//

int laser2_start_delay  = 1000;      // Time to start (ms)
int laser2_time_on      = 15;     // Duration of the on pulse (ms)
int laser2_time_off     = 10;     // Duration of the off pulse (ms)
int laser2_pulse_count  = 40;      // Number of pulses per sequence

int laser2_auto_min_wait = 30000;
int laser2_auto_max_wait = 60000;

//
// Pump parameters
//

int pump_duration = 200;

// --END_CONFIGURABLE_VARIABLES--

#include <LabLaser.h>
#include <LabPokebox.h>
#include <LabCueLed.h>
#include <LabPump.h>
#include <LabMessenger.h>
#include <LabTrial.h>
#include <LabSync.h>

#define baudrate 115200
#define frequency 500   // THIS VALUE MUST BE AN INTEGER THAT DIVIDES 1000 EVENLY

// ############## OK TO EDIT BELOW ##############

//
// Declare all instances of the class objects we will be using
//

LabLaser laser1;
LabLaser laser2;
LabCueLed rb_led;
LabPokebox rb_nosepoke;
LabPump pump;
LabMessenger msger;
LabTrial trial;
LabSync sync;

const char laser1_ID[7]             = "Laser1";
const char laser2_ID[7]             = "Laser2";
const char reward_photores_ID[10]   = "RewardBox";
const char reward_led_ID[10]        = "RewardLED";
const char pump_ID[5]               = "Pump";



//
// Pins to use while testing in the office
//
/*

const int laser1_pin                  = 10;
const int laser2_pin                  = 13;
const int reward_led_pin              = 12;
const int reward_photores_pin         = A1;
const int pump_pin                    = 7;

const bool rb_read_digital            = false;
*/

//
// Pins to use for real experimental setup
//
 

const int laser1_pin = 10;
const int laser2_pin = 40;
const int reward_led_pin = 2;
const int reward_photores_pin = 52;
const int pump_pin = 7;

const bool rb_read_digital = true;




//
// Other variables
//

volatile uint32_t iterator = 0;

//
// Setup of Arduino
//

void setup() {
    Serial.begin(baudrate);

    msger      = LabMessenger(&Serial, &iterator);

    rb_nosepoke = LabPokebox(reward_photores_pin,reward_photores_ID,rb_read_digital,&msger);

    pump        = LabPump(pump_pin, pump_ID, pump_duration, &msger);

    rb_led      = LabCueLed(reward_led_pin,reward_led_ID, &msger);
            
    laser1      = LabLaser(laser1_pin, laser1_ID, laser1_pulse_count, laser1_time_on, laser1_time_off, laser1_start_delay, laser1_auto_min_wait, laser1_auto_max_wait, &msger);

    laser2      = LabLaser(laser2_pin, laser2_ID, laser2_pulse_count, laser2_time_on, laser2_time_off, laser2_start_delay, laser2_auto_min_wait, laser2_auto_max_wait, &msger);

    sync        = LabSync(&iterator, frequency, &msger);

    trial       = LabTrial(min_intertrial_time, max_intertrial_time, max_trial_duration, penalty_cooldown, &sync, &msger);
    
    // Wait for setup to be ready
    msger.wait_for_message();
    // Ask camera to start and start internal clock
    sync.start(); 
}

//
// Begin loop
//

void loop() {
  
  if(Serial.available()){ //Read single characters from serial port and handle appropriately
    
    char inchar = Serial.read();

      if(inchar == 'a'){
        trial.start();
        rb_led.blink();

      } else if(inchar == 'b') { 
        
        trial.abort();
        rb_led.turn_off();

      } else if(inchar == 'c') {
        
        laser1.start();
        laser2.start();

      } else if(inchar == 'd'){
        
        laser1.start();

      } else if(inchar == 'e'){
        
        laser2.start();

      } else if(inchar == 'f'){
        
        laser1.couple(&laser2);
        laser1.auto_run();

      } else if(inchar == 'g'){
        
        laser1.decouple();
        laser1.auto_run();

      } else if(inchar == 'h'){

        laser1.decouple();
        laser2.auto_run();

      } else if(inchar == 'i'){

        laser1.decouple();
        laser1.auto_stop();
        laser2.auto_stop();
        
      } else if(inchar == 'j'){

        trial.toggle_penalty();

      } else if(inchar == 'R'){

        laser1.decouple();
        laser1.auto_stop();
        laser2.auto_stop();
        rb_led.turn_off();
        trial.reset();

      } else if(inchar == 'Q'){
        laser1.decouple();
        laser1.auto_stop();
        laser2.auto_stop();
        rb_led.turn_off();
        trial.quit();
      }
    }

    if(trial.get_stage() == 0){
        // Stage 0 == In ITI
      
        if(rb_nosepoke.is_broken()){

            // If box poked before ready, restart ITI from beginning (same value).
            // Now has cooldown so penalties don't stack like before.

            trial.time_penalty();    // UNCOMMENT THIS TO ADD TIME PENALTY

        } else if(trial.iti_completed()){

            // Once ITI has passed - start trial.
            trial.start();
            rb_led.blink();
        }
    }
    else if(trial.get_stage() == 1){
        // Stage 1 == Trial running

        if(rb_nosepoke.is_broken()){

            // If box poked - give reward -> restart (after given ITI)
            trial.success();
            
            pump.start();
            rb_led.turn_off();

        } else if(trial.timedout()){

            // If took too long - no reward -> restart (after given ITI)
            trial.fail();
            rb_led.turn_off();

        }
    }

    // If stage is neither 0 or 1 then we do nothing.
    
  rb_led.update();
  pump.update();
  laser1.update();
  laser2.update();
}

//DO NOT EDIT OR MOVE THIS LINE OF CODE
//This is a Interrupt Service Routine which sends timestamps in unison with the camera
ISR(TIMER1_COMPA_vect){sync._ISR_func();}