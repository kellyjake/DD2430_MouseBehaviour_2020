// Behavioural script V - For executing experiment
// (13/07/2020) Magnus Pierrau
// Magnus Pierrau

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
// Trial time variables
//

int min_intertrial_time = 0;   //max & min durations
int max_intertrial_time = 0;
int max_trial_duration  = 10000;         //Longest trial time permissible
int penalty_cooldown = 0;
//
// Laser 1 parameters (635)
//

int laser1_start_delay  = 0;            // Time to start (ms)
int laser1_time_on      = 100;         // Duration of the on pulse (ms)
int laser1_time_off     = 0;            // Duration of the off pulse (ms)
int laser1_pulse_count  = 1;            // Number of pulses per sequence

int laser1_auto_min_wait = 30000;
int laser1_auto_max_wait = 50000;

//
// Laser 2 parameters (473)
//

int laser2_start_delay    = 0;          // Time to start (ms)
int laser2_time_on        = 100;         // Duration of the on pulse (ms)
int laser2_time_off       = 00;         // Duration of the off pulse (ms)
int laser2_pulse_count    = 1;          // Number of pulses per sequence

int laser2_auto_min_wait = 30000;
int laser2_auto_max_wait = 50000;

//
// Pump parameters
//

int pump_duration          = 200;

// --END_CONFIGURABLE_VARIABLES--

#include <LabLaser.h>
#include <LabPokebox.h>
#include <LabCueLed.h>
#include <LabPump.h>
#include <LabMessenger.h>
#include <LabTrial.h>
#include <LabSync.h>

#define baudrate 115200

// THIS VALUE MUST BE AN INTEGER THAT DIVIDES 1000 EVENLY.
// It sets the frequency of the ISR, which in turn determines the FPS of the camera (freq 500 -> 1000/500= 2 ms per frame )
#define frequency 500   

// ############## OK TO EDIT BELOW ##############

//
// Declare all instances of the class objects we will be using
//

LabLaser laser1;
LabLaser laser2;
LabCueLed rb_led;
LabCueLed tb_led;
LabPokebox rb_nosepoke;
LabPokebox tb_nosepoke;
LabPump pump;
LabMessenger msger;
LabTrial trial;
LabSync sync;

// Unique ID's for saving data

const char laser1_ID[7]             = "Laser1";
const char laser2_ID[7]             = "Laser2";
const char trial_photores_ID[9]     = "TrialBox";
const char trial_led_ID[9]          = "TrialLED";
const char reward_photores_ID[10]   = "RewardBox";
const char reward_led_ID[10]        = "RewardLED";
const char pump_ID[5]               = "Pump";



//
// Pins and settings to use while testing in the office
//
/*
const int laser1_pin            = 5;
const int laser2_pin            = 8;
const int trial_led_pin         = 2;
const int trial_photores_pin    = A0;
const int reward_led_pin        = 12;
const int reward_photores_pin   = A1;
const int pump_pin              = 7;

const bool tb_read_digital            = false;
const bool rb_read_digital            = false;
*/


// 
// Pins and settings to use for real experimental setup
//


const int laser1_pin = 10;
const int laser2_pin = 40;
const int trial_led_pin = 4;
const int trial_photores_pin = 50;
const int reward_led_pin = 2;
const int reward_photores_pin = 52;
const int pump_pin = 7;

const bool tb_read_digital            = true;
const bool rb_read_digital            = true;



//
// Other variables
//

volatile uint32_t iterator = 0; 


//TODO: 
//https://stackoverflow.com/questions/28847007/what-is-the-correct-way-of-using-c-objects-and-volatile-inside-interrupt-rou
//Her bør du bør du bruke switch case, den er mere effektiv til slike oppgaver


//
// Setup of Arduino
//

void setup() {
    Serial.begin(baudrate);

    msger       = LabMessenger(&Serial, &iterator);

    tb_nosepoke = LabPokebox(trial_photores_pin, trial_photores_ID, tb_read_digital, &msger);
    
    rb_nosepoke = LabPokebox(reward_photores_pin, reward_photores_ID, rb_read_digital, &msger);

    pump        = LabPump(pump_pin, pump_ID, pump_duration, &msger);

    tb_led      = LabCueLed(trial_led_pin,trial_led_ID, &msger);

    rb_led      = LabCueLed(reward_led_pin,reward_led_ID, &msger);

    laser1      = LabLaser(laser1_pin, laser1_ID, laser1_pulse_count, laser1_time_on, laser1_time_off, laser1_start_delay, laser1_auto_min_wait, laser1_auto_max_wait, &msger);

    laser2      = LabLaser(laser2_pin, laser2_ID, laser2_pulse_count, laser2_time_on, laser2_time_off, laser2_start_delay, laser2_auto_min_wait, laser2_auto_max_wait, &msger);

    sync        = LabSync(&iterator, frequency, &msger);
    
    trial       = LabTrial(min_intertrial_time, max_intertrial_time, max_trial_duration, penalty_cooldown, &sync, &msger);

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
        tb_led.turn_on();
      
    } else if (inchar == 'b'){
        
        trial.abort();
        tb_led.turn_off();
        rb_led.turn_off();

    } else if(inchar == 'c'){

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
        tb_led.turn_off();
        rb_led.turn_off();
        trial.reset();

    } else if(inchar == 'Q'){
        laser1.decouple();
        laser1.auto_stop();
        laser2.auto_stop();
        tb_led.turn_off();
        rb_led.turn_off();
        trial.quit();
    }
  }

    if(trial.get_stage() == 0){
        // Stage 0 == In ITI

        if(trial.iti_completed()){
            // Once ITI has passed - start trial.
            trial.start();
            tb_led.turn_on();
        }

    } else if (trial.get_stage() == 1){
        // Stage 1 == Trial box stage running

        if(tb_nosepoke.is_broken()){
            tb_led.turn_off();
            rb_led.blink();
            trial.stage_completed();
        }

    } else if (trial.get_stage() == 2){
        // Stage 2 == Trial box poked -> Reward stage running

        if(rb_nosepoke.is_broken()){
            rb_led.turn_off();
            pump.start();
            trial.success();
        }
    }

    // If stage is neither 0, 1 or 2 we do nothing.

    tb_led.update();
    rb_led.update();
    pump.update();
    laser1.update();
    laser2.update();
}

//DO NOT EDIT OR MOVE THIS LINE OF CODE
//This is a Interrupt Service Routine which sends timestamps in unison with the camera
ISR(TIMER1_COMPA_vect){sync._ISR_func();}
