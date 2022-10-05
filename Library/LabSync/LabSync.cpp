#include "LabSync.h"

#include "Arduino.h"



/*
    LabSync.cpp - Library for syncing camera with arduino events.

    Here we make use of the microcontroller ISR (Interrupt Service Routine).
    In the setup we manipulate various entries in the microcontroller register
    (the 1's and 0's in the memory) to create our own clock.

    When we turn on _START_CAMERA_ACQUISITION() (defined in the header file for brevity)
    we send a square wave pulse to the camera via pin 11, which functions as an external hardware
    trigger. This makes sure that we capture images every 2 ms (given that freq=500 hz),
    and makes the clock move one tick ahead every 2 milliseconds.

    The interrupt routine runs 500 times every second and returns a timestamp for that
    frame to the computer. At the same time this is the clock used by all classes in 
    the custom Lab-Library I have created. This should give us very precise sync.

    cli() stops the interrupts and sei() resumes them.
    
    Created by Magnus Pierrau and Espen Teigen, 10/08/2020
    For use at Karolinska Institutet
*/


/* ---------------------------About frequency calculations----------------------------
    Formula to calculate prescaler and output compare value. Value must be integer

    OCR1A = (FREQ_CLK - 2 * FREQ_PIN * prescaler) / (2 * FREQ_PIN * prescaler)

    example
    clock on Arduino Mega is
    freq_clk = 16MHz

    We want
    freq_pin = 500Hz

    Freq. of 500 Hz means the interrupt sequence will run 500 times in a second (every 2 ms).
    This is the max FPS of the camera.

    We try to use prescaler 
    N = 8

    OCR1A = (16 000 000 - 2 * 500 * 8) / (2 * 500 * 8) = 1999

-----------------------------------------------------------------------------------*/

LabSync::LabSync(){

}

LabSync::LabSync(volatile uint32_t *iterator, int freq, LabMessenger *inMsg){
    _syncMsg = inMsg;
    _iterator = iterator;
    _freq = freq;

    /*
        Below is the setup for the use of the Interrupt Service Routine used to 
        sync the camera and arduino. Do not edit anything below.
    */

    //Turn off interrupts
    cli();
    
    //Clear registers
    TCCR1A = 0;
    TCCR1B = 0;
    TIMSK1 = 0;
  
    //Make sure pull-up is turned of and set pin 11 as output
    PORTB &= (1 << PORTB5);
    DDRB |= (1 << DDB5);
  
    // Clear timer on compare and set prescaler to 8
    TCCR1B |= (1 << WGM12) | (1 << CS11);
  
    //Set timer compare value we calculated
    OCR1A = (_freq_clk - 2 * _freq * _prescaler) / (2 * _freq * _prescaler);
    
    //Turn on interrupts
    sei();
}

void LabSync::start(){
    _START_CAMERA_ACQUISITION();
    _syncMsg->send_timestamp();
    _is_stopped = false;
}

void LabSync::stop(){
    _STOP_CAMERA_ACQUISITION();

    _syncMsg->send_timestamp();
    _syncMsg->reset_timestamp();
    _is_stopped = true;
}


void LabSync::_ISR_func(){
    /*
    We increase the iterator (clock) by 500/_freq each tick.
    Given that freq=500 means that we increase it by 1 each tick.
    */
    cli();
    *_iterator = *_iterator + 500/_freq;
    sei();
}

bool LabSync::is_stopped(){
    return _is_stopped;
}