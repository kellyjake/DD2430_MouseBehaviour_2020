/*--------------------------------Licence------------------------------------------
	This is free and unencumbered software released into the public domain.

	Anyone is free to copy, modify, publish, use, compile, sell, or
	distribute this software, either in source code form or as a compiled
	binary, for any purpose, commercial or non-commercial, and by any
	means.

	In jurisdictions that recognize copyright laws, the author or authors
	of this software dedicate any and all copyright interest in the
	software to the public domain. We make this dedication for the benefit
	of the public at large and to the detriment of our heirs and
	successors. We intend this dedication to be an overt act of
	relinquishment in perpetuity of all present and future rights to this
	software under copyright law.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
	IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
	OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
	OTHER DEALINGS IN THE SOFTWARE.

	For more information, please refer to <http://unlicense.org/>


Comments from the author: This code can affect functions in the arduino. Things like PWM, delay, millis may be affected, and may not work 
properly

Author: Espen Teigen 
Date: 2020.08.05

-----------------------------------------------------------------------------------



---------------------------About frequency calculations----------------------------
Formula to calculate prescaler and output compare value. Value must be integer

OCR1A = (FREQ_CLK - 2 * FREQ_PIN * prescaler) / (2 * FREQ_PIN * prescaler)

example
clock on Arduino Mega is
freq_clk = 16MHz

We want
freq_pin = 100Hz

We try to use prescaler 
N = 8

OCR1A = (16 000 000 - 2 * 100 * 8) / (2 * 100 * 8) = 9999
OCR1A = (16 000 000 - 2 * 50 * 8) / (2 * 50 * 8) = 19999
OCR1A = (16 000 000 - 2 * 50 * 8) / (2 * 50 * 8) = 4999

-----------------------------------------------------------------------------------*/


//---------------------------The Code-----------------------------------------------


#include <avr/io.h>
#include <avr/interrupt.h>

//shortcut to Turn on square wave and interrupt ISR
#define SQUAREWAVE_ON() TCCR1A |= (1 << COM1A0)

//shortcut to turn off square wave and interrupt ISR
#define SQUAREWAVE_OFF() TCCR1A &= (1 << COM1A0)




volatile bool flag;

void setup() {
	//Turn off interrupts
	cli();
	//Clear registers
	TCCR1A = 0;
	TCCR1B = 0;
	TIMSK1 = 0;

	//Make sure pull-up is turned of and set pin 11 as output
	PORTB &= (1 << PORTB5);
	DDRB |= (1 << DDB5);

	//Turn on Clear timer on compare and set prescaler to 8
	TCCR1B |= (1 << WGM12) | (1 << CS11);

	//Turn on interrupt sub-routine TIMER1_COMPA_vect
	TIMSK1 |= (1 << OCIE1A);
	
	//Set timer compare value we calculated
	OCR1A = 4999;

	//Turn on Square wave on pin and interrupt sub-routine
	//SQUAREWAVE_ON();

	//Turn on interrupts
	//sei();

  flag = false;
  Serial.begin(115200);

}

void loop() {
  //sei();
  if (Serial.available()){
    cli();
    char in_char = Serial.read();
    //delay(2000);
    Serial.println(in_char);
    //delay(2000);
    
    if (in_char == "a"){
      //sei();
      flag = true;
    } else if (in_char == "b"){
      //cli();
      flag = false;
    }
  }
  /*
  if (flag == true){
    cli();
    Serial.println("Doing stuff in main");
    delay(1000);
    sei();
  }
  */
}


/*This sub-routine runs everytime the pin changes(This means that with the values used, this wil run 200 times pr second)
Remember to do as little as possible in this routine
*/

ISR(TIMER1_COMPA_vect){

//Measure time her, remember that the variable you store the time in must be volatile
  //Serial.printlnln("---- In ISR ----");
  Serial.println(millis());
  //Serial.printlnln("-- End of ISR --");
}
