#include <TestClass.h>
#include <LabLaser.h>
#include <LabMessenger.h>

int testVar = 1;
char c;
char v;

int laser1_start_delay  = 0;            // Time to start (ms)
int laser1_time_on      = 2000;         // Duration of the on pulse (ms)
int laser1_time_off     = 0;            // Duration of the off pulse (ms)
int laser1_pulse_count  = 1;            // Number of pulses per sequence

int laser1_auto_min_wait = 10000;
int laser1_auto_max_wait = 20000;

//
// Laser 1 parameters (635)
//

int laser2_start_delay    = 0;          // Time to start (ms)
int laser2_time_on        = 15;         // Duration of the on pulse (ms)
int laser2_time_off       = 2;         // Duration of the off pulse (ms)
int laser2_pulse_count    = 50;          // Number of pulses per sequence

int laser2_auto_min_wait = 10000;
int laser2_auto_max_wait = 20000;

LabLaser laser1;
LabLaser laser2;
LabMessenger msger;
TestClass tester;

const char laser1_ID[7] = "Las1";
const char laser2_ID[7] = "Las2";

const int laser1_pin = 10;
const int laser2_pin = 40;

volatile uint32_t iterator = 0;

int* var_list[] = {&laser1_auto_max_wait, &laser2_auto_max_wait};

void setup(){
  Serial.begin(115200);

    tester = TestClass(var_list);

    msger = LabMessenger(&Serial, &iterator);

    laser1      = LabLaser(laser1_pin, laser1_ID, laser1_pulse_count, laser1_time_on, laser1_time_off, laser1_start_delay, laser1_auto_min_wait, laser1_auto_max_wait, &msger);

    laser2      = LabLaser(laser2_pin, laser2_ID, laser2_pulse_count, laser2_time_on, laser2_time_off, laser2_start_delay, laser2_auto_min_wait, laser2_auto_max_wait, &msger);
}

void loop(){
  delay(1000);
  String input;
  String idx;

  if(Serial.available()){
      if (c == 'a'){
        
        Serial.println("Please enter index of variable to change:");
        while(!Serial.available());
        
        idx = Serial.read();

        Serial.print("Got: ");
        Serial.println(idx);

        Serial.println("Please enter an integer value for variable to take: ");

        while(!Serial.available());

        String val;

        while(Serial.available()){
          v = Serial.read();
          val += v;
        }
        
        Serial.print("Read ");
        Serial.println(val);

        tester.SetVal((int) idx.toInt(),(int) val.toInt());
    }
  }
  
  Serial.println("Variable values in tester:");
  tester.print();
  Serial.println("Variable values in main:");
  Serial.println(laser1_auto_max_wait);
  Serial.println(laser2_auto_max_wait);
}
