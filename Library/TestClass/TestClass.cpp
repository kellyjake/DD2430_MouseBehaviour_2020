#include "TestClass.h"

#include "Arduino.h"

/*
    LabTimer.cpp - Library for controlling internal Timers for various events
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/
TestClass::TestClass(){}

TestClass::TestClass(int* var[]){
    _this_var = var;
}

void TestClass::SetVal(int idx, int new_val){
    
    Serial.println("In TestClass::SetVal");

    Serial.println("Setting _this_var to new_val");
    Serial.print("new_val = ");
    Serial.println(new_val);
    Serial.print("*_this_var before = ");
    Serial.println(*_this_var[idx]);

    *_this_var[idx] = new_val;

    Serial.print("*_this_var after = ");
    Serial.println(*_this_var[idx]);
}
      
void TestClass::print(){
    for(int i = 0; i <= sizeof(_this_var) / sizeof(int); i++){
        Serial.println(*_this_var[i]);
    }
}