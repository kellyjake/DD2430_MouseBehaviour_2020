#pragma once

/*
    LabTimer.h - Library for controlling internal Timers for various events
    
    Created by Magnus Pierrau, 16/07/2020
    For use at Karolinska Institutet
*/

#include "Arduino.h"


class TestClass
{
    public:
        TestClass();
        TestClass(int* var[]);
        void SetVal(int idx, int new_val);
        void print();

    private:
        int** _this_var;
};
