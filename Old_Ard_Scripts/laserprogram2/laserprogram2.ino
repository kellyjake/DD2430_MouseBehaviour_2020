// HOW TO USE
//
// Set the parameters as desired below then press the arrow in the top bar
// to start the program. Note that there is no error checking for parameters, 
// so the program may behave in unexpected ways if given nonsensical ones. 
//
// Press ctrl+m to open the command box. Writing 'a' (without quotations marks) 
// activates both laser, 'b' only laser 1 and 'c' only laser2.
// SERIAL PORT COM 3 FOR LASER CONTROLS 


//
// Laser 1 parameters (635)
//

int laser1_start_delay = 0;     // Time to start (ms)
int laser1_time_on  = 2000;     // Duration of the on pulse (ms)
int laser1_time_off = 0;        // Duration of the off pulse (ms)
int laser1_pulse_count = 1;     // Number of pulses per sequence
int laser1_pin = 10;             // Pin number on the Arduino board


//
// Laser 2 parameters (473)
//

int laser2_start_delay = 000;  // Time to start (ms)
int laser2_time_on = 10;        // Duration of the on pulse (ms)
int laser2_time_off = 15;        // Duration of the off pulse (ms)
int laser2_pulse_count =5;     // Number of pulses per sequence
int laser2_pin = 40;            // Pin number on the Arduino





















class Timers {
  public:

  double duration(){
    return millis() - timestamp;
    };
    
  void reset(){
    timestamp = millis();
    }

  private:
    double timestamp;
};

class Laser{
  public:
  
  void init(int laser_pin){
    pin = laser_pin;
    pinMode(pin, OUTPUT);
  }
    
  void update(){
    if(!is_active) return;

    if(cycles_count >= max_cycles_count) {
      is_active=0;
      return;
    }

    double t_cycle = cycle_clock.duration();

    //Wait for start
    if(wait){
      if(delay_clock.duration()<t_delay) return;
      wait = 0;
    }

    

    if((!is_on && t_cycle>=t_off) || (should_start)){

      digitalWrite(pin, HIGH);
      is_on=1;
      should_start=0;
      cycle_clock.reset();
    }else if(is_on && t_cycle>=t_on){

      digitalWrite(pin, LOW);
      is_on=0;
      cycles_count += 1;
      cycle_clock.reset();
    }   
}

    void start(){
      is_active = 1;
      should_start = 1;
      cycles_count=0;

      //Cycle variables
      is_on=0;
      cycle_clock.reset();

      //Delay variables
      if(t_delay>0){
        wait=1;
      }
      delay_clock.reset();
       
    }

    void configure(int max_cycles, int time_on, int time_off, int time_delay){
      max_cycles_count = max_cycles;
      t_on = time_on;
      t_off = time_off;
      t_delay = time_delay;
    }

  private:

  bool is_active;
  bool should_start; 
  
  //Cycle variables
  Timers cycle_clock;
  int t_on;
  int t_off;
  bool is_on;
  int max_cycles_count;
  int cycles_count;

  //Start delay variables
  Timers delay_clock;
  int t_delay;
  bool wait;
  
  int pin; 
   
};





Laser laser1;
Laser laser2;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  
  //
  //Number of cycles, time on, time off, starting delay
  //
  laser2.init(laser2_pin);
  laser2.configure(laser2_pulse_count, laser2_time_on, laser2_time_off, laser2_start_delay);
  
  laser1.init(laser1_pin);
  laser1.configure(laser1_pulse_count, laser1_time_on, laser1_time_off, laser1_start_delay);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  
  if(Serial.available()){
    char inchar = Serial.read();

    
    switch(inchar){
      case 'a':
        Serial.println("LASER 1 & 2 STARTS");
        laser1.start();
        laser2.start();
        break;
      
      case 'b':
        Serial.println("LASER 1 STARTS");
        laser1.start();
        break;
        
      case 'c':
        Serial.println("LASER 2 STARTS");
        laser2.start();
        break;  
      
      }
    }

    laser1.update();
    laser2.update();






}
