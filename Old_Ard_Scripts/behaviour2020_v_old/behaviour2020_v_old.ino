// (12/05)  
// This is the same as program 3 but it now starts/stops automatically

class Timer{
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



class Pump{
 public:
    void init(int duration = 100, int control_pin = 7){
      pinMode(control_pin, OUTPUT);
      digitalWrite(control_pin, HIGH);
      t_max = duration;
      pin = control_pin;
    }
    void set_duration(int duration){
      t_max = duration;  
    }
  void update(){
    double dur = cycle_clock.duration();
    if(is_active != 1) return;
    
    if(dur >= t_max){
      Serial.print("Pump deactivates: ");
      Serial.print(dur);
      Serial.println(" ms");
      is_active = 0;
      digitalWrite(pin, HIGH);
      }

    }

  void start(){
    digitalWrite(pin, LOW);
    is_active = 1;
    cycle_clock.reset();
    }
  void stop(){
    digitalWrite(pin, HIGH);
    is_active = 0;
  }

  void hard_start(){
    digitalWrite(pin, LOW);
  }

  void hard_stop(){
    digitalWrite(pin, HIGH);
  }
  bool is_pumping(){
    return is_active;
    }
  
 private:
 Timer cycle_clock;
 bool is_active;
 int t_max = 100;
 int pin;



};



class Cue_led{

public:
  void init(int led_pin){ 
    pin = led_pin;
    pinMode(pin, OUTPUT);
  }

  void update(){ 
     double t_cycle = cycle_clock.duration();

     if(!is_on && t_cycle >= t_off){
       is_on = 1;
       if(t_on>0) digitalWrite(pin, HIGH);
       cycle_clock.reset();
     }else if(is_on && t_cycle >= t_on){
      is_on = 0;
      if(t_off>0) digitalWrite(pin, LOW);
      cycle_clock.reset();
     }
  }

  void configure(int time_on, int time_off){
      t_on = time_on;
      t_off = time_off;

      is_on=1;
      digitalWrite(pin, HIGH);
      cycle_clock.reset();
  }
    
private:
  bool is_on;
  
  int pin;
  int t_on;
  int t_off;

  Timer cycle_clock;
  };




class Ambient_led{
  public:
  void init(int led_pin){
    pin = led_pin;
    pinMode(pin, OUTPUT);
  }

  void configure(byte led_intensity){
    intensity = led_intensity;
    analogWrite(pin, intensity);
    }

  private:
    int pin;
    byte intensity;
  
  };



class Nosepoke{
public:
  void init(int nosepoke_pin){
      pin = nosepoke_pin;
      pinMode(pin, INPUT);
    }
  //Method to check if the box is being poked
  bool is_broken(){
   
    if(digitalRead(pin) == LOW){
        return 1;
    }

    return 0;
    
    }

private:
  int pin;
  
};


double success_count = 0;
double fail_count    = 0;

bool is_trial = 0; //flag to track whether the training is in its active stage (e.g. blinking led + nosepoke-reward-coupling)

bool check_tb = 0;
bool check_rb = 0;

Cue_led rb_led;
Cue_led tb_led;

Ambient_led amb_led;

Nosepoke rb_nosepoke;
Nosepoke tb_nosepoke;

Pump pump;

Timer trial_timer;
int intertrial_time = -1;                         //time between consecutive trials, is reset randomly
const double min_intertrial_time = 15000;         //max & min durations
const double max_intertrial_time = 25000;
const double max_trial_duration  = 10000;         //Longest trial time permissible




void setup() {
  Serial.begin(115200);
  while (!Serial);
  rb_nosepoke.init(52);
  tb_nosepoke.init(50);
  
  pump.init(200, 7); //Here

  tb_led.init(4);
  tb_led.configure(0,1);
  
  rb_led.init(2);
  rb_led.configure(0,1);

  
  trial_timer.reset();
}


void loop() {
  
  if(Serial.available()){ //Read single characters from serial port and handle appropriately
    char inchar = Serial.read(); 
    Serial.print(inchar);

    if(inchar == 'a'){
      success_count = 0;
      double fail_count    = 0;
      
      Serial.println("[TRIAL] STARTED");
      is_trial = 0;
      trial_timer.reset();
      
      //intertrial_time = random(min_intertrial_time,max_intertrial_time);
      intertrial_time = 0;
        Serial.print("Sets intertrial time to ");  
        Serial.println(intertrial_time);
    }
    if(inchar == 'b'){
      Serial.println("[TRIAL] ABORTED (manually)");
      rb_led.configure(0,1);
      is_trial = 0;
      check_rb = 0;
      check_tb = 0;
      
      trial_timer.reset();
      
      rb_led.configure(0,1);
      tb_led.configure(0,1);
      intertrial_time = -1;
      Serial.print("Number of successful trials: ");
      Serial.println(success_count);
    }
  }

  if(is_trial){
    if(check_tb && tb_nosepoke.is_broken()){
      Serial.println("[TRIAL] TB STAGE COMPLETE");
      tb_led.configure(0,1);
      check_rb = 1;
      check_tb = 0;
      rb_led.configure(10,990);
      }
    if(check_rb && rb_nosepoke.is_broken()){
      Serial.println("[TRIAL] SUCCESSFUL. ");
      Serial.print("\tTook ");
      Serial.println(trial_timer.duration());
      success_count++;
      //Reset timer and itit
      trial_timer.reset();
      Serial.print("Sets intertrial time to ");  
      Serial.println(intertrial_time);
      
      //Reset state variables
      check_rb = 0;
      rb_led.configure(0,1);
      is_trial = 0;

      //Pump reward pump
      //intertrial_time = random(min_intertrial_time,max_intertrial_time);
      intertrial_time = 0;
      pump.start();
      
      }
    
    
    
    }else{
      if((trial_timer.duration() > intertrial_time) && intertrial_time != -1){
        Serial.println("[TRIAL] STARTS TRIAL");
        is_trial = 1;
        trial_timer.reset();
        tb_led.configure(999,1);
        
        check_tb = 1;
      }
   }


  
 tb_led.update();
 rb_led.update();
 pump.update();
}
