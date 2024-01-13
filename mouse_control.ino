#include <Mouse.h>
#include <cmath>

class SinglePendulumControlModel
{
  public:
  // mouse movement increment
  float dx = 0;
  // mouse movement executed
  bool mouse_moved = false;
  // time variables
  float time_0;
  float time_1;
  // serial variable
  char variable;
  // data index
  int index = 0;
  // mouse position
  char xm[10];
  String xm_S;
  float xm_f = 0;
  // base postion
  char x[10];
  String x_S;
  float x_f = 0;
  // base velocity
  char xdot[10];
  String xdot_S;
  float xdot_f = 0;
  // angle of first link
  char angle[10];
  String angle_S;
  float angle_f = 0;
  // angular velocity of first link
  char angledot[10];
  String angledot_S;
  float angledot_f = 0;
  // screen-size/simulation-unit scale factor
  char s[10];
  String s_S;
  float s_f = 0;
  // pendulum radius
  char r[10];
  String r_S;
  float r_f = 0;
  // specific energy
  float energy = 0
  // position,velocity, and acceleration commands
  float command_x = 0;
  float command_xdot = 0;
  float command_xdotdot = 0;
  // assign serial byte to proper data array
  void read_serial(char incomingByte){
    if(incomingByte == '\n'){
      calculate_mouse_movement();
      index = 0;
    }else if(isalpha(incomingByte)){
      variable = incomingByte;
      index = 0;
    }else{
      if(variable == 'm'){
        xm[index] = incomingByte;
        index++;
        xm[index]='\0';
      }else if(variable == 'x'){
        x[index] = incomingByte;
        index++;
        x[index]='\0';
      }else if(variable == 'v'){
        xdot[index] = incomingByte;
        index++;
        xdot[index]='\0';
      }else if(variable == 'a'){
        angle[index] = incomingByte;
        index++;
        angle[index]='\0';
      }else if(variable == 'w'){
        angledot[index] = incomingByte;
        index++;
        angledot[index]='\0';
      }else if(variable == 's'){
        s[index] = incomingByte;
        index++;
        s[index]='\0';
      }else if(variable == 'r'){
        r[index] = incomingByte;
        index++;
        r[index]='\0';
      }
    }
  }

  // update floating point data values
  void update_data(){
    xm_S = xm;
    xm_f = xm_S.toFloat();
    x_S = x;
    x_f = x_S.toFloat();
    xdot_S = xdot;
    xdot_f = xdot_S.toFloat();
    angle_S = angle;
    angle_f = angle_S.toFloat();
    angledot_S = angledot;
    angledot_f = angledot_S.toFloat();
    s_S = s;
    s_f = s_S.toFloat();
  }
  // calculate acceleration of postion command
  void calculate_acceleration_command(){
    command_xdotdot = 1*(80*(0.09*(x_f+0.8*xdot_f) + 3.14159/2 - fmod(6.28318+fmod(angle_f,6.28318),6.28318)) - 30*angledot_f);
  }
  // update postion command and its velocity from calculated acceleration
  void update_position_and_velocity_commands(){
    command_x += command_xdot*0.001*(time_1-time_0) + 0.5*command_xdotdot*(pow(0.001*(time_1-time_0),2));
    command_xdot += command_xdotdot*0.001*(time_1-time_0);
  }
  // manually set position command and its/velocity/acceleration
  void set_commands(float c_x, float c_xdot, float c_xdotdot){
    command_x = c_x;
    command_xdot = c_xdot;
    command_xdotdot = c_xdotdot;
  }
  // calculate mouse movement from available data
  void calculate_mouse_movement(){
    update_data();
    time_1 = millis();
    calculate_acceleration_command();
    update_position_and_velocity_commands();
    dx = (command_x - xm_f)*s_f;
    if(dx>127){dx=127;}
    if(dx<-127){dx=-127;}
    mouse_moved = false;
    time_0 = time_1;
  }
};
// activation pin
int activation_pin = 4;
SinglePendulumControlModel model;
void setup() {
  // put your setup code here, to run once:
  digitalWrite(activation_pin,HIGH);
  model.time_0 = millis();
  Serial.begin(115200);
  Mouse.begin();
}

void loop() {
  // put your main code here, to run repeatedly:
  // if activation pin is high, execute control
  if(digitalRead(activation_pin)){
    while(Serial.available() > 0){
      // read the incoming byte:
      char incomingByte = Serial.read();
      model.read_serial(incomingByte);
      // move mouse if necessary
      if(!model.mouse_moved){
        Mouse.move(model.dx,0);
        model.mouse_moved = true;
      }
    }
  }//else if actiavation pin is low, just read data and let model follow position of base from data
  else{
    while(Serial.available() > 0){
      char incomingByte = Serial.read();
      model.read_serial(incomingByte);
      model.time_0 = millis();
      model.set_commands(model.x_f,0,0);
    }
  }
}
