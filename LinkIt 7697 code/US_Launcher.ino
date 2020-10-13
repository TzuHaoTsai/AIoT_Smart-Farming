/*
 *  this is for yiso MTK application
 *  use 7697 for controller, and S76S for lora module
 *  use RF command for communication
 *  master to client: 112233
 *  11: client id
 *  22: function, 00: read status, 11: set status, 
 *  33: status,   00: power off,   11: power on
 *  design by Kenny Huang on 2020/10/3
 *  
 */
#include <hal_wdt.h>  // The header for WDT API

#define power_PORT 12 //12V relay on pin
#define lora_RESET 4  //lora reset pin
#define door_CTRL 17  //door for utra sonic device
#define door_LED 16  //door open or close indicter
#define ir_OUT 8     //infar ray output
#define read_PERIOD 240000  //ms, and need to *3

int voltagebatt01;
float voltagebatt02;

String msgStr = "";
String client_ID="01"; //for id 01;
//String client_ID="02"; //for id 02;
bool  power_status=0; //1:on, 0:off
bool  ir_ON=0;  //1: ir detecting, 0:ir stop
uint8_t  door_status;  //1:open, 0:close
uint8_t  door_cpu_status;  //1:running, 0:stop
unsigned long current, current_door;
unsigned char cnt;


/////////////////////////////////////////////////////////
void measurevoltage() {
    voltagebatt01=analogRead(15);      
    //Serial.println(voltagebatt01); 
    voltagebatt02=(voltagebatt01)/327.6;
    Serial.print("battery voltage") ;
    Serial.println(voltagebatt02);     
}
/////////////////////////////////////////////////////////////////////////////////////////// 
void sys_restart(){
  hal_wdt_software_reset(); //will do H/W reset
}
/////////////////////////////////////////////////////////////////////////////////   
void setup() {
  // put your setup code here, to run once:  
  pinMode(ir_OUT, INPUT);
  
  pinMode(power_PORT, OUTPUT);
  digitalWrite(power_PORT, LOW);  //turn 12V relay off
  
  Serial.begin(115200); //in order to follow up lora 115200
                       //so, the debug port need to set to this baud rate
  Serial1.begin(115200);
  Serial1.setTimeout(500);
  //reset lora
  digitalWrite(lora_RESET,LOW);
  delay(100);
  digitalWrite(lora_RESET,HIGH);
  cnt=0;
  while(!Serial1.available()) {
    Serial.print(".");delay(100);
    cnt++;
    if (cnt>30){
      cnt=0;
      Serial.println();
    }
  }
  
  Serial.println();
  while(Serial1.available()){
    Serial.write(Serial1.read()); 
    //since lora reset will response lots of message
    //can not use readString, need to read every character
    
    if (!Serial1.available()) delay(10);//this is must for S76S
  }

  Serial.println("\nSystem starting...");
  Serial1.print("rf rx_con on");
  cnt=0;
  while(!Serial1.available()) {
    Serial.print("r");
    cnt++;
    if (cnt>30){
      cnt=0;
      Serial.println();
    }
  }
  Serial.println(Serial1.readString());
  /////////////////////////////////////////
  
  pinMode(door_CTRL, OUTPUT);
  pinMode(door_LED, INPUT);
  cnt=0;
  current=millis();
  digitalWrite(door_CTRL, LOW); //set key pressed now.
  while(!digitalRead(door_LED) && (millis()-current)<3000){   //need low 2 sec to turn on system.
    digitalWrite(door_CTRL, LOW);
    
  }
  if ((millis()-current)>=3000) sys_restart();  //system reset
  Serial.println("after 2 sec");
  Serial.println("Set button unpress");
  digitalWrite(door_CTRL, HIGH); //normal is HI
  Serial.println("LED on1 now");
  while(digitalRead(door_LED));//when power on , led will flash 2 times, wait till led off
  Serial.println("LED off1");
  //and now, led is off
  current=millis();
  cnt=0;
  while(!digitalRead(door_LED)&& (millis()-current)<1000){ //now, led is off, and wait second time to be on
    delay(10);
    Serial.print("s");
    cnt++;
    if (cnt>30) {
      cnt=0;
      Serial.println();
    }
    
  }
  if ((millis()-current)>=1000) {//means this time is to turn off door cpu
    sys_restart();  //system reset
  }
  
  // now, the power is on for door device, so there is second led
  Serial.println();
  //LED turn on now for the second time
  Serial.println("LED on2 now");
  while(digitalRead(door_LED));//wait till led off, and system reset OK.
  Serial.println("LED off2 now");

  delay(3000);
  
}

void loop() {
    // put your main code here, to run repeatedly:
      set_door(); //need to set the door all the time
      
      msgStr="";
      if (Serial1.available()) {
        msgStr=Serial1.readString();
        Serial.println(msgStr);
      }
      
      if (msgStr.indexOf("radio_rx "+client_ID+"1100")>=0) {//turn off power
        digitalWrite(power_PORT, LOW);//turn off 12V output
        power_status=0;
      }
      else if (msgStr.indexOf("radio_rx "+client_ID+"1111")>=0) {
        digitalWrite(power_PORT, HIGH);//turn on 12V output
        power_status=1;
        set_door();
      }
      else if (msgStr.indexOf("radio_rx "+client_ID+"00")>=0) { //read status
        Serial1.print("rf rx_con off");
        while(!Serial1.available()) {
          Serial.print("!");
          set_door(); //need to keep door at the same status
        }
        Serial.println(Serial1.readString());

        current=millis();
        while((millis()-current)<3000){
          set_door();
        }
        
        if (power_status) {
          Serial.println("rf tx "+client_ID+"0011");//power on
          Serial1.print("rf tx "+client_ID+"0011");//power on
          
        }
        else {
          Serial.println("rf tx "+client_ID+"0000"); //power off
          Serial1.print("rf tx "+client_ID+"0000"); //power off
          
        }
        while(!Serial1.available()) {
          Serial.print("t");
          set_door();
        }
        Serial.println(Serial1.readString());

        current=millis();
        while((millis()-current)<3000){//wait master to be ready
          set_door();
        }
        
        Serial1.print("rf rx_con on");
        while(!Serial1.available()) {
          Serial.print("r");
          set_door();
        }
        Serial.println(Serial1.readString());
      }
    //}
}

//////////////////////////////////////////////////////////////////////////////////
void set_door(){
    if (power_status){  //power on now
      while(!digitalRead(ir_OUT)); //waiting for ir ON

      delay(1); //when ir on, need to keep 1ms hi

        pinMode(ir_OUT, OUTPUT);  //if still hi, need to pull low 1ms for open door
        digitalWrite(ir_OUT, LOW);
        delay(1);
        pinMode(ir_OUT, INPUT);
        return;
      
      ir_ON=1;
      while(ir_ON){
        pinMode(ir_OUT, INPUT);
        delay(1); //when ir on, need to keep 1ms hi
        if(digitalRead(ir_OUT)){  //after 1 ms, check it still hi or not
          pinMode(ir_OUT, OUTPUT);  //if still hi, need to pull low 1ms for open door
          digitalWrite(ir_OUT, LOW);
          delay(1);
        }
        else{ //no, ir is off
          pinMode(ir_OUT, INPUT); //in this mode, door will close automatically
          ir_ON=0;  //exit while(1)
        }      
      }//end of while(1)
      
    } //end of POWER is ON     
}
/////////////////////////////////////////////////////////////////////////
