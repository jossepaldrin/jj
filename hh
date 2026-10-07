Develop an ESP32 program to read temperature and humidity values using a DHT22 sensor and display the readings using serial communication. 


Live workout :  https://wokwi.com/projects/476219515826726913


Temperature & Humidity Reader:

DHT22  3 pin

Connections :
DHT22
ESP32
VCC
3.3V
DATA
GPIO 18
GND
GND


Code
#include <DHT.h>
 
#define DHT_PIN 18
#define DHT_TYPE DHT22
 
DHT dht(DHT_PIN, DHT_TYPE);
 
void setup() {
  Serial.begin(115200);
  dht.begin();
}
 
void loop() {
 
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
 
  delay(2000);
}



Develop an ESP32 program to read light intensity using an LDR module and display the readings using serial communication. 

ESP32 + LDR Module

Live Workout : https://wokwi.com/projects/476220076419690497


LDR (Light Dependent Resistor) is used to detect the intensity of light.


Connections
LDR Module
ESP32
VCC
3.3V
GND
GND
DO
GPIO 34



 

#define LDR_PIN 34
 
void setup() {
  Serial.begin(115200);
  pinMode(LDR_PIN, INPUT);
}
 
void loop() {
 
  int lightStatus = digitalRead(LDR_PIN);
 
  Serial.print("LDR Status: ");
  Serial.println(lightStatus);
 
  delay(1000);
}


Output : 
LDR Status: 0 
LDR Status: 1 

You can simply cover the LDR with your hand and observe which value appears. 
3-pin modules with VCC, GND, DO have a small potentiometer (screw) on the board. This controls the light/dark threshold. 

Develop an ESP32 program to detect motion using a PIR sensor and display the detection status using serial communication 

PIR Sensor 



A PIR (Passive Infrared) sensor detects movement of people or objects based on infrared radiation.


Live Workout : https://wokwi.com/projects/476221497810891777

PIR
ESP32
VCC
3.3V
GND
GND
OUT/ECHO
GPIO 27




#define PIR_PIN 27

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
}

void loop() {
  if (digitalRead(PIR_PIN) == HIGH) {
    Serial.println("Motion Detected");
  } else {
    Serial.println("No Motion");
  }

  delay(500);
}
