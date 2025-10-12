// Arduino ESP32 code for SAEAS

#include <Wire.h>
#include <DHT.h>

// Sensor pins
#define DHT_PIN 4
#define DHT_TYPE DHT22
#define TURBIDITY_PIN A0
#define PH_PIN A2
#define MQ135_PIN A1

// Sensor objects
DHT dht(DHT_PIN, DHT_TYPE);

// Calibration values
float turbidityCalibration = 1.0;
float pHCalibration = 7.0;
float airQualityCalibration = 400.0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  // Initialize sensors
  dht.begin();
  pinMode(TURBIDITY_PIN, INPUT);
  pinMode(PH_PIN, INPUT);
  pinMode(MQ135_PIN, INPUT);
  
  // Sensor warm-up
  delay(2000);
  
  Serial.println("SAEAS_ARDUINO_READY");
  Serial.println("SENSOR:INIT:COMPLETE");
}

void loop() {
  // Read all sensors
  float temperature = readTemperature();
  float humidity = readHumidity();
  float turbidity = readTurbidity();
  float pH = readPH();
  float airQuality = readAirQuality();
  
  // Send data in structured format
  Serial.print("SENSOR:");
  Serial.print("TEMP:"); Serial.print(temperature, 1); Serial.print(",");
  Serial.print("HUM:"); Serial.print(humidity, 1); Serial.print(",");
  Serial.print("TURB:"); Serial.print(turbidity, 1); Serial.print(",");
  Serial.print("PH:"); Serial.print(pH, 1); Serial.print(",");
  Serial.print("AQI:"); Serial.print(airQuality, 1);
  Serial.println();
  
  // Check for commands from computer
  if (Serial.available() > 0) {
    String command = Serial.read_stringUntil('\n');
    processCommand(command);
  }
  
  delay(5000); // Send data every 5 seconds
}

float readTemperature() {
  float temp = dht.readTemperature();
  if (isnan(temp)) {
    return -999.0;
  }
  return temp;
}

float readHumidity() {
  float hum = dht.readHumidity();
  if (isnan(hum)) {
    return -999.0;
  }
  return hum;
}

float readTurbidity() {
  int sensorValue = analogRead(TURBIDITY_PIN);
  float voltage = sensorValue * (3.3 / 4095.0);
  float turbidity = voltage * turbidityCalibration * 100;
  return turbidity;
}

float readPH() {
  int measure = analogRead(PH_PIN);
  float voltage = measure * (3.3 / 4095.0);
  float pHValue = pHCalibration + (voltage - 1.5) / 0.18;
  return pHValue;
}

float readAirQuality() {
  int sensorValue = analogRead(MQ135_PIN);
  float voltage = sensorValue * (3.3 / 4095.0);
  float airQuality = (voltage / 3.3) * airQualityCalibration;
  return airQuality;
}

void processCommand(String command) {
  command.trim();
  
  if (command == "PING") {
    Serial.println("PONG");
  } else if (command == "STATUS") {
    Serial.println("STATUS:READY:SENSORS_ACTIVE");
  } else if (command == "RESTART") {
    Serial.println("RESTARTING...");
    ESP.restart();
  } else {
    Serial.println("ERROR:UNKNOWN_COMMAND");
  }
}
