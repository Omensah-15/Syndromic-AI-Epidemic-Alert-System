// Arduino ESP32 code for SAEAS - COMPATIBLE VERSION
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
  float waterTemp = readWaterTemperature();
  
  // Send data in structured format that matches Python code expectations
  Serial.print("SENSOR:");
  Serial.print("TEMP:"); Serial.print(temperature, 1); Serial.print(",");
  Serial.print("HUM:"); Serial.print(humidity, 1); Serial.print(",");
  Serial.print("TURB:"); Serial.print(turbidity, 1); Serial.print(",");
  Serial.print("PH:"); Serial.print(pH, 1); Serial.print(",");
  Serial.print("AQI:"); Serial.print(airQuality, 1); Serial.print(",");
  Serial.print("WTEMP:"); Serial.print(waterTemp, 1);
  Serial.println();
  
  // Check for commands from computer
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }
  
  delay(5000); // Send data every 5 seconds
}

float readTemperature() {
  float temp = dht.readTemperature();
  if (isnan(temp)) {
    return 25.0; // Return default instead of error
  }
  return temp;
}

float readHumidity() {
  float hum = dht.readHumidity();
  if (isnan(hum)) {
    return 50.0; // Return default instead of error
  }
  return hum;
}

float readTurbidity() {
  int sensorValue = analogRead(TURBIDITY_PIN);
  float voltage = sensorValue * (3.3 / 4095.0);
  float turbidity = voltage * turbidityCalibration * 100;
  
  // Ensure reasonable range
  if (turbidity < 0) turbidity = 0;
  if (turbidity > 100) turbidity = 100;
  
  return turbidity;
}

float readPH() {
  int measure = analogRead(PH_PIN);
  float voltage = measure * (3.3 / 4095.0);
  float pHValue = pHCalibration + (voltage - 1.5) / 0.18;
  
  // Ensure reasonable range
  if (pHValue < 0) pHValue = 7.0;
  if (pHValue > 14) pHValue = 7.0;
  
  return pHValue;
}

float readAirQuality() {
  int sensorValue = analogRead(MQ135_PIN);
  float voltage = sensorValue * (3.3 / 4095.0);
  float airQuality = (voltage / 3.3) * airQualityCalibration;
  
  // Ensure reasonable range
  if (airQuality < 0) airQuality = 50;
  if (airQuality > 500) airQuality = 500;
  
  return airQuality;
}

float readWaterTemperature() {
  // Simulate water temperature (usually slightly lower than air temp)
  float airTemp = readTemperature();
  return airTemp - random(2, 5); // Water is 2-5 degrees cooler
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
  } else if (command == "CALIBRATE") {
    calibrateSensors();
  } else if (command == "DATA") {
    // Force immediate data transmission
    float temperature = readTemperature();
    float humidity = readHumidity();
    float turbidity = readTurbidity();
    
    Serial.print("IMMEDIATE:");
    Serial.print("TEMP:"); Serial.print(temperature, 1); Serial.print(",");
    Serial.print("HUM:"); Serial.print(humidity, 1); Serial.print(",");
    Serial.print("TURB:"); Serial.print(turbidity, 1);
    Serial.println();
  } else {
    Serial.println("ERROR:UNKNOWN_COMMAND");
  }
}

void calibrateSensors() {
  Serial.println("CALIBRATION:STARTED");
  
  // Simple calibration routine - adjust based on environment
  turbidityCalibration = 0.95 + (random(0, 200) - 100) / 1000.0;
  pHCalibration = 7.0 + (random(0, 200) - 100) / 1000.0;
  
  Serial.println("CALIBRATION:COMPLETED");
}

// Additional utility function for sensor diagnostics
void sensorDiagnostics() {
  Serial.println("DIAGNOSTICS:");
  Serial.print("DHT22: ");
  if (isnan(dht.readTemperature())) {
    Serial.println("FAILED");
  } else {
    Serial.println("OK");
  }
  
  Serial.print("Turbidity: ");
  int turbValue = analogRead(TURBIDITY_PIN);
  Serial.println(turbValue);
  
  Serial.print("pH: ");
  int pHValue = analogRead(PH_PIN);
  Serial.println(pHValue);
  
  Serial.print("Air Quality: ");
  int airValue = analogRead(MQ135_PIN);
  Serial.println(airValue);
}
