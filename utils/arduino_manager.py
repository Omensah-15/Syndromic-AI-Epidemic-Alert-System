import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime
import random
import json

class ArduinoManager:
    """Arduino ESP32 connection and data management"""
    
    def __init__(self):
        self.serial_conn = None
        self.is_connected = False
        self.current_data = {}
        self.data_buffer = []
        self.read_thread = None
        self.should_read = False
    
    def discover_ports(self):
        """Discover available Arduino ports"""
        arduino_ports = []
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            if any(keyword in port.description.upper() for keyword in 
                  ['ARDUINO', 'ESP32', 'USB SERIAL', 'CH340', 'CP210']):
                arduino_ports.append(port.device)
        
        return arduino_ports
    
    def connect(self, port, baudrate=115200):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            
            self.is_connected = True
            self.start_reading()
            
            return True, f"Connected to Arduino on {port}"
            
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.stop_reading()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
    
    def start_reading(self):
        """Start background data reading"""
        if self.is_connected and not self.should_read:
            self.should_read = True
            self.read_thread = threading.Thread(target=self._read_data)
            self.read_thread.daemon = True
            self.read_thread.start()
    
    def stop_reading(self):
        """Stop background data reading"""
        self.should_read = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
    
    def _read_data(self):
        """Background data reading thread"""
        while self.should_read and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode().strip()
                    self._process_data(line)
            except Exception as e:
                print(f"Serial read error: {e}")
            time.sleep(0.1)
    
    def _process_data(self, data_line):
        """Process incoming Arduino data"""
        try:
            if data_line.startswith('SENSOR:'):
                # Expected format: SENSOR:TEMP:25.6,HUM:65.2,TURB:12.3
                sensor_data = {}
                parts = data_line.split(':')[1].split(',')
                
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':')
                        try:
                            sensor_data[key.lower()] = float(value)
                        except ValueError:
                            sensor_data[key.lower()] = value
                
                sensor_data['timestamp'] = datetime.now()
                self.current_data = sensor_data
                self.data_buffer.append(sensor_data.copy())
                
                # Keep buffer manageable
                if len(self.data_buffer) > 1000:
                    self.data_buffer.pop(0)
                    
        except Exception as e:
            print(f"Data processing error: {e}")
    
    def get_current_data(self):
        """Get current sensor data"""
        return self.current_data.copy()
    
    def get_historical_data(self):
        """Get historical sensor data"""
        return self.data_buffer.copy()
    
    def generate_simulated_data(self, location):
        """Generate simulated sensor data for demo"""
        base_data = {
            'temperature': random.uniform(20, 35),
            'humidity': random.uniform(30, 90),
            'turbidity': random.uniform(0, 100),
            'air_quality': random.uniform(20, 150),
            'water_ph': random.uniform(6.5, 8.0),
            'timestamp': datetime.now()
        }
        
        # Location-specific variations
        if 'Urban' in location:
            base_data['temperature'] += random.uniform(2, 5)
            base_data['air_quality'] += random.uniform(20, 50)
        elif 'Coastal' in location:
            base_data['humidity'] += random.uniform(10, 20)
            base_data['turbidity'] += random.uniform(10, 30)
        
        return base_data
