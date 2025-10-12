import random
from datetime import datetime

class ArduinoManager:
    def __init__(self):
        self.is_connected = False
        self.current_data = {}
    
    def discover_ports(self):
        # Simulate port discovery
        return []  # Return empty list for simulation
    
    def connect(self, port, baudrate=115200):
        self.is_connected = True
        return True, f"Connected to Arduino on {port}"
    
    def disconnect(self):
        self.is_connected = False
    
    def get_current_data(self):
        return self.generate_simulated_data("Default")
    
    def generate_simulated_data(self, location):
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
