import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime
import random

class ArduinoManager:
    def __init__(self):
        self.serial_conn = None
        self.is_connected = False
        self.current_data = {}
        self.data_buffer = []
        self.read_thread = None
        self.should_read = False
        self.last_read_time = None
        self.auto_detect_thread = None
        self.should_auto_detect = False
        self.last_scan_time = None
        
    def start_auto_detection(self):
        """Start automatic Arduino detection in background"""
        if not self.should_auto_detect:
            self.should_auto_detect = True
            self.auto_detect_thread = threading.Thread(target=self._auto_detect_loop)
            self.auto_detect_thread.daemon = True
            self.auto_detect_thread.start()
            print("Auto-detection started")
    
    def stop_auto_detection(self):
        """Stop automatic Arduino detection"""
        self.should_auto_detect = False
        if self.auto_detect_thread:
            self.auto_detect_thread.join(timeout=2.0)
        print("Auto-detection stopped")
    
    def _auto_detect_loop(self):
        """Background thread for automatic Arduino detection"""
        while self.should_auto_detect:
            try:
                # Only scan if not already connected and not scanned recently
                if not self.is_connected and (
                    self.last_scan_time is None or 
                    (datetime.now() - self.last_scan_time).total_seconds() > 5
                ):
                    self.last_scan_time = datetime.now()
                    ports = self.discover_ports()
                    
                    if ports:
                        print(f"Auto-detected Arduino on {ports[0]}, attempting connection...")
                        success, message = self.connect(ports[0])
                        if success:
                            print(f"Auto-connected to Arduino: {message}")
                        else:
                            print(f"Auto-connection failed: {message}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Auto-detection error: {e}")
                time.sleep(5)  # Longer delay on error
    
    def discover_ports(self):
        """Discover available Arduino ports with better detection"""
        arduino_ports = []
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            port_info = f"{port.device} - {port.description}"
            
            # Common Arduino/ESP32 identifiers
            arduino_indicators = [
                'arduino', 'esp32', 'usb serial', 'ch340', 'cp210', 
                'silicon labs', 'usb2.0-serial'
            ]
            
            if any(keyword in port.description.lower() for keyword in arduino_indicators):
                arduino_ports.append(port.device)
            elif port.vid is not None and port.pid is not None:
                # Common Arduino/ESP32 USB VID/PID pairs
                arduino_vid_pid = [
                    (0x2341, 0x0043), (0x2341, 0x0001),  # Arduino Uno
                    (0x2A03, 0x0043), (0x2341, 0x0243),  # Arduino Leonardo
                    (0x2341, 0x0042), (0x2341, 0x0010),  # Arduino Mega
                    (0x10C4, 0xEA60), (0x1A86, 0x7523),  # ESP32
                    (0x303A, 0x1001), (0x303A, 0x0001),  # ESP32-S3
                    (0x1A86, 0x55D4), (0x0403, 0x6001)   # Common USB-Serial chips
                ]
                if (port.vid, port.pid) in arduino_vid_pid:
                    arduino_ports.append(port.device)
        
        # Remove duplicates and return
        return list(set(arduino_ports))
    
    def connect(self, port, baudrate=115200):
        """Connect to Arduino with better error handling"""
        try:
            # Close existing connection if any
            if self.serial_conn and self.serial_conn.is_open:
                self.disconnect()
            
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                write_timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for Arduino to reset and initialize
            time.sleep(2)
            
            # Clear any existing data in buffer
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Test connection by waiting for ready message
            start_time = time.time()
            while time.time() - start_time < 5:  # Wait up to 5 seconds
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    print(f"Arduino response: {line}")  # Debug
                    
                    if 'SAEAS_ARDUINO_READY' in line or 'SENSOR:INIT:COMPLETE' in line:
                        self.is_connected = True
                        self.start_reading()
                        return True, f"Connected to Arduino on {port}"
                time.sleep(0.1)
            
            # If no ready message, try to send ping
            self.serial_conn.write(b'PING\n')
            time.sleep(1)
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if 'PONG' in response:
                    self.is_connected = True
                    self.start_reading()
                    return True, f"Connected to Arduino on {port} (PING successful)"
            
            # If we get here, assume connection is good anyway
            self.is_connected = True
            self.start_reading()
            return True, f"Connected to Arduino on {port} (assumed ready)"
            
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.stop_reading()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        self.current_data = {}
        print("Arduino disconnected")
    
    def start_reading(self):
        """Start background data reading"""
        if self.is_connected and not self.should_read:
            self.should_read = True
            self.read_thread = threading.Thread(target=self._read_data)
            self.read_thread.daemon = True
            self.read_thread.start()
            print("Data reading started")
    
    def stop_reading(self):
        """Stop background data reading"""
        self.should_read = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        print("Data reading stopped")
    
    def _read_data(self):
        """Background data reading thread"""
        while self.should_read and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    self._process_data(line)
                    self.last_read_time = datetime.now()
                time.sleep(0.1)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(1)
    
    def _process_data(self, data_line):
        """Process incoming Arduino data"""
        try:
            # Expected format: "SENSOR:TEMP:25.6,HUM:65.2,TURB:12.3,PH:7.2,AQI:45,WTEMP:22.1"
            if data_line.startswith('SENSOR:'):
                sensor_data = {}
                parts = data_line.split(':')[1].split(',')
                
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':')
                        try:
                            sensor_data[key.lower()] = float(value)
                        except ValueError:
                            sensor_data[key.lower()] = value
                
                # Add timestamp and store data
                sensor_data['timestamp'] = datetime.now()
                sensor_data['data_source'] = 'arduino'
                self.current_data = sensor_data
                
                # Add to buffer for historical data
                self.data_buffer.append(sensor_data.copy())
                if len(self.data_buffer) > 1000:
                    self.data_buffer.pop(0)
                    
                print(f"Received Arduino data: {sensor_data}")
                
        except Exception as e:
            print(f"Data processing error: {e}")
    
    def get_current_data(self):
        """Get current sensor data with fallback to simulated data"""
        if self.is_connected and self.current_data:
            # Check if data is recent (within last 30 seconds)
            if self.last_read_time and (datetime.now() - self.last_read_time).total_seconds() < 30:
                return self.current_data.copy()
        
        # Fallback to simulated data if no recent Arduino data
        return self.generate_simulated_data("Default")
    
    def get_historical_data(self):
        """Get historical sensor data"""
        return self.data_buffer.copy()
    
    def send_command(self, command):
        """Send command to Arduino"""
        if self.is_connected and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                return True
            except Exception as e:
                print(f"Command send error: {e}")
                return False
        return False
    
    def generate_simulated_data(self, location):
        """Generate realistic simulated sensor data"""
        base_data = {
            'temp': random.uniform(20, 35),
            'hum': random.uniform(30, 90),
            'turb': random.uniform(0, 100),
            'ph': random.uniform(6.5, 8.0),
            'aqi': random.uniform(20, 150),
            'wtemp': random.uniform(18, 30),
            'timestamp': datetime.now(),
            'data_source': 'simulated'
        }
        
        # Location-specific variations
        if 'Urban' in location:
            base_data['temp'] += random.uniform(2, 5)
            base_data['aqi'] += random.uniform(20, 50)
        elif 'Coastal' in location:
            base_data['hum'] += random.uniform(10, 20)
            base_data['turb'] += random.uniform(10, 30)
        
        # Ensure values are within reasonable bounds
        base_data['temp'] = max(15, min(40, base_data['temp']))
        base_data['hum'] = max(20, min(95, base_data['hum']))
        base_data['turb'] = max(0, min(100, base_data['turb']))
        base_data['ph'] = max(6.0, min(8.5, base_data['ph']))
        base_data['aqi'] = max(0, min(300, base_data['aqi']))
        
        return base_data
    
    def get_connection_status(self):
        """Get detailed connection status"""
        status = {
            'connected': self.is_connected,
            'data_source': 'arduino' if self.is_connected and self.current_data else 'simulated',
            'last_read_time': self.last_read_time,
            'data_age_seconds': None,
            'auto_detection_active': self.should_auto_detect
        }
        
        if self.last_read_time:
            status['data_age_seconds'] = (datetime.now() - self.last_read_time).total_seconds()
        
        return status
    
    def force_rescan(self):
        """Force immediate rescan for Arduino devices"""
        self.last_scan_time = None  # Reset scan timer
        ports = self.discover_ports()
        return ports
