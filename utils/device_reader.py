import serial
import serial.tools.list_ports
import threading
import time
import numpy as np
from collections import deque


class DeviceReader:
    def __init__(self):
        self.serial_conn = None
        self.is_connected = False
        self.is_running = False
        self.data_buffer = {
            'time': deque(maxlen=500),
            'aX': deque(maxlen=500),
            'aY': deque(maxlen=500),
            'aZ': deque(maxlen=500),
            'gX': deque(maxlen=500),
            'gY': deque(maxlen=500),
            'gZ': deque(maxlen=500)
        }
        self.debug_log = deque(maxlen=100)  # Store last 100 debug messages
        self.start_time = None
        self.reader_thread = None

    def get_available_ports(self):
        """Get list of available serial ports"""
        ports = serial.tools.list_ports.comports()
        return [{'label': f"{port.device} - {port.description}", 'value': port.device}
                for port in ports]

    def connect(self, port, baudrate=115200):
        """Connect to serial port"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.disconnect()

            self._log(f"Connecting to {port} @ {baudrate} baud...")
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            self.is_connected = True
            self.start_time = time.time()

            # Start reader thread
            self.is_running = True
            self.reader_thread = threading.Thread(
                target=self._read_loop, daemon=True)
            self.reader_thread.start()

            self._log(f"✓ Connected successfully to {port}")
            return True, "Connected successfully"
        except Exception as e:
            self._log(f"✗ Connection failed: {str(e)}")
            return False, f"Connection failed: {str(e)}"

    def disconnect(self):
        """Disconnect from serial port"""
        self._log("Disconnecting...")
        self.is_running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=2)

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

        self.is_connected = False
        self.clear_buffer()
        self._log("✓ Disconnected")

    def _read_loop(self):
        """Background thread to read serial data"""
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:  # Only parse non-empty lines
                        self._parse_line(line)
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.1)
            else:
                time.sleep(0.01)  # Small delay to prevent CPU spinning

    def _parse_line(self, line):
        """Parse incoming serial data line"""
        try:
            # Skip comment lines
            if line.startswith('#'):
                self._log(f"[INFO] {line}")
                return

            if not line:
                return

            # Expected format: aX,aY,aZ,gX,gY,gZ
            parts = line.split(',')

            if len(parts) >= 6:
                current_time = time.time() - self.start_time if self.start_time else 0

                self.data_buffer['time'].append(current_time)
                self.data_buffer['aX'].append(float(parts[0]))
                self.data_buffer['aY'].append(float(parts[1]))
                self.data_buffer['aZ'].append(float(parts[2]))
                self.data_buffer['gX'].append(float(parts[3]))
                self.data_buffer['gY'].append(float(parts[4]))
                self.data_buffer['gZ'].append(float(parts[5]))

                # Log first valid sample
                if len(self.data_buffer['time']) == 1:
                    self._log(f"✓ First data sample received: {line[:50]}...")
            else:
                error_msg = f"Invalid format (expected 6 values, got {len(parts)}): {line[:80]}"
                self._log(f"✗ {error_msg}")
                print(error_msg)

        except (ValueError, IndexError) as e:
            error_msg = f"Parse error for '{line[:80]}': {e}"
            self._log(f"✗ {error_msg}")
            print(error_msg)

    def get_data(self):
        """Get current buffered data"""
        return {
            'time': list(self.data_buffer['time']),
            'aX': list(self.data_buffer['aX']),
            'aY': list(self.data_buffer['aY']),
            'aZ': list(self.data_buffer['aZ']),
            'gX': list(self.data_buffer['gX']),
            'gY': list(self.data_buffer['gY']),
            'gZ': list(self.data_buffer['gZ'])
        }

    def get_latest_window(self, window_size=150):
        """Get latest N samples for inference"""
        data = self.get_data()
        if len(data['time']) < window_size:
            return None

        return {
            'aX': data['aX'][-window_size:],
            'aY': data['aY'][-window_size:],
            'aZ': data['aZ'][-window_size:],
            'gX': data['gX'][-window_size:],
            'gY': data['gY'][-window_size:],
            'gZ': data['gZ'][-window_size:]
        }

    def clear_buffer(self):
        """Clear all buffered data"""
        for key in self.data_buffer:
            self.data_buffer[key].clear()
        self.start_time = time.time()
        self._log("Buffer cleared")

    def _log(self, message):
        """Add message to debug log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")

    def get_debug_log(self):
        """Get debug log as list of strings"""
        return list(self.debug_log)

    def clear_debug_log(self):
        """Clear debug log"""
        self.debug_log.clear()
        self._log("Console cleared")


# Global device reader instance
device_reader = DeviceReader()
