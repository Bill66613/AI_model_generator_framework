import serial
import serial.tools.list_ports
import threading
import time
import numpy as np
from collections import deque

from config.config import SENSOR_COLUMNS


class DeviceReader:
    def __init__(self, sensor_cols=None):
        self.serial_conn = None
        self.is_connected = False
        self.is_running = False
        self.sensor_cols = sensor_cols or SENSOR_COLUMNS
        self.data_buffer = {'time': deque(maxlen=500)}
        for col in self.sensor_cols:
            self.data_buffer[col] = deque(maxlen=500)
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

            # Expected format: comma-separated values matching sensor_cols order
            parts = line.split(',')
            n_cols = len(self.sensor_cols)

            if len(parts) >= n_cols:
                current_time = time.time() - self.start_time if self.start_time else 0

                self.data_buffer['time'].append(current_time)
                for i, col in enumerate(self.sensor_cols):
                    self.data_buffer[col].append(float(parts[i]))

                # Log first valid sample
                if len(self.data_buffer['time']) == 1:
                    self._log(f"✓ First data sample received: {line[:50]}...")
            else:
                error_msg = f"Invalid format (expected {n_cols} values, got {len(parts)}): {line[:80]}"
                self._log(f"✗ {error_msg}")
                print(error_msg)

        except (ValueError, IndexError) as e:
            error_msg = f"Parse error for '{line[:80]}': {e}"
            self._log(f"✗ {error_msg}")
            print(error_msg)

    def get_data(self):
        """Get current buffered data as dict of lists."""
        result = {'time': list(self.data_buffer['time'])}
        for col in self.sensor_cols:
            result[col] = list(self.data_buffer[col])
        return result

    def get_latest_window(self, window_size=150):
        """Get latest N samples for inference."""
        data = self.get_data()
        if len(data['time']) < window_size:
            return None

        return {col: data[col][-window_size:] for col in self.sensor_cols}

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
