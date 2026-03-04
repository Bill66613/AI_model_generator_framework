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
        self.inference_buffer = deque(maxlen=500)  # Store inference results
        self.latest_prediction = None  # Most recent prediction string
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
        """Parse incoming serial data line.

        Expected CSV format from generated sketches:
            aX,aY,aZ,gX,gY,gZ              (sensor data only)
            aX,aY,aZ,gX,gY,gZ,activity     (sensor data + inference result)

        Lines starting with '#' or containing non-numeric first fields are
        treated as info/debug messages and logged to the debug console.
        """
        try:
            # Skip empty lines
            if not line:
                return

            # Skip comment / info lines (e.g. "HAR Model Ready!", "# ...")
            if line.startswith('#'):
                self._log(f"[INFO] {line}")
                return

            parts = line.split(',')
            n_cols = len(self.sensor_cols)

            # Need at least n_cols comma-separated parts with numeric sensor values
            if len(parts) < n_cols:
                # Could be a human-readable status message — log it instead of erroring
                self._log(f"[DEVICE] {line[:100]}")
                return

            # Try parsing the first n_cols parts as floats
            try:
                sensor_values = [float(parts[i]) for i in range(n_cols)]
            except ValueError:
                # First fields aren't numeric — treat as info message
                self._log(f"[DEVICE] {line[:100]}")
                return

            current_time = time.time() - self.start_time if self.start_time else 0

            self.data_buffer['time'].append(current_time)
            for i, col in enumerate(self.sensor_cols):
                self.data_buffer[col].append(sensor_values[i])

            # Check for optional inference result after sensor columns
            prediction = None
            if len(parts) > n_cols:
                extra = parts[n_cols].strip()
                if extra:
                    prediction = extra
                    self.latest_prediction = prediction
                    self.inference_buffer.append({
                        'time': current_time,
                        'prediction': prediction
                    })

            # Log first valid sample
            if len(self.data_buffer['time']) == 1:
                suffix = f" → {prediction}" if prediction else ""
                self._log(f"✓ First data sample received: {line[:60]}{suffix}")

        except Exception as e:
            error_msg = f"Parse error for '{line[:80]}': {e}"
            self._log(f"✗ {error_msg}")
            print(error_msg)

    def get_data(self):
        """Get current buffered data as dict of lists."""
        result = {'time': list(self.data_buffer['time'])}
        for col in self.sensor_cols:
            result[col] = list(self.data_buffer[col])
        return result

    def get_inference_results(self):
        """Get buffered inference results as list of dicts with 'time' and 'prediction'."""
        return list(self.inference_buffer)

    def get_latest_prediction(self):
        """Get the most recent inference prediction string, or None."""
        return self.latest_prediction

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
        self.inference_buffer.clear()
        self.latest_prediction = None
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
