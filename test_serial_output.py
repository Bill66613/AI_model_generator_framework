"""
Simple script to test what your serial device is actually sending.
This helps debug connection issues with the Device Testing tab.

Usage: 
1. Make sure your device is connected
2. Update COM_PORT below to match your device
3. Run: python test_serial_output.py
4. Watch the output for 10 seconds
"""

import serial
import serial.tools.list_ports
import time

# Configuration
# Change this to your actual port (e.g., 'COM3', 'COM5', '/dev/ttyUSB0')
COM_PORT = 'COM3'
BAUD_RATE = 115200
DURATION = 10  # seconds


def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    print("\n=== Available Serial Ports ===")
    if not ports:
        print("No serial ports found!")
        return

    for port in ports:
        print(f"  {port.device} - {port.description}")
    print()


def test_serial_output(port, baudrate, duration):
    """Read and display serial output"""
    try:
        print(f"\n=== Connecting to {port} @ {baudrate} baud ===")
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino reset

        print(f"Connected! Reading data for {duration} seconds...\n")
        print("=== Raw Serial Output ===")

        start_time = time.time()
        line_count = 0
        valid_count = 0
        error_count = 0

        while time.time() - start_time < duration:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    line_count += 1

                    # Print the raw line
                    print(f"[{line_count:3d}] {line}")

                    # Try to parse it
                    if line.startswith('#'):
                        print("      ^ Comment line (will be skipped)")
                    elif not line:
                        print("      ^ Empty line (will be skipped)")
                    else:
                        parts = line.split(',')
                        if len(parts) >= 6:
                            try:
                                # Try to convert to floats
                                values = [float(p) for p in parts[:6]]
                                print(
                                    f"      ✓ Valid: aX={values[0]:.2f}, aY={values[1]:.2f}, aZ={values[2]:.2f}, gX={values[3]:.2f}, gY={values[4]:.2f}, gZ={values[5]:.2f}")
                                valid_count += 1
                            except ValueError as e:
                                print(f"      ✗ Parse error: {e}")
                                error_count += 1
                        else:
                            print(
                                f"      ✗ Wrong format (expected 6 values, got {len(parts)})")
                            error_count += 1

                except UnicodeDecodeError as e:
                    print(f"      ✗ Unicode decode error: {e}")
                    error_count += 1

        ser.close()

        print(f"\n=== Summary ===")
        print(f"Total lines: {line_count}")
        print(f"Valid data lines: {valid_count}")
        print(f"Errors: {error_count}")
        print(f"Data rate: {valid_count / duration:.1f} samples/second")

        if valid_count == 0:
            print("\n⚠️  WARNING: No valid data received!")
            print("   Possible issues:")
            print("   1. Arduino sketch not uploaded or not running")
            print("   2. Wrong baud rate (should be 115200)")
            print("   3. Wrong COM port selected")
            print("   4. Data format doesn't match (should be: aX,aY,aZ,gX,gY,gZ)")
        elif valid_count < duration * 50:
            print(
                f"\n⚠️  WARNING: Low data rate ({valid_count / duration:.1f} Hz)")
            print("   Expected ~100 Hz. Check your Arduino sketch.")
        else:
            print("\n✓ Device is working correctly!")
            print(f"  You can now use the Device Testing tab in the app.")

    except serial.SerialException as e:
        print(f"\n❌ Serial connection error: {e}")
        print("   Check that:")
        print("   1. The device is connected")
        print("   2. The COM port is correct")
        print("   3. No other program is using the port")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    list_available_ports()

    print("\n" + "="*50)
    print("Edit COM_PORT variable at the top of this file")
    print("to match your device, then run this script again.")
    print("="*50)

    # Uncomment the line below after setting the correct COM_PORT
    # test_serial_output(COM_PORT, BAUD_RATE, DURATION)
