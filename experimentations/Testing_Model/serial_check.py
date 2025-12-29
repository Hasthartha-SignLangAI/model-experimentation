import serial

PORT = "/dev/tty.usbserial-0001"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
print("Connected. Printing 30 lines...\n")

for i in range(30):
    line = ser.readline().decode(errors="ignore").strip()
    print(line)

ser.close()
