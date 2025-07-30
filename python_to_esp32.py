import serial

port = serial.Serial('COM1', 115200, timeout=1)

def write_ser(cmd):
    cmd = cmd + '\n'
    port.write(cmd.encode())

while (1):
    cmd = input()
    if (cmd):
        write_ser(cmd)