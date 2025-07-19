import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    if s.connect_ex(('localhost', int(sys.argv[1]))) != 0:  # 双重检测机制
        s.bind(('', int(sys.argv[1])))
        sys.exit(0)
    sys.exit(1)
except Exception as e:
    sys.exit(1)
finally:
    s.close()