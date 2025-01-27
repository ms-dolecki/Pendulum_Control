import subprocess
import time
sync_file = "sync_with_lambda.txt"
sync = open(sync_file, "r")
for command in sync:
    while True:
        s = subprocess.getstatusoutput(command)
        if s[0] == 0:
            print(s[1])
        else:
            print('Custom Error {}'.format(s[1]))
        time.sleep(1)