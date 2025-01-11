import subprocess
requsites_file = "requisits.txt"
requisites = open(requsites_file, "r")
for command in requisites:
    s = subprocess.getstatusoutput(command)
    if s[0] == 0:
        print(s[1])
    else:
        print('Custom Error {}'.format(s[1]))