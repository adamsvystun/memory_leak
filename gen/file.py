def write(filename, data):
    f = open(filename, 'w')
    for line in data:
        line_str = ""
        for par in line:
            line_str += str(par) + ","
        f.write(line_str+"\n")
