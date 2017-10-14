def write(filename, data):
    f = open(filename, 'w')
    for line in data:
        f.write(",".join(line)+"\n")
