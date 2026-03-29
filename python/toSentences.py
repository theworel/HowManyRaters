import sys
for line in sys.stdin:
  line = line.strip()
  if(line == ""):
    print("")
  else:

    print(line.split("\t")[0], end=" ")