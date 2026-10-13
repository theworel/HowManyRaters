import sys
# for line in sys.stdin:
#   print(line, file=sys.stderr)
for line in sys.stdin:
  # print(line, file=sys.stderr)
  line = line.strip()
  if(line == ""):
    print("")
  else:

    print(line.split("\t")[0], end=" ")