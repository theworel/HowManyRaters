import subprocess
import sys

MORPHO_RUN_ANALYZE_SCRIPT = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/bash/short_bad_md_analyze_en.sh"
MORPHO_RUN_ANALYZE_ARGS = ""
MORPHO_RUN_GENERATE_SCRIPT = "/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/bash/short_bad_md_generate_en.sh"
MORPHO_RUN_GENERATE_ARGS = ""
print("start")
def runScript(scriptfile, args, input_string): 
  p = subprocess.Popen(scriptfile, shell=True, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
  output,_ = p.communicate(input = input_string.encode())
  decoded = output.decode()

  return (decoded)

def analyzeLine(line):
  l = runScript(MORPHO_RUN_ANALYZE_SCRIPT, MORPHO_RUN_ANALYZE_ARGS, line)
  return l

def generateLine(line):
  l = runScript(MORPHO_RUN_GENERATE_SCRIPT, MORPHO_RUN_GENERATE_ARGS, line)
  return l

def prepForGenerate(line):
  l = ""
  print("----")
  for token in line.split("\n"):
    if len(token) > 0:
      brokenUp = token.split("\t")
      print(brokenUp)
      line += brokenUp[1] + "\t" + brokenUp[2] + "\n"
  print("----")
  return line, []

def fixLine(line):
  result = ""
  for l in line.split("\n"):
    l= l.strip()
    if(len(l) > 0):
      result += l.split("\t")[0] + " "
  return result

for line in sys.stdin:
  print(line)
  l = analyzeLine(line)
  print(l)
  l, extras = prepForGenerate(l)
  print(l)
  print("generate")
  l = generateLine(l)
  print(l)
  print("fixline")
  l = fixLine(l)
  print(l)