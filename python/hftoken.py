def getToken():
  with open("../token") as tfile:
    token = tfile.readline().strip()
    return token