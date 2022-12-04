import os
import time


personRunning = "rahil"

def run(layouts):
    for agent in agents:
        for layout in layouts:
            if layout[:3]=="big":
                timeout = 7*60
            elif layout[:6]=="medium":
                timeout = 3*60
            else:
                timeout = 1*60
            print("\n\nWhich Layout "+layout+" and Agent "+agent)
            if agent!="MonteCarloTreeSearchAgent":
                fileToRun = "python pacman.py -q -p "+ agent +" -l "+ layout +" -n 3 -a depth=3 --frameTime 0 -c --timeout "+str(timeout) 
            else:
                fileToRun = "python pacman.py -q -p "+ agent +" -l "+ layout +" -n 3 --frameTime 0 -c --timeout "+str(timeout)

            os.system(fileToRun)

agents = ["MinimaxAgent", "AlphaBetaAgent", "ExpectimaxAgent", "MonteCarloTreeSearchAgent"]
# layouts = ["mediumClassic", "trickyClassic", "openClassic_R","smallClassic", "powerClassic","trickyClassic"]
layouts = []
layouts_rahil = []
layouts_nikhil = []
layouts_austin = []

count=0
files =[ "small", "medium", "big"]
for i in range(3):
    fileName = files[i]
    for j in range(1,31):
        layouts.append(fileName+str(j))

layouts_rahil = layouts[0:10] + layouts[30:40] + layouts[60:70]
layouts_nikhil = layouts[10:20] + layouts[40:50] + layouts[70:80]
layouts_austin = layouts[20:30] + layouts[50:60] + layouts[80:90]

if personRunning.lower()=="rahil":
    run(layouts_rahil)
if personRunning.lower()=="nikhil":
    run(layouts_nikhil)
if personRunning.lower()=="austin":
    run(layouts_austin)

# time.sleep(220)

# p = subprocess.run("python ./pacman.py")
# mediumClassic
# trickyClassic
# openClassic_R
# smallClassic
# powerClassic / testClassic
# mctsModelClassic