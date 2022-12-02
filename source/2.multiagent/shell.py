import os
import time

agents = ["MinimaxAgent", "AlphaBetaAgent", "ExpectimaxAgent", "MonteCarloTreeSearchAgent"]
layouts = ["mediumClassic", "trickyClassic", "openClassic_R","smallClassic", "powerClassic","trickyClassic"]
for agent in agents:
    for layout in layouts:
        if agent=="MinimaxAgent" and (layout=="smallClassic" or layout=="testClassic"):
            continue
        print("\n\nWhich Layout "+layout+" and Agent "+agent)
        if agent!="MonteCarloTreeSearchAgent":
            fileToRun = "python pacman.py -p "+ agent +" -l "+ layout +" -n 100 -a depth=3 --frameTime 0 -q -c --timeout 120" 
        else:
            fileToRun = "python pacman.py -p "+ agent +" -l "+ layout +" -n 100 --frameTime 0 -q -c --timeout 120" 

        os.system(fileToRun)
        # time.sleep(220)

# p = subprocess.run("python ./pacman.py")
# mediumClassic
# trickyClassic
# openClassic_R
# smallClassic
# powerClassic / testClassic
# mctsModelClassic