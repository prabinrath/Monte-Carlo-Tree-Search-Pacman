import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import json

LAYOUTS = ["smallClassic", "testClassic"]

def getData():
    directory = "../runs"
    data_specific_layout = dict()
    for fname in os.listdir(directory):
        _fileName = os.path.join(directory, fname)
        if os.path.isfile(_fileName):
            f = open(_fileName, "r")
            f_Data = json.loads(f.readline())
            fileName = _fileName.split("_")
            layout = fileName[2]
            if layout in LAYOUTS:
                if layout in data_specific_layout:
                    data_specific_layout[layout].append(f_Data)
                else:
                    data_specific_layout[layout] = [f_Data]
            f.close()    
    return data_specific_layout

def analysis(data):
    analysis = {}
    for layout in data:
        analysis[layout] = {}
        for d in data[layout]:
            # print(analysis)
            count, totalScore, totalTime = 0, 0, 0
            for i in range(d['TotalGames']):
                if not d["Crashed"][i]:
                    totalScore += d["Scores"][i]
                    totalTime += d["Time"][i]
                    count+=1
                else:
                    continue
            analysis[layout][d["Pacman_Agent"]] = {
                "meanScore": round((totalScore/count),2), 
                "meanTime": round((totalTime/count),2),
                "winPercent": round((d["#Wins"]/d["TotalGames"]),2)
            }
    return analysis

def plot(data):
    x_axis = np.arange(len(LAYOUTS))
    width = 0.1
    AlphaBetaAgent_score, ExpectimaxAgent_score, MinimaxAgent_score = [], [], []
    AlphaBetaAgent_time, ExpectimaxAgent_time, MinimaxAgent_time = [], [], []
    AlphaBetaAgent_win, ExpectimaxAgent_win, MinimaxAgent_win = [], [], []
    for layout in data:
        for agent in data[layout]:
            if agent=="AlphaBetaAgent":
                AlphaBetaAgent_score.append(data[layout]["AlphaBetaAgent"]["meanScore"])
                AlphaBetaAgent_time.append(data[layout]["AlphaBetaAgent"]["meanTime"])
                AlphaBetaAgent_win.append(data[layout]["AlphaBetaAgent"]["winPercent"])
            elif agent=="ExpectimaxAgent":
                ExpectimaxAgent_score.append(data[layout]["ExpectimaxAgent"]["meanScore"])
                ExpectimaxAgent_time.append(data[layout]["ExpectimaxAgent"]["meanTime"])
                ExpectimaxAgent_win.append(data[layout]["ExpectimaxAgent"]["winPercent"])
            elif agent=="MinimaxAgent":
                MinimaxAgent_score.append(data[layout]["MinimaxAgent"]["meanScore"])
                MinimaxAgent_time.append(data[layout]["MinimaxAgent"]["meanTime"])
                MinimaxAgent_win.append(data[layout]["MinimaxAgent"]["winPercent"])
    # print(AlphaBetaAgent_score)
    # print(ExpectimaxAgent_score)
    # print(MinimaxAgent_score)
    plt.figure("Scores")
    plt.bar(x_axis -width, AlphaBetaAgent_score, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_score, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_score, width, label = 'MinimaxAgent')
    plt.ylabel("Scores")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()


    plt.figure("Time")
    plt.bar(x_axis -width, AlphaBetaAgent_time, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_time, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_time, width, label = 'MinimaxAgent')
    plt.ylabel("Time")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()

    plt.figure("Win Percent")
    plt.bar(x_axis -width, AlphaBetaAgent_win, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_win, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_win, width, label = 'MinimaxAgent')
    plt.ylabel("Win Percent")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    data = getData()
    per_layout_values = analysis(data)
    print(per_layout_values)
    plot(per_layout_values)
    

