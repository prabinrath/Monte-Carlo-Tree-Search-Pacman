import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import json

ALLLAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic","openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]
LAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic"]
CUSTOMLAYOUTS = ["openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]

def getData():
    directory = "../runs"
    data_specific_layout = dict()
    data_specific_layout_custom = dict()
    for fname in os.listdir(directory):
        _fileName = os.path.join(directory, fname)
        if os.path.isfile(_fileName):
            f = open(_fileName, "r")
            f_Data = json.loads(f.readline())
            fileName = _fileName.split("_")
            layout = fileName[2]
            if layout in ALLLAYOUTS:
                if layout=="openClassicR" or layout=="openClassicN" or layout=="openClassicP" or layout=="openClassicA" or layout=="mctsmodelClassic":
                    if layout in data_specific_layout_custom:
                        data_specific_layout_custom[layout].append(f_Data)
                    else:
                        data_specific_layout_custom[layout] = [f_Data]                    
                else:
                    if layout in data_specific_layout:
                        data_specific_layout[layout].append(f_Data)
                    else:
                        data_specific_layout[layout] = [f_Data]
            
            f.close()    
    return data_specific_layout,data_specific_layout_custom

def analysis(data, dataCustom):
    analysis = {}
    analysisCustom = {}
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

    for layout in dataCustom:
        analysisCustom[layout] = {}
        for d in dataCustom[layout]:
            # print(analysis)
            count, totalScore, totalTime = 0, 0, 0
            for i in range(d['TotalGames']):
                if not d["Crashed"][i]:
                    totalScore += d["Scores"][i]
                    totalTime += d["Time"][i]
                    count+=1
                else:
                    continue
            analysisCustom[layout][d["Pacman_Agent"]] = {
                "meanScore": 0 if count==0 else round((totalScore/count),2), 
                "meanTime": 0 if count==0 else round((totalTime/count),2),
                "winPercent": round((d["#Wins"]/d["TotalGames"]),2)
            }
    return analysis, analysisCustom

def plot(data):
    x_axis = np.arange(len(LAYOUTS))
    width = 0.1
    AlphaBetaAgent_score, ExpectimaxAgent_score, MinimaxAgent_score, MonteCarloTreeSearchAgent_score = [], [], [], []
    AlphaBetaAgent_time, ExpectimaxAgent_time, MinimaxAgent_time,MonteCarloTreeSearchAgent_time = [], [], [], []
    AlphaBetaAgent_win, ExpectimaxAgent_win, MinimaxAgent_win,MonteCarloTreeSearchAgent_win = [], [], [], []
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
            elif agent=="MonteCarloTreeSearchAgent":
                MonteCarloTreeSearchAgent_score.append(data[layout]["MonteCarloTreeSearchAgent"]["meanScore"])
                MonteCarloTreeSearchAgent_time.append(data[layout]["MonteCarloTreeSearchAgent"]["meanTime"])
                MonteCarloTreeSearchAgent_win.append(data[layout]["MonteCarloTreeSearchAgent"]["winPercent"])

    # print(ExpectimaxAgent_score)
    # print(MinimaxAgent_score)
    plt.figure("Scores")
    plt.bar(x_axis -width, AlphaBetaAgent_score, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_score, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_score, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_score, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Scores")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()
    plt.savefig('score_result.png')


    plt.figure("Time")
    plt.bar(x_axis -width, AlphaBetaAgent_time, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_time, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_time, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_time, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Time")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()
    plt.savefig('time_result.png')

    plt.figure("Win Percent")
    plt.bar(x_axis -width, AlphaBetaAgent_win, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_win, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_win, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_win, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Win Percent")
    plt.xticks(x_axis,LAYOUTS)
    plt.legend()
    plt.savefig('win_result.png')
    
    plt.show()

def plotCUSTOM(data):
    x_axis = np.arange(len(CUSTOMLAYOUTS))
    width = 0.1
    AlphaBetaAgent_score, ExpectimaxAgent_score, MinimaxAgent_score, MonteCarloTreeSearchAgent_score = [], [], [], []
    AlphaBetaAgent_time, ExpectimaxAgent_time, MinimaxAgent_time,MonteCarloTreeSearchAgent_time = [], [], [], []
    AlphaBetaAgent_win, ExpectimaxAgent_win, MinimaxAgent_win,MonteCarloTreeSearchAgent_win = [], [], [], []
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
            elif agent=="MonteCarloTreeSearchAgent":
                MonteCarloTreeSearchAgent_score.append(data[layout]["MonteCarloTreeSearchAgent"]["meanScore"])
                MonteCarloTreeSearchAgent_time.append(data[layout]["MonteCarloTreeSearchAgent"]["meanTime"])
                MonteCarloTreeSearchAgent_win.append(data[layout]["MonteCarloTreeSearchAgent"]["winPercent"])

    # print(ExpectimaxAgent_score)
    # print(MinimaxAgent_score)
    plt.figure("Scores")
    plt.bar(x_axis -width, AlphaBetaAgent_score, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_score, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_score, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_score, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Scores")
    plt.xticks(x_axis,CUSTOMLAYOUTS)
    plt.legend()
    plt.savefig('score_result_custom.png')


    plt.figure("Time")
    plt.bar(x_axis -width, AlphaBetaAgent_time, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_time, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_time, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_time, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Time")
    plt.xticks(x_axis,CUSTOMLAYOUTS)
    plt.legend()
    plt.savefig('time_result_custom.png')

    plt.figure("Win Percent")
    plt.bar(x_axis -width, AlphaBetaAgent_win, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_win, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_win, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_win, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Win Percent")
    plt.xticks(x_axis,CUSTOMLAYOUTS)
    plt.legend()
    plt.savefig('win_result_custom.png')
    
    plt.show()

if __name__ == '__main__':
    data, dataCustom = getData()
    per_layout_values, per_layout_values_custom = analysis(data, dataCustom)
    plot(per_layout_values)
    plotCUSTOM(per_layout_values_custom)
    

