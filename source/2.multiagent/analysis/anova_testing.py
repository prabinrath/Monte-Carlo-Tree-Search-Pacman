import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import json

# ALLLAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic","openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]
# LAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic"]
# CUSTOMLAYOUTS = ["openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]

LAYOUT_CLASS = ["SMALL", "MEDIUM", "BIG"]
COUNT=30 #Layouts per Class {small, medium, big}
LAYOUTS_NEW = []
LAYOUT_NAME =[ "small", "medium", "big"]
for i in range(3):
    for j in range(1,31):
        LAYOUTS_NEW.append(LAYOUT_NAME[i]+str(j))

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

def plotNEW(data):
    x_axis = np.arange(len(LAYOUT_CLASS))
    width = 0.1
    AlphaBetaAgent_score, ExpectimaxAgent_score, MinimaxAgent_score, MonteCarloTreeSearchAgent_score = [], [], [], []
    AlphaBetaAgent_time, ExpectimaxAgent_time, MinimaxAgent_time,MonteCarloTreeSearchAgent_time = [], [], [], []
    AlphaBetaAgent_win, ExpectimaxAgent_win, MinimaxAgent_win,MonteCarloTreeSearchAgent_win = [], [], [], []
    for layout in data:
        for agent in data[layout]:
            if agent=="AlphaBetaAgent":
                AlphaBetaAgent_score.append(data[layout]["AlphaBetaAgent"]["meanScore"]/COUNT)
                AlphaBetaAgent_time.append(data[layout]["AlphaBetaAgent"]["meanTime"]/COUNT)
                AlphaBetaAgent_win.append(data[layout]["AlphaBetaAgent"]["winPercent"]/COUNT)
            elif agent=="ExpectimaxAgent":
                ExpectimaxAgent_score.append(data[layout]["ExpectimaxAgent"]["meanScore"]/COUNT)
                ExpectimaxAgent_time.append(data[layout]["ExpectimaxAgent"]["meanTime"]/COUNT)
                ExpectimaxAgent_win.append(data[layout]["ExpectimaxAgent"]["winPercent"]/COUNT)
            elif agent=="MinimaxAgent":
                MinimaxAgent_score.append(data[layout]["MinimaxAgent"]["meanScore"]/COUNT)
                MinimaxAgent_time.append(data[layout]["MinimaxAgent"]["meanTime"]/COUNT)
                MinimaxAgent_win.append(data[layout]["MinimaxAgent"]["winPercent"]/COUNT)
            elif agent=="MonteCarloTreeSearchAgent":
                MonteCarloTreeSearchAgent_score.append(data[layout]["MonteCarloTreeSearchAgent"]["meanScore"]/COUNT)
                MonteCarloTreeSearchAgent_time.append(data[layout]["MonteCarloTreeSearchAgent"]["meanTime"]/COUNT)
                MonteCarloTreeSearchAgent_win.append(data[layout]["MonteCarloTreeSearchAgent"]["winPercent"]/COUNT)

    # print(ExpectimaxAgent_score)
    # print(MinimaxAgent_score)
    plt.figure("Scores")
    plt.bar(x_axis -width, AlphaBetaAgent_score, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_score, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_score, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_score, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Scores")
    plt.xticks(x_axis,LAYOUT_CLASS)
    plt.legend()
    plt.savefig('NEW_SCORES.png')
    plt.show()

    plt.figure("Time")
    plt.bar(x_axis -width, AlphaBetaAgent_time, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_time, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_time, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_time, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Time")
    plt.xticks(x_axis,LAYOUT_CLASS)
    plt.legend()
    plt.savefig('NEW_time.png')

    plt.figure("Win Percent")
    plt.bar(x_axis -width, AlphaBetaAgent_win, width, label = 'AlphaBetaAgent')
    plt.bar(x_axis, ExpectimaxAgent_win, width, label = 'ExpectimaxAgent')
    plt.bar(x_axis +width, MinimaxAgent_win, width, label = 'MinimaxAgent')
    plt.bar(x_axis +width*2, MonteCarloTreeSearchAgent_win, width, label = 'MonteCarloTreeSearchAgent')
    plt.ylabel("Win Percent")
    plt.xticks(x_axis,LAYOUT_CLASS)
    plt.legend()
    plt.savefig('NEW_win.png')
    
    plt.show()

def getData():
    directory = "../runs"
    # data_specific_layout = dict()
    # data_specific_layout_custom = dict()
    small_layout_data = dict()
    medium_layout_data = dict()
    big_layout_data = dict()
    _small_layout_data = []
    _medium_layout_data = []
    _big_layout_data = []
    for fname in os.listdir(directory):
        _fileName = os.path.join(directory, fname)
        if os.path.isfile(_fileName):
            f = open(_fileName, "r")
            f_Data = json.loads(f.readline())
            fileName = _fileName.split("_")
            layout = fileName[2]
            if layout in LAYOUTS_NEW:
                '''
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
                '''
                if layout[0]=="s":
                    _small_layout_data.append(f_Data)
                elif layout[0]=="m":
                    _medium_layout_data.append(f_Data)
                else:
                    _big_layout_data.append(f_Data)
            f.close()
    # print(_small_layout_data)
    # print("==============================")
    # print(medium_layout_data)
    # print("++++++++++++++++++++++++")
    # print(big_layout_data)
    return _small_layout_data,_medium_layout_data,_big_layout_data

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

def genOutput(DATA):
    analysis = {}
    for data in DATA:
        agent= {}
        if data["Layout"] not in analysis.keys():
            analysis[data["Layout"]] = list()
            
        count, totalScore, totalTime = 0, 0, 0
        for g in range(data["TotalGames"]):
            if not data["Crashed"][g]:
                totalScore += data["Scores"][g]
                totalTime += data["Time"][g]
                count+=1
            else:
                continue

        agentSpecificData = {
            "meanScore": 0 if count==0 else round((totalScore/count),2), 
            "meanTime": 0 if count==0 else round((totalTime/count),2),
            "winPercent": round((data["#Wins"]/data["TotalGames"]),2)
            }
        agent[data["Pacman_Agent"]] = agentSpecificData
        analysis[data["Layout"]].append(agent)
    return analysis

def analysis2(small_data, medium_data, big_data):

    small_analysis= genOutput(small_data)
    medium_analysis= genOutput(medium_data)
    big_analysis = genOutput(big_data)

    '''
    for data in small_data:
        agent= {}
        if data["Layout"] not in small_analysis.keys():
            small_analysis[data["Layout"]] = list()
            
        count, totalScore, totalTime = 0, 0, 0
        for g in range(data["TotalGames"]):
            if not data["Crashed"][g]:
                totalScore += data["Scores"][g]
                totalTime += data["Time"][g]
                count+=1
            else:
                continue

        agentSpecificData = {
            "meanScore": 0 if count==0 else round((totalScore/count),2), 
            "meanTime": 0 if count==0 else round((totalTime/count),2),
            "winPercent": round((data["#Wins"]/data["TotalGames"]),2)
            }
        agent[data["Pacman_Agent"]] = agentSpecificData
        small_analysis[data["Layout"]].append(agent)
        break
            small_analysis[data["Layout"]].append()

    for layout in small_data:
        small_analysis[layout["Layout"]] = {}
        # for d in small_data[layout]:
        count, totalScore, totalTime = 0, 0, 0
        for i in range(int(layout['TotalGames'])):
            if not layout["Crashed"][i]:
                totalScore += layout["Scores"][i]
                totalTime += layout["Time"][i]
                count+=1
            else:
                continue
        small_analysis[layout["Layout"]][layout["Pacman_Agent"]] = {
            "meanScore": 0 if count==0 else round((totalScore/count),2), 
            "meanTime": 0 if count==0 else round((totalTime/count),2),
            "winPercent": round((layout["#Wins"]/layout["TotalGames"]),2)
        }
        _small_analysis.append(small_analysis)
    '''
    print(small_analysis)
    return small_analysis, medium_analysis, big_analysis
    
def calAverage(small, medium, big):
    averageValues = {"small": {}, "medium":{}, "big":{}}
    agentAverage = {
        "ExpectimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MonteCarloTreeSearchAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "AlphaBetaAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MinimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        }
    }
    for data,val in small.items():
        for d in val:
            for key,value in d.items():
                agentAverage[key]["meanScore"]+=value['meanScore']
                agentAverage[key]["meanTime"]+=value['meanTime']
                agentAverage[key]["winPercent"]+=value['winPercent']

    averageValues["small"] = agentAverage

    agentAverage = {
        "ExpectimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MonteCarloTreeSearchAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "AlphaBetaAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MinimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        }
    }
    for data,val in medium.items():
        for d in val:
            for key,value in d.items():
                agentAverage[key]["meanScore"]+=value['meanScore']
                agentAverage[key]["meanTime"]+=value['meanTime']
                agentAverage[key]["winPercent"]+=value['winPercent']
    averageValues["medium"] = agentAverage
    
    agentAverage = {
        "ExpectimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MonteCarloTreeSearchAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "AlphaBetaAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        },
        "MinimaxAgent":{
            "meanScore":0,
            "meanTime":0,
            "winPercent":0
        }
    }
    for data,val in big.items():
        for d in val:
            for key,value in d.items():
                agentAverage[key]["meanScore"]+=value['meanScore']
                agentAverage[key]["meanTime"]+=value['meanTime']
                agentAverage[key]["winPercent"]+=value['winPercent']
    averageValues["big"] = agentAverage 
    return averageValues

if __name__ == '__main__':
    # data, dataCustom = getData()
    small_data, medium_data, big_data = getData()
    # per_layout_values, per_layout_values_custom = analysis(data, dataCustom)
    small_per_layout_values, medium_per_layout_values, big_per_layout_values = analysis2(small_data, medium_data, big_data)
    average_all= calAverage(small_per_layout_values, medium_per_layout_values, big_per_layout_values)
    plotNEW(average_all)
    # plot(per_layout_values)
    # plotCUSTOM(per_layout_values_custom)
    

