import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import json
import scipy.stats as stats # for ANOVA one-way
from statsmodels.stats.multicomp import pairwise_tukeyhsd # for Tukey test if ANOVA rejects Null Hypo

# ALLLAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic","openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]
# LAYOUTS = ["smallClassic", "powerClassic","mediumClassic","trickyClassic"]
# CUSTOMLAYOUTS = ["openClassicR", "openClassicA", "openClassicN", "openClassicP", "mctsmodelClassic"]

LAYOUT_CLASS = ["SMALL", "MEDIUM", "BIG"]
COUNT=20 #Layouts per Class {small, medium, big}
LAYOUTS_NEW = []
LAYOUT_NAME =[ "small", "medium", "big"]
for i in range(3):
    for j in range(1,31):
        LAYOUTS_NEW.append(LAYOUT_NAME[i]+str(j))
class StatisticalTesting:
    def __init__(self):
        pass

    def plotNEW(self,data):
        # x_axis = np.arange(len(LAYOUTS))
        # x_axis = np.arange(len(CUSTOMLAYOUTS))

        x_axis = np.arange(len(LAYOUT_CLASS))
        width = 0.1
        self.AlphaBetaAgent_score, self.ExpectimaxAgent_score, self.MinimaxAgent_score, self.MonteCarloTreeSearchAgent_score = [], [], [], []
        self.AlphaBetaAgent_time, self.ExpectimaxAgent_time, self.MinimaxAgent_time,self.MonteCarloTreeSearchAgent_time = [], [], [], []
        self.AlphaBetaAgent_win, self.ExpectimaxAgent_win, self.MinimaxAgent_win,self.MonteCarloTreeSearchAgent_win = [], [], [], []
        for layout in data:
            for agent in data[layout]:
                if agent=="AlphaBetaAgent":
                    self.AlphaBetaAgent_score.append(data[layout]["AlphaBetaAgent"]["meanScore"]/COUNT)
                    self.AlphaBetaAgent_time.append(data[layout]["AlphaBetaAgent"]["meanTime"]/COUNT)
                    self.AlphaBetaAgent_win.append(data[layout]["AlphaBetaAgent"]["winPercent"]/COUNT)
                elif agent=="ExpectimaxAgent":
                    self.ExpectimaxAgent_score.append(data[layout]["ExpectimaxAgent"]["meanScore"]/COUNT)
                    self.ExpectimaxAgent_time.append(data[layout]["ExpectimaxAgent"]["meanTime"]/COUNT)
                    self.ExpectimaxAgent_win.append(data[layout]["ExpectimaxAgent"]["winPercent"]/COUNT)
                elif agent=="MinimaxAgent":
                    self.MinimaxAgent_score.append(data[layout]["MinimaxAgent"]["meanScore"]/COUNT)
                    self.MinimaxAgent_time.append(data[layout]["MinimaxAgent"]["meanTime"]/COUNT)
                    self.MinimaxAgent_win.append(data[layout]["MinimaxAgent"]["winPercent"]/COUNT)
                elif agent=="MonteCarloTreeSearchAgent":
                    self.MonteCarloTreeSearchAgent_score.append(data[layout]["MonteCarloTreeSearchAgent"]["meanScore"]/COUNT)
                    self.MonteCarloTreeSearchAgent_time.append(data[layout]["MonteCarloTreeSearchAgent"]["meanTime"]/COUNT)
                    self.MonteCarloTreeSearchAgent_win.append(data[layout]["MonteCarloTreeSearchAgent"]["winPercent"]/COUNT)

        plt.figure("Scores")
        plt.bar(x_axis -width, self.AlphaBetaAgent_score, width, label = 'AlphaBetaAgent')
        plt.bar(x_axis, self.ExpectimaxAgent_score, width, label = 'ExpectimaxAgent')
        plt.bar(x_axis +width, self.MinimaxAgent_score, width, label = 'MinimaxAgent')
        plt.bar(x_axis +width*2, self.MonteCarloTreeSearchAgent_score, width, label = 'MonteCarloTreeSearchAgent')
        plt.ylabel("Scores")
        plt.xticks(x_axis,LAYOUT_CLASS)
        # plt.xticks(x_axis,LAYOUTS)
        # plt.xticks(x_axis,CUSTOMLAYOUTS)
        plt.legend()
        plt.savefig('NEW_SCORES.png')
        plt.show()

        plt.figure("Time")
        plt.bar(x_axis -width, self.AlphaBetaAgent_time, width, label = 'AlphaBetaAgent')
        plt.bar(x_axis, self.ExpectimaxAgent_time, width, label = 'ExpectimaxAgent')
        plt.bar(x_axis +width, self.MinimaxAgent_time, width, label = 'MinimaxAgent')
        plt.bar(x_axis +width*2, self.MonteCarloTreeSearchAgent_time, width, label = 'MonteCarloTreeSearchAgent')
        plt.ylabel("Time")
        plt.xticks(x_axis,LAYOUT_CLASS)
        # plt.xticks(x_axis,LAYOUTS)
        # plt.xticks(x_axis,CUSTOMLAYOUTS)
        plt.legend()
        plt.savefig('NEW_time.png')

        plt.figure("Win Percent")
        plt.bar(x_axis -width, self.AlphaBetaAgent_win, width, label = 'AlphaBetaAgent')
        plt.bar(x_axis, self.ExpectimaxAgent_win, width, label = 'ExpectimaxAgent')
        plt.bar(x_axis +width, self.MinimaxAgent_win, width, label = 'MinimaxAgent')
        plt.bar(x_axis +width*2, self.MonteCarloTreeSearchAgent_win, width, label = 'MonteCarloTreeSearchAgent')
        plt.ylabel("Win Percent")
        plt.xticks(x_axis,LAYOUT_CLASS)
        # plt.xticks(x_axis,LAYOUTS)
        # plt.xticks(x_axis,CUSTOMLAYOUTS)
        plt.legend()
        plt.savefig('NEW_win.png')
        
        plt.show()

    def statTesting(self):
        
        # Constructing the data needed to perform Anova testing and Turkey testing
        win_list = [self.AlphaBetaAgent_win, self.ExpectimaxAgent_win, self.MinimaxAgent_win, self.MonteCarloTreeSearchAgent_win]
        val_win_list = self.AlphaBetaAgent_win+self.ExpectimaxAgent_win+self.MinimaxAgent_win+self.MonteCarloTreeSearchAgent_win
        key_win_list = ["AlphaBetaAgent_win"]*len(self.AlphaBetaAgent_win)+["ExpectimaxAgent_win"]*len(self.ExpectimaxAgent_win)+["MinimaxAgent_win"]*len(self.MinimaxAgent_win)+["MonteCarloTreeSearchAgent_win"]*len(self.MonteCarloTreeSearchAgent_win)
        # Anova one-way test. If p value is > 0.05 it indicates that four agents behaved equally,  
        # otherwise, they are statistically not equal and agent behaviour influences the values.
        f_val_win, p_val_win = stats.f_oneway(*win_list)
        print(f"Agent Win Testing - F value: {str(f_val_win)}, P value: {str(p_val_win)} \n")
        # Turkey test.
        tukey_test_win = pairwise_tukeyhsd(val_win_list, key_win_list, alpha=0.05)
        print(tukey_test_win)

        time_list = [self.AlphaBetaAgent_time, self.ExpectimaxAgent_time, self.MinimaxAgent_time, self.MonteCarloTreeSearchAgent_time]
        val_time_list = self.AlphaBetaAgent_time+self.ExpectimaxAgent_time+self.MinimaxAgent_time+self.MonteCarloTreeSearchAgent_time
        key_time_list = ["AlphaBetaAgent_time"]*len(self.AlphaBetaAgent_time)+["ExpectimaxAgent_time"]*len(self.ExpectimaxAgent_time)+["MinimaxAgent_time"]*len(self.MinimaxAgent_time)+["MonteCarloTreeSearchAgent_time"]*len(self.MonteCarloTreeSearchAgent_time)
        f_val_time, p_val_time = stats.f_oneway(*time_list)
        print(f"Agent Time testing - F value: {str(f_val_time)}, P value: {str(p_val_time)} \n")
        tukey_test_time = pairwise_tukeyhsd(val_time_list, key_time_list, alpha=0.05)
        print(tukey_test_time)

        score_list = [self.AlphaBetaAgent_score, self.ExpectimaxAgent_score, self.MinimaxAgent_score, self.MonteCarloTreeSearchAgent_score]
        val_score_list = self.AlphaBetaAgent_score+self.ExpectimaxAgent_score+self.MinimaxAgent_score+self.MonteCarloTreeSearchAgent_score
        key_score_list = ["AlphaBetaAgent_score"]*len(self.AlphaBetaAgent_score)+["ExpectimaxAgent_score"]*len(self.ExpectimaxAgent_score)+["MinimaxAgent_score"]*len(self.MinimaxAgent_score)+["MonteCarloTreeSearchAgent_score"]*len(self.MonteCarloTreeSearchAgent_score)
        f_val_score, p_val_score = stats.f_oneway(*score_list)
        print(f"Agent Score testing- F value: {str(f_val_score)}, P value: {str(p_val_score)} \n")
        tukey_test_score = pairwise_tukeyhsd(val_score_list, key_score_list, alpha=0.05)
        print(tukey_test_score)



    def getData(self):
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

        return _small_layout_data,_medium_layout_data,_big_layout_data

    def analysis(self, data, dataCustom):
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

    def genOutput(self, DATA):
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

    def analysis2(self, small_data, medium_data, big_data):

        small_analysis= self.genOutput(small_data)
        medium_analysis= self.genOutput(medium_data)
        big_analysis = self.genOutput(big_data)

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

    def avg(self, layoutSize):
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
        for data,val in layoutSize.items():
            for d in val:
                for key,value in d.items():
                    agentAverage[key]["meanScore"]+=value['meanScore']
                    agentAverage[key]["meanTime"]+=value['meanTime']
                    agentAverage[key]["winPercent"]+=value['winPercent']

        return agentAverage

    def calAverage(self, small, medium, big):
        averageValues = {"small": {}, "medium":{}, "big":{}}

        averageValues["small"] = self.avg(small)
        averageValues["medium"] = self.avg(medium)
        averageValues["big"] = self.avg(big)
        return averageValues

if __name__ == '__main__':
    # data, dataCustom = getData()
    statsObj = StatisticalTesting()
    small_data, medium_data, big_data = statsObj.getData()
    # per_layout_values, per_layout_values_custom = analysis(data, dataCustom)
    small_per_layout_values, medium_per_layout_values, big_per_layout_values = statsObj.analysis2(small_data, medium_data, big_data)
    average_all= statsObj.calAverage(small_per_layout_values, medium_per_layout_values, big_per_layout_values)
    statsObj.plotNEW(average_all)
    statsObj.statTesting()