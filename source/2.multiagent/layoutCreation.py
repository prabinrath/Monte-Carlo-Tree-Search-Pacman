import random
from math import floor

smallLayout = [10, 12]
mediumLayout = [20, 15]
largeLayout = [35, 20]

# Small Layout
# numberOfLayouts = 30
# while numberOfLayouts:
#     f = open("layouts/small/small"+str(numberOfLayouts)+".lay", "w")
#     rows = random.randint(5,smallLayout[0]) # 2 rows for walls
#     columns = random.randint(6, smallLayout[1]) # 2 cols for walls

#     totalBlocks = rows*columns
#     randomNumber = random.randint(60,80) #No of food items in the layout between 75-100%
#     foodItems = int((totalBlocks*randomNumber)/100)

#     itemsLayout = [random.randint(1,rows), "%", foodItems]

#     pacmanPos = (random.randint(2,rows-2), random.randint(2, columns-2))

#     GHOST = [1,2,3]
#     noOfGhosts = random.choices(GHOST,weights=(20, 60, 25))
#     ghostNumbers = noOfGhosts[0]
#     ghostPos = []
#     for i in range(ghostNumbers):
#         indexGhost = (random.randint(2,rows-2), random.randint(2, columns-2))
#         if pacmanPos==indexGhost or indexGhost in ghostPos:
#             while pacmanPos!=indexGhost and indexGhost not in ghostPos:
#                 indexGhost = (random.randint(1,rows), random.randint(1, columns))
#         ghostPos.append(indexGhost)

#     # print(noOfGhosts)
#     # print(ghostPos)
#     # print(columns)
#     for r in range(rows):
#         line = ""
#         for c in range(columns):
#             if (r,c) == pacmanPos:
#                 # print("P",end="")
#                 line+="P"
#                 continue
#             if (r,c) in ghostPos:
#                 # print("G",end="")
#                 line+="G"
#                 continue
#             if (r==0 or r==rows-1) or (c==0 or c==columns-1):
#                 # print("%", end="")
#                 line+="%"
#             else:
#                 randomList = random.choices(itemsLayout, weights=(5, 15, 100))
#                 index = itemsLayout.index(randomList[0])
#                 if index==0:
#                     # print("o",end="")
#                     line+="o"
#                 elif index==1:
#                     # print("%",end="")
#                     line+="%"
#                 else:
#                     # print(".",end="")
#                     line+="."
#         f.write(line+"\n")
#         print(line)
    
#     numberOfLayouts-=1
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")

# # Medium Layout
# numberOfLayouts = 30
# while numberOfLayouts:
#     f = open("layouts/medium/medium"+str(numberOfLayouts)+".lay", "x")
#     rows = random.randint(13,mediumLayout[0]) # 2 rows for walls
#     columns = random.randint(8, mediumLayout[1]) # 2 cols for walls

#     totalBlocks = rows*columns
#     randomNumber = random.randint(60,80) #No of food items in the layout between 75-100%
#     foodItems = int((totalBlocks*randomNumber)/100)

#     itemsLayout = [random.randint(1,rows), "%", foodItems]

#     pacmanPos = (random.randint(2,rows-2), random.randint(2, columns-2))

#     GHOST = [1,2,3]
#     noOfGhosts = random.choices(GHOST,weights=(20, 25, 25))
#     ghostNumbers = noOfGhosts[0]
#     ghostPos = []
#     for i in range(ghostNumbers):
#         indexGhost = (random.randint(2,rows-2), random.randint(2, columns-2))
#         if pacmanPos==indexGhost or indexGhost in ghostPos:
#             while pacmanPos!=indexGhost and indexGhost not in ghostPos:
#                 indexGhost = (random.randint(1,rows), random.randint(1, columns))
#         ghostPos.append(indexGhost)

#     for r in range(rows):
#         line = ""
#         for c in range(columns):
#             if (r,c) == pacmanPos:
#                 line+="P"
#                 continue
#             if (r,c) in ghostPos:
#                 line+="G"
#                 continue
#             if (r==0 or r==rows-1) or (c==0 or c==columns-1):
#                 line+="%"
#             else:
#                 randomList = random.choices(itemsLayout, weights=(5, 25, 100))
#                 index = itemsLayout.index(randomList[0])
#                 if index==0:
#                     line+="o"
#                 elif index==1:
#                     line+="%"
#                 else:
#                     line+="."
#         f.write(line+"\n")
#         print(line)
    
#     numberOfLayouts-=1
#     print("++++++++++++++++++++++++++++++++++++++++++++++++")

# HARD Layout
numberOfLayouts = 30
while numberOfLayouts:
    f = open("layouts/big/big"+str(numberOfLayouts)+".lay", "w")
    rows = random.randint(15,largeLayout[0]) # 2 rows for walls
    columns = random.randint(10, largeLayout[1]) # 2 cols for walls

    totalBlocks = rows*columns
    randomNumber = random.randint(60,80) #No of food items in the layout between 75-100%
    foodItems = int((totalBlocks*randomNumber)/100)

    itemsLayout = [random.randint(1,rows), "%", foodItems] #powerFood, Wall, Food

    pacmanPos = (random.randint(2,rows-2), random.randint(2, columns-2))

    GHOST = [1,2,3,4]
    noOfGhosts = random.choices(GHOST,weights=(10, 25, 25,25))
    ghostNumbers = noOfGhosts[0]
    ghostPos = []
    for i in range(ghostNumbers):
        indexGhost = (random.randint(2,rows-2), random.randint(2, columns-2))
        if pacmanPos==indexGhost or indexGhost in ghostPos:
            while pacmanPos!=indexGhost and indexGhost not in ghostPos:
                indexGhost = (random.randint(1,rows), random.randint(1, columns))
        ghostPos.append(indexGhost)
    
    layoutArray = []

    for r in range(rows):
        line = ""
        layoutArray.append([])
        for c in range(columns):
            if (r,c) == pacmanPos:
                line+="P"
                continue
            if (r,c) in ghostPos:
                line+="G"
                continue
            if (r==0 or r==rows-1) or (c==0 or c==columns-1):
                line+="%"
            else:
                randomList = random.choices(itemsLayout, weights=(5, 40, 90))
                index = itemsLayout.index(randomList[0])
                if index==0:
                    line+="o"
                elif index==1:
                    try:
                        if layoutArray[r-1][0][c] == "%":
                            if itemsLayout[2]==0:
                                line+=" "
                                continue
                            else:
                                line+="."
                                itemsLayout[2]-=1
                                continue
                    except:
                        print("out of bounds"+str(r))
                        line+="."
                        continue
                    line+="%"
                else:
                    if itemsLayout[2]==0:
                        continue
                    # try:
                    #     if layoutArray[r-1][0][c] == "%" or layoutArray[r][0][c-1] == "%":
                    #         line+="%"
                    #         continue
                    # except:
                    #     line+=" "
                    #     continue
                    line+="."
                    itemsLayout[2]-=1
        # f.write(line+"\n")
        layoutArray[r].append(line)
        print(line)
    
    f.write(layoutArray[0][0]+"\n")
    for i in range(1,rows-1):
        for j in range(1,columns-1):
            if layoutArray[i][0][j] == ".":
                    if (layoutArray[i-1][0][j]=="%" and layoutArray[i][0][j+1]=="%" and layoutArray[i][0][j-1]=="%") or\
                        (layoutArray[i-1][0][j]=="%" and layoutArray[i+1][0][j]=="%" and layoutArray[i][0][j+1]=="%") or\
                        (layoutArray[i][0][j+1]=="%" and layoutArray[i+1][0][j]=="%" and layoutArray[i][0][j-1]=="%") or\
                        (layoutArray[i-1][0][j]=="%" and layoutArray[i+1][0][j]=="%" and layoutArray[i][0][j+1]=="%") :
                        splitLayout = [*layoutArray[i][0]]
                        splitLayout[j] = "%"
                        joinAgain = "".join(splitLayout)
                        layoutArray[i] = [joinAgain]
        f.write(layoutArray[i][0]+"\n")
    
    f.write(layoutArray[-1][0]+"\n")

    
    numberOfLayouts-=1
    print("++++++++++++++++++++++++++++++++++++++++++++++++")