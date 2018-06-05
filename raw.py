with open("mallet-2.0.8/log.txt","r") as file:
    allLines = file.readlines()
f= open("processedRaw.txt","w")
trainFlag = False
testFlag = False
for i in allLines:
    if "Trainer" in i :
        trainFlag = False
        testFlag = False
    if "Raw Training Data" in i :
        trainFlag = True
        testFlag = False
        continue
    if "Raw Testing Data" in i :
        testFlag = True
        trainFlag = False
        continue
    if  testFlag or trainFlag :
        f.write(i)

with open("processedRaw.txt","r") as file:
    allLines = file.readlines()
counter = 0
wrongsList = []

for i in allLines:
    data = i.split(" ")
    x = data[2].split(":")
    y = data[3].split(":")
    if float(x[1]) > float(y[1]) :
        if x[0] != data[1]:
            print (data)
            wrongsList.append( (int(data[0]),data[1]) )
            counter +=1

    elif float(x[1]) < float(y[1]) :
        if y[0] != data[1]:
            print(data)
            wrongsList.append( (int(data[0]),data[1]) )
            counter += 1
    else :
        print ("Mistake! ")
print ("all wrong tests and trains",counter)

emam = open("emam.txt","r").readlines()
shah = open("shah.txt","r").readlines()

print (wrongsList,"\n")

for i in wrongsList:
    if i[1]=="shah":
        print ("shah: "+ shah[i[0]])
    else :
        print ("emam: "+ emam[i[0]])