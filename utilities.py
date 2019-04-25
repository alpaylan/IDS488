import pandas as pd

fileNames = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv","Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv","Friday-WorkingHours-Morning.pcap_ISCX.csv","Monday-WorkingHours.pcap_ISCX.csv","Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv","Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv","Tuesday-WorkingHours.pcap_ISCX.csv","Wednesday-workingHours.pcap_ISCX.csv"]
editedFile = "dataset.csv"
features = None
normalTrafficLabel = "BENIGN"
averageSampleSize = 20000

def loadCSV(name):
        print "Loading " + name
        ret = pd.read_csv(name, low_memory=False)
        print "Done"
        return ret
        

def labelTypes(l):
    tmp = []
    for i in l:
        if tmp.count(i) == 0:
            tmp.append(i)
    return tmp

def editNames(l):
        for i in range(0, len(l)):
                while l[i][0] == " ":
                        l[i] = l[i][1:]
                l[i] = l[i].replace(" ", "_")
        return l

def mergeFrames(l):
        print "Initializing DataFrames..."
        if len(l) == 0:
                print "Error: no DataFrame in list"
                return None
        tmp = l[0]
        for i in range(1, len(l)):
                tmp = tmp.append(l[i])
        print "Done"
        return tmp

def clearNaValues(df):
        initialShape = df.shape
        print "Cleaning NaN values..."
        df = df.dropna()
        finalShape = df.shape
        print "Nan values cleared"
        print "Shape of DF before cleaning:" + str(initialShape)
        print "Shape of DF after cleaning :" + str(finalShape)
        return df

def saveFeatures(df, fileName):
        print "Saving DataFrame.."
        df.to_csv(fileName, ",", index = False)

def labelCounts(df):
        labelNames = labelTypes(list(df["Label"]))
        labels = list(df["Label"])
        print "Labels:" + str(labelNames)
        labelCnt = [0]*len(labelNames)
        for i in labels:
                labelCnt[labelNames.index(i)]+=1
        return ([(labelNames[i], labelCnt[i]) for i in range(0, len(labelNames))], len(labels))




"""--------------------------------------------------interface functions-----------------------------------------------"""


def loadRaw():
        global features
        featureList = [loadCSV(i) for i in fileNames]
        features = mergeFrames(featureList)
        del featureList
        featureNames = list(features)
        editNames(featureNames)
        features.columns = featureNames
        features = clearNaValues(features)

def loadEdited():
        global features
        features = loadCSV(editedFile)

def save(fileName = editedFile, frame = features):
        saveFeatures(frame, fileName)

def printLabelCounts():
        counts, total = labelCounts(features)
        counts.sort(key = lambda i : i[1], reverse = True)
        for i in counts:
                print i[0] + " -> " + str(i[1]) + " (" + str(100.0*float(i[1])/float(total)) + "%)"
        print "Total:" + str(total)

def getNaLocations():
        nulls = list(features.isnull().any(axis = 1))
        indexes = []
        for i in range(0, len(nulls)):
                if nulls[i]:
                        indexes.append(i)
        return indexes

def getNaRows():
        return features.iloc[getNaLocations()]

def dropLabeled(n):
        global features
        labelList = labelCounts(features)[0]
        labelList.sort(key = lambda i : i[1], reverse = True)
        labels = []
        for i in labelList[-1:-n-1:-1]:
                labels.append(i[0])
        print "labels to drop=" + str(labels)
        l = list(features["Label"])
        indexes = []
        for i in range(0, len(l)):
                if l[i] in labels:
                        indexes.append(i)
        features = features.drop(indexes)

def getLabels():
        global features
        return labelTypes(list(features["Label"]))

def getDivided():
        global features
        types = labelTypes(list(features["Label"]))
        labels = list(features["Label"])
        ret = []
        for i in types:
                indexes = []
                for j in range(0, len(labels)):
                        if labels[j] == i:
                                indexes.append(j)
                ret.append(features.loc[indexes,:])
        return ret

def sample():
        divided = getDivided()
        benignIndex = -1
        for i in range(0, len(divided)):
                if list(divided[i]["Label"])[0] == normalTrafficLabel:
                        benignIndex = i
                        continue
                if len(divided[i]) < averageSampleSize:
                        divided[i] = divided[i].sample(averageSampleSize, replace = True)
                else:
                        divided[i] = divided[i].sample(averageSampleSize)
        divided[benignIndex] = divided[benignIndex].sample(averageSampleSize * (len(divided)-1))
        for i in range(0, len(divided)):
                divided[i] = divided[i].reset_index(drop = True)
        return divided

def getSampledFeatures():
        merged = mergeFrames(sample())
        merged = merged.sample(frac = 1)
        merged = merged.reset_index(drop = True)
        return merged