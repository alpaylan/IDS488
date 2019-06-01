from matplotlib import pyplot as plt

def savePie(labels, percents, name):
        fig1, ax1 = plt.subplots()
        ax1.pie(percents, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=180)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.savefig(name)
percents = [80.3244521336, 8.13335760245, 5.617121742,
            4.52490558902, 100 - (80.3244521336 + 8.13335760245 + 5.617121742 + 4.52490558902)]

labels = ["BENIGN", "DoS Hulk", "PortScan", "DDoS",
          "Other (11)"]

savePie(labels, percents, "piechart.png")

percents = [80.3877250582, 8.13976438529, 5.62154644346,
            4.52846993341, 100 - (80.3877250582 + 8.13976438529 + 5.62154644346 + 4.52846993341)]

labels = ["BENIGN", "DoS Hulk", "PortScan", "DDoS",
          "Other (6)"]
savePie(labels, percents, "piechart2.png")

percents = 10*[10]

labels = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP Patator", "SSH Patator", "DoS slowloris", "DoS Slowhttptest", "Bot"]

savePie(labels, percents, "piechart3.png")