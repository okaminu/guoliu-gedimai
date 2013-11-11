import numpy as np
import math
import matplotlib.pyplot as plt
import re
import wx
import copy
import os

# Vytauto testas
#import threading

class Signal:

    signalNames = []
    signalData = []

    def initValues(self):
        self._fileCol = 3
        self._lastFigure = 0
        self._rolls = 0
        self._rmsOrig = ''
        self._rmsClean = ''
        self._rmsMean = ''
        self._rmsCleanSignal = ''
        self._frameSize = 0
        self._rolls= 0
        self._limit = 0
        self._skip = 0
        self._limit = 0
        self._freMark = 0.0
        self._proportion = 9.81
        self._displayParams = 'g'
        self._originalData = []
        self._originalDataFreq = []
        self._cleanData = []
        self._cleanDataFreq = []
        self._meanFrame = []
        self._meanFrameFreq = []
        self._cleanTimeFrame = []
        self._cleanFreqFrame = [] #this one is made when each clean frame is converted to Freq and stacked (Freq frames stacked)
        self._cleanTimeFrameFreq = [] # this one is made when full cleaned signal time interval is converted to freq spectrum
        self._SingleRollTime = 0;
        self.isDrawLegend = 0
        self.rmsLocation = "statistics/RMS/"
        self._hideFreq
        self.corrMode = 'full'

        self._originalDataCorr = []
        self._cleanDataCorr = []
        self._meanFrameCorr = []
        self._cleanTimeFrameCorr = []
        self._originalDataFreqCorr = []
        self._cleanDataFreqCorr = []
        self._meanFrameFreqCorr = []
        self._cleanTimeFrameFreqCorr = []
        self._cleanFreqFrameCorr = []
        self._originalDataCeps = []
        self._cleanDataCeps = []
        self._meanFrameCeps = []
        self._cleanTimeFrameCeps = []

    def __init__(self, range, frameSize, skip, freMark, singleRollTime, isRangeTime, isSkipTime, hideFreq):
        self._hideFreq = int(hideFreq)
        self.initValues();
        self._SingleRollTime = int(singleRollTime)
        self._rolls = int(range)
        if(isRangeTime == 1):
            self._rolls = self.convertTimeToRolls(range)
        self._frameSize = int(frameSize)
        skipFrames = int(skip)
        if(isSkipTime == 1):
            skipFrames = self.convertTimeToRolls(skip)
        self._skip = self._skip + (skipFrames * self._frameSize)
        self._limit = self._rolls * self._frameSize
        self._limit = self._skip + self._limit
        self._freMark = float(freMark)
        self._displayParams = self._displayParams

    def convertTimeToRolls(self, time):
        msTime = int(float(time)) * 1000
        return int(float(msTime / self._SingleRollTime))

    def convertPointsToRolls(self, points):
        return float(points / self._frameSize)

    def convertRollsToTime(self, rolls):
        rollsMs = float(rolls * self._SingleRollTime)
        return  float(rollsMs / 1000)


    def _loadOriginal_File(self, location):
        fileData1 = open(location, "r")
        dialog = wx.ProgressDialog('Skaiciuoja duomenis', 'Prasome palaukti', self._limit, style=wx.PD_REMAINING_TIME)
        dataBegin = 0

        for itera in range(self._limit):
            line = fileData1.readline()
            if itera >= self._skip and dataBegin == 1:
                dataRaw = re.split("\t", line)
                data = float(dataRaw[self._fileCol-1])
                self._originalData.append(data)
                # dialog.Update(itera)
            if re.match("BEGIN_DATA", line):
                dataBegin = 1
        wx.CallAfter(dialog.Destroy)

    def _alterTimeSignal(self):
        for single in range(len(self._originalData)):
            self._originalData[single] = self._originalData[single] * self._proportion

    def _calcMeanFrame(self):

        bufIter = 0
        buffer = [0] * self._frameSize
        originalData = self._originalData
        frameSize = self._frameSize

        for iter in range(len(originalData)-frameSize):
            buffer[bufIter] = (buffer[bufIter] + originalData[iter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                bufIter=0
        self._meanFrame = buffer


    def _cleanSignal(self):

        bufIter = 0
        originalData = self._originalData
        frameSize = self._frameSize
        cleanData=[0]* (len(originalData)-frameSize)
        meanFrame = self._meanFrame

        for itera in range(len(originalData)-frameSize):
            cleanData[itera] = (originalData[itera] - meanFrame[bufIter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                bufIter=0
        self._cleanData = cleanData

    def _stackCleanSignalFrames_Time_Freq(self):
        bufIter = 0
        frameSize = self._frameSize
        cleanTimeFrame = [0] * frameSize
        cleanFreqFrame = [0]*((frameSize/2)+1)
        cleanFreqFrameTemp = [0]*((frameSize/2)+1)
        cleanData = self._cleanData
        buffer = [0] * frameSize

        for itera in range(len(cleanData) - frameSize):

            buffer[bufIter] = cleanData[itera]
            cleanTimeFrame[bufIter] = (cleanData[itera] + cleanTimeFrame[bufIter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                cleanFreqFrameTemp = abs(np.fft.rfft(buffer))
                for itera2 in range(len(cleanFreqFrameTemp)):
                    cleanFreqFrame[itera2] = (cleanFreqFrameTemp[itera2] + cleanFreqFrame[itera2]) / 2
                bufIter=0
        self._cleanTimeFrame = cleanTimeFrame
        self._cleanFreqFrame = cleanFreqFrame

    def _calcFreqSpectrums(self):

        self._originalDataFreq = abs(np.fft.rfft(self._originalData)) / 555.555555
        for iter2 in range(len(self._originalDataFreq)):
            if self._originalDataFreq[iter2] < 0:
                self._originalDataFreq[iter2] = self._originalDataFreq[iter2] * -1
        self._cleanDataFreq = abs(np.fft.rfft(self._cleanData)) / 555.555555
        self._meanFrameFreq = abs(np.fft.rfft(self._meanFrame))
        self._cleanTimeFrameFreq = abs(np.fft.rfft(self._cleanTimeFrame))

        #JS pradzia
    def calcCorrelation(self, signal):
        self._originalDataCorr = np.correlate(self._originalData, signal._originalData, mode=self.corrMode)
        self._cleanDataCorr = np.correlate(self._cleanData, signal._cleanData, mode=self.corrMode)
        self._meanFrameCorr = np.correlate(self._meanFrame, signal._meanFrame, mode=self.corrMode)
        self._cleanTimeFrameCorr = np.correlate(self._cleanTimeFrame, signal._cleanTimeFrame, mode=self.corrMode)
        self._originalDataFreqCorr = np.correlate(self._originalDataFreq, signal._originalDataFreq, mode=self.corrMode)
        self._cleanDataFreqCorr = np.correlate(self._cleanDataFreq, signal._cleanDataFreq, mode=self.corrMode)
        self._meanFrameFreqCorr = np.correlate(self._meanFrameFreq, signal._meanFrameFreq, mode=self.corrMode)
        self._cleanTimeFrameFreqCorr = np.correlate(self._cleanTimeFrameFreq, signal._cleanTimeFrameFreq, mode=self.corrMode)
        self._cleanFreqFrameCorr = np.correlate(self._cleanFreqFrame, signal._cleanFreqFrame, mode=self.corrMode)

    def _calcCepstrums(self):
        self._originalDataCeps = self.cepstrum(self._originalData)
        self._cleanDataCeps = self.cepstrum(self._cleanData)
        self._meanFrameCeps = self.cepstrum(self._meanFrame)
        self._cleanTimeFrameCeps = self.cepstrum(self._cleanTimeFrame)


    def cepstrum(self, signal):
        temp = abs(np.fft.rfft(signal))
        for index, item in enumerate(temp):
            if(temp[index] == 0):
                temp[index] = 0.00001

        cepstrum = abs(np.fft.irfft(np.log(temp)))
        return cepstrum

    def _rmsOriginal (self):
        sum = 0
        originalData = self._originalData
        for itera in range (len(originalData)):
            sum += ( math.pow(originalData[itera], 2))
        aver = sum / len(originalData)

        return str(math.sqrt(aver))

    def _rmsCleaned (self):
        sum = 0

        cleanData = self._cleanData
        for itera in range (len(cleanData)):
            sum += ( math.pow(cleanData[itera], 2))

        aver = sum / len(cleanData)

        return str(math.sqrt(aver))

    def _rmsMeanF (self):
        sum = 0

        meanData = self._meanFrame

        for itera in range (len(meanData)):
            sum += ( math.pow(meanData[itera], 2))

        aver = sum / len(meanData)

        return str(math.sqrt(aver))

    def _rmsCleanedSF (self):
        sum = 0
        number = 0
        kiekN = 0
        cleanTFData = self._cleanTimeFrame
        for itera in range (len(cleanTFData)):
            sum += (cleanTFData[itera]*(math.sin(2*3.14*50*0.02)))
            if itera % 1024 == 0:
                number += sum ** 2
                kiekN+=1
                sum = 0
        return str((number / kiekN) ** (1.0/2))

    def _saveRMS(self, fileName):

        if(os.path.exists(self.rmsLocation) == 0):
            os.makedirs(self.rmsLocation)

        file = open(self.rmsLocation+fileName+".txt", 'w')
        file.writelines("Originalus: "+ self._rmsOriginal() +"\n")
        file.writelines("Centruotas: "+ self._rmsCleaned() +"\n")
        file.writelines("Originalo vidurkintas: "+ self._rmsMean +"\n")
        file.writelines("Centruoto vidurkintas: "+ self._rmsCleanedSF() +"\n")

    def _displayTime(self, data, displayParams, marking = 's'):
        self._lastFigure+=1
        plt.figure(self._lastFigure)
        subplots = 211
        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            plt.plot(range(0, len(data[iter]['values'])), data[iter]['values'], displayParams)
            plt.title(data[iter]['title'])
            subplots+=1

            Lenght = float(len(data[iter]['values']))
            xAxisMarkerValues = []
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            for marker in range(len(xAxisMarkerPlacement)):
                rolls = self.convertPointsToRolls(xAxisMarkerPlacement[marker])
                if(marking == 's'):
                    time = round(float(self.convertRollsToTime(rolls)), 2)
                elif(marking == 'ms'):
                    time = round(self.convertRollsToTime(rolls),2)
                xAxisMarkerValues.append(time)


            xAxisMarkerValues[len(xAxisMarkerValues) -1] = str(xAxisMarkerValues[len(xAxisMarkerValues) -1]) + " s"
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')
        if(self.isDrawLegend == 1):
                self.drawLegend()


    def _displayFreq(self, data, displayParams):
        self._lastFigure +=1
        xAxisMarkerValues = ['0', '5', '10', '15', '20', '25 kHz']
        plt.figure(self._lastFigure)
        subplots = 211

        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            Lenght = float(len(data[iter]['values']))
            for i in range(0,int((Lenght/25000)*int(self._hideFreq)),1):
                data[iter]['values'][i]=0
            localFreMark = (Lenght / 25000) * self._freMark
            plt.bar(localFreMark, max(data[iter]['values']), width=0.8, edgecolor = '#CCCCCC')
            plt.plot(range(0, int(Lenght)), data[iter]['values'], displayParams)
            plt.title(data[iter]['title'])
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')
            subplots+=1

        if(self.isDrawLegend == 1):
            self.drawLegend()

    def _displayCorr(self, data, displayParams):
        self._lastFigure+=1
        plt.figure(self._lastFigure)
        subplots = 211
        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            plt.plot(range(0, len(data[iter]['values'])), data[iter]['values'], displayParams)
            plt.title(data[iter]['title'])
            subplots+=1


        if(self.isDrawLegend == 1):
            self.drawLegend()
    def getDrawLegend(self):
        return self.isDrawLegend

    def setDrawLegend(self, state):
        self.isDrawLegend = state

    def drawLegend(self):
        plt.legend(self.signalData, self.signalNames,
                   bbox_to_anchor=[0.5, 0], loc='upper center', ncol=4, borderaxespad=1)

    def addToLegend(self, signalName, color):
        self.signalNames.append(signalName)
        self.signalData.append(plt.Rectangle([0,0],1,1, fc = color))

    def displayAllTime(self, displayParams):
        originalDisplay = {'values' : self._originalData, 'title' : 'Originalus (Laikas)'}
        cleanDisplay = {'values' : self._cleanData, 'title' : 'Centruotas (Laikas)'}
        self._displayTime({0:originalDisplay, 1: cleanDisplay}, displayParams, 's')

        meanDisplay = {'values' : self._meanFrame, 'title' : 'Originalo vidurkis (Laikas)'}
        cleanFrameDisplay = {'values' : self._cleanTimeFrame, 'title' : 'Centruoto vidurkis (Laikas)'}
        self._displayTime({0:meanDisplay, 1: cleanFrameDisplay}, displayParams,'ms')


    def displayAllFreq(self, displayParams):

        origFreqDisplay = {'values' : self._originalDataFreq, 'title' : 'Originalus (Daznis)'}
        cleanFreqDisplay = {'values' : self._cleanDataFreq, 'title' : 'Centruotas (Daznis)'}
        self._displayFreq({0:origFreqDisplay, 1: cleanFreqDisplay}, displayParams)

        meanFreqFrameDisplay = {'values' : self._meanFrameFreq, 'title' : 'Originalo vidurkis (Daznis)'}
        cleanFreqFrameDisplay = {'values' : self._cleanTimeFrameFreq, 'title' : 'Centruoto vidurkis (Daznis)'}
        self._displayFreq({0:meanFreqFrameDisplay, 1: cleanFreqFrameDisplay}, displayParams)

        cleanFreqFrame2Display = {'values' : self._cleanFreqFrame, 'title' : 'Centruoto Signalo dazniu vidurkis'}
        self._displayFreq({0: cleanFreqFrame2Display}, displayParams)

    def displayAllCepstrums(self, displayParams):
        originalCepstDisplay = {'values' : self._originalDataCeps, 'title' : 'Originalus (Laikas) Kepstras'}
        cleanCepstDisplay = {'values' : self._cleanDataCeps, 'title' : 'Centruotas (Laikas) Kepstras'}
        self._displayTime({0:originalCepstDisplay, 1: cleanCepstDisplay}, displayParams, 's')

        meanCepstDisplay = {'values' : self._meanFrameCeps, 'title' : 'Originalo vidurkis (Laikas) Kepstras'}
        cleanFrameCepstDisplay = {'values' : self._cleanTimeFrameCeps, 'title' : 'Centruoto vidurkis (Laikas) Kepstras'}
        self._displayTime({0:meanCepstDisplay, 1: cleanFrameCepstDisplay}, displayParams,'ms')

    def displayAllCorrelations(self, displayParams):
        originalCorrDisplay = {'values' : self._originalDataCorr, 'title' : 'Originalus (Laikas) Koreliacija'}
        cleanCorrDisplay = {'values' : self._cleanDataCorr, 'title' : 'Centruotas (Laikas) Koreliacija'}
        self._displayCorr({0:originalCorrDisplay, 1: cleanCorrDisplay}, displayParams)

        meanCorrDisplay = {'values' : self._meanFrameCorr, 'title' : 'Originalo vidurkis (Laikas) Koreliacija'}
        cleanFrameCorrDisplay = {'values' : self._cleanTimeFrameCorr, 'title' : 'Centruoto vidurkis (Laikas) Koreliacija'}
        self._displayCorr({0:meanCorrDisplay, 1: cleanFrameCorrDisplay}, displayParams)

        origFreqCorrDisplay = {'values' : self._originalDataFreqCorr, 'title' : 'Originalus (Daznis) Koreliacija'}
        cleanFreqCorrDisplay = {'values' : self._cleanDataCorr, 'title' : 'Centruotas (Daznis) Koreliacija'}
        self._displayCorr({0:origFreqCorrDisplay, 1: cleanFreqCorrDisplay}, displayParams)

        meanFreqFrameCorrDisplay = {'values' : self._meanFrameFreqCorr, 'title' : 'Originalo vidurkis (Daznis) Koreliacija'}
        cleanFreqFrameCorrDisplay = {'values' : self._cleanTimeFrameFreqCorr, 'title' : 'Centruoto vidurkis (Daznis) Koreliacija'}
        self._displayCorr({0:meanFreqFrameCorrDisplay, 1: cleanFreqFrameCorrDisplay}, displayParams)

        cleanFreqFrame2CorrDisplay = {'values' : self._cleanFreqFrameCorr, 'title' : 'Centruoto Signalo dazniu vidurkis Koreliacija'}
        self._displayCorr({0: cleanFreqFrame2CorrDisplay}, displayParams)

    def displayAllData(self, color, signalName, isCorr = 0):
        self._saveRMS(signalName)

        displayParams = color;

        self.displayAllTime(displayParams)

        self.displayAllFreq(displayParams)

        self.displayAllCepstrums(displayParams)

        if(isCorr == 1):
            self.displayAllCorrelations(displayParams)

        self._lastFigure = 0

    def substractFrom(self, signal2):
        leng = min({len(signal2._originalData), len(self._originalData)})
        for iter1 in range(leng):
            self._originalData[iter1] = self._originalData[iter1] - signal2._originalData[iter1]

        leng2 = min({len(signal2._cleanData), len(self._cleanData)})
        for iter2 in range(leng2):
            self._cleanData[iter2] = self._cleanData[iter2] - signal2._cleanData[iter2]

        leng3 = min({len(signal2._meanFrame), len(self._meanFrame)})
        for iter4 in range(leng3):
            self._meanFrame[iter4] = self._meanFrame[iter4] - signal2._meanFrame[iter4]

        leng4 = min({len(signal2._cleanTimeFrame), len(self._cleanTimeFrame)})
        for iter5 in range(leng4):
            self._cleanTimeFrame[iter5] = self._cleanTimeFrame[iter5] - signal2._cleanTimeFrame[iter5]

        leng5 = min({len(signal2._originalDataFreq), len(self._originalDataFreq)})
        for iter6 in range(leng5):
            self._originalDataFreq[iter6] = self._originalDataFreq[iter6] - signal2._originalDataFreq[iter6]

        leng6 = min({len(signal2._cleanDataFreq), len(self._cleanDataFreq)})
        for iter7 in range(leng6):
            self._cleanDataFreq[iter7] = self._cleanDataFreq[iter7] - signal2._cleanDataFreq[iter7]

        leng7 = min({len(signal2._meanFrameFreq), len(self._meanFrameFreq)})
        for iter8 in range(leng7):
            self._meanFrameFreq[iter8] = self._meanFrameFreq[iter8] - signal2._meanFrameFreq[iter8]

        leng8 = min({len(signal2._cleanTimeFrame), len(self._cleanTimeFrameFreq)})
        for iter9 in range(leng8):
            self._cleanTimeFrameFreq[iter9] = self._cleanTimeFrameFreq[iter9] - signal2._cleanTimeFrameFreq[iter9]

        leng9 = min({len(signal2._cleanFreqFrame), len(self._cleanFreqFrame)})
        for iter10 in range(leng9):
            self._cleanFreqFrame[iter10] = self._cleanFreqFrame[iter10] - signal2._cleanFreqFrame[iter10]

    def processSignalFromFile(self, location):
        self._loadOriginal_File(location)
        self._alterTimeSignal()
        self._calcMeanFrame()
        self._cleanSignal()
        self._stackCleanSignalFrames_Time_Freq()
        self._calcFreqSpectrums()
        self._calcCepstrums()

def execCalc(event):

    skipTime = 1
    rangeTime = 1

    if((isInputTime.GetValue() == 0) and (isInputPoint.GetValue() == 1)):
        rangeTime = 0
    if((isSkipTime.GetValue() == 0) and (isSkipPoint.GetValue() == 1)):
        skipTime = 0
    signal1 = Signal(
        inputRange.GetValue(),
        inputFrame.GetValue(),
        inputSkip.GetValue(),
        inputFreMark.GetValue(),
        inputSingleRollTime.GetValue(),
        rangeTime,
        skipTime,
        inputHideFreq.GetValue()
    )
    signal1.processSignalFromFile('matavimai/'+ inputFile.GetValue())
    signal1.addToLegend('Rezonansas', '#CCCCCC')

    corrSignal = signal1
    if(inputFile2.GetValue() != ''):
        signal2 = Signal(
            inputRange.GetValue(),
            inputFrame.GetValue(),
            inputSkip.GetValue(),
            inputFreMark.GetValue(),
            inputSingleRollTime.GetValue(),
            rangeTime,
            skipTime,
            inputHideFreq.GetValue()
        )
        signal2.processSignalFromFile('matavimai/'+ inputFile2.GetValue())
        signalDiff = copy.deepcopy(signal1)
        signalDiff.substractFrom(signal2)
        signalDiff.displayAllData('r', 'Skirtumas')
        signal2.displayAllData('b', 'Antras')
        corrSignal = signal2
        del signalDiff
        del signal2

    #signal1.calcCorrelation(corrSignal)
    signal1.setDrawLegend(1)
    signal1.displayAllData('g', 'Pirmas')
    del signal1
    # Draw the plot to the screen
    plt.show()




appTitle = 'Guoliu Gedimai v0.7.8'
app = wx.App(False)  # Create a new app, don't redirect stdout/stderr to a window.
frame = wx.Frame(None, wx.ID_ANY, title=appTitle, size=(450, 400)) # A Frame is a top-level window.
frame.Show(True)     # Show the frame.
button = wx.Button(frame, label="Vykdyti", pos=(170, 300))
inputFile = wx.TextCtrl(frame,-1,pos=(180, 60), size=(110, 20), value=('m6.txt'))
inputFile2 = wx.TextCtrl(frame,-1,pos=(180, 90), size=(110, 20), value=(''))
inputRange = wx.TextCtrl(frame,-1,pos=(180, 120), size=(50, 20), value=('1'))
isInputTime = wx.RadioButton(frame,label = 'sekundes',pos=(240, 120), style=wx.RB_GROUP)
isInputPoint = wx.RadioButton(frame,label = 'apsisukimai',pos=(340, 120))
isInputTime.SetValue(1)
inputSkip = wx.TextCtrl(frame,-1,pos=(180, 150), size=(50, 20), value=('25'))
isSkipTime = wx.RadioButton(frame,label = 'sekundes',pos=(240, 150), style=wx.RB_GROUP)
isSkipPoint = wx.RadioButton(frame,label = 'apsisukimai',pos=(340, 150))
isSkipTime.SetValue(1)
inputFrame = wx.TextCtrl(frame,-1,pos=(180, 180), size=(50, 20), value=('1024'))
inputSingleRollTime = wx.TextCtrl(frame,-1,pos=(180, 210), size=(50, 20), value=('20'))
inputFreMark = wx.TextCtrl(frame,-1,pos=(180, 240), size=(50, 20), value=('0'))
inputHideFreq = wx.TextCtrl(frame,-1,pos=(180, 270), size=(50, 20), value=('0'))

label0 = wx.StaticText(frame, -1, appTitle , pos=(30, 20))
font = wx.Font(16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
label0.SetFont(font)
label0.SetForegroundColour(wx.Colour(14,181,56));
frame.SetBackgroundColour(wx.Colour(225,225,225));
label1 = wx.StaticText(frame, -1, 'Pirmas matavimu failas' , pos=(15, 60))
label1 = wx.StaticText(frame, -1, 'Antras matavimu failas' , pos=(15, 90))
label9 = wx.StaticText(frame, -1, 'Imtis' , pos=(15, 120))
label5 = wx.StaticText(frame, -1, 'Praleisti' , pos=(15, 150))
label3 = wx.StaticText(frame, -1, 'Tasku kiekis apsisukime' , pos=(15, 180))
label14 = wx.StaticText(frame, -1,'Apsisukimo trukme (ms)' , pos=(15, 210))
label13 = wx.StaticText(frame, -1,'Rezonansas (Hz)', pos=(15, 240))
label2 = wx.StaticText(frame, -1,'Slept pirmus daznius (Hz)', pos=(15, 270))
label10 = wx.StaticText(frame, -1,'Veiksmas', pos=(15, 300))
label12 = wx.StaticText(frame, -1,"Autorius: AurimasDGT", pos=(15, 330))
label12.SetForegroundColour(wx.Colour(173,88,88));

button.Bind(wx.EVT_BUTTON, execCalc)

app.MainLoop()