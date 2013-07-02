import numpy as np
import math
import matplotlib.pyplot as plt
import re
import wx


#import threading

class Signal:

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

    def __init__(self, time, frameSize, skip, skipFramesTime, freMark, singleRollTime):
        self.initValues();
        self._SingleRollTime = int(singleRollTime)
        self._rolls = self.convertTimeToRolls(time)
        self._frameSize = int(frameSize)
        self._skip = int(skip)
        skipFrames = self.convertTimeToRolls(skipFramesTime)
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

        for itera in range(self._limit):
            line = fileData1.readline()
            if itera < self._skip:
                continue
            dataRaw = re.split("\t", line)
            data = float(dataRaw[self._fileCol-1])
            self._originalData.append(data)
            # dialog.Update(itera)
        wx.CallAfter(dialog.Destroy)

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

        self._originalDataFreq = abs(np.fft.rfft(self._originalData)) /100
        for iter2 in range(len(self._originalDataFreq)):
            if self._originalDataFreq[iter2] < 0:
                self._originalDataFreq[iter2] = self._originalDataFreq[iter2] * -1
        self._cleanDataFreq = abs(np.fft.rfft(self._cleanData))
        self._meanFrameFreq = abs(np.fft.rfft(self._meanFrame))
        self._cleanTimeFrameFreq = abs(np.fft.rfft(self._cleanTimeFrame))

        #JS pradzia

    def _rmsOriginal (self):
        sum = 0
        originalData = self._originalData
        for itera in range (len(originalData)):
            sum += ( math.pow(originalData[itera], 2))
        aver = sum / len(originalData)

        self.rmsOrig = str(math.sqrt(aver))

    def _rmsCleaned (self):
        sum = 0

        cleanData = self._cleanData
        for itera in range (len(cleanData)):
            sum += ( math.pow(cleanData[itera], 2))

        aver = sum / len(cleanData)

        self.rmsClean = str(math.sqrt(aver))

    def _rmsMeanF (self):
        sum = 0

        meanData = self._meanFrame

        for itera in range (len(meanData)):
            sum += ( math.pow(meanData[itera], 2))

        aver = sum / len(meanData)

        self.rmsMean = str(math.sqrt(aver))

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
        self.rmsCleanSignal = str((number / kiekN) ** (1.0/2))


    def _displayTime(self, data, marking = 's'):
        self._lastFigure+=1
        plt.figure(self._lastFigure)
        subplots = 211
        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            plt.plot(range(0, len(data[iter]['values'])), data[iter]['values'], self._displayParams)
            plt.title(data[iter]['title'])
            subplots+=1

            Lenght = float(len(data[iter]['values']))
            xAxisMarkerValues = []
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            for marker in range(len(xAxisMarkerPlacement)):
                rolls = self.convertPointsToRolls(xAxisMarkerPlacement[marker])
                if(marking == 's'):
                    time = int(math.ceil(self.convertRollsToTime(rolls)))
                elif(marking == 'ms'):
                    time = self.convertRollsToTime(rolls)
                xAxisMarkerValues.append(time)

            xAxisMarkerValues[len(xAxisMarkerValues) -1] = str(xAxisMarkerValues[len(xAxisMarkerValues) -1]) + " s"
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')


    def _displayFreq(self, data):
        self._lastFigure+=1
        xAxisMarkerValues = ['0', '5', '10', '15', '20', '25 kHz']
        plt.figure(self._lastFigure)
        subplots = 211

        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            Lenght = float(len(data[iter]['values']))
            localFreMark = (Lenght / 25000) * self._freMark
            plt.bar(localFreMark, max(data[iter]['values']), width=0.8)
            plt.plot(range(0, int(Lenght)), data[iter]['values'], self._displayParams)
            plt.title(data[iter]['title'])
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')

            subplots+=1


    def displayAllData(self):
        originalDisplay = {'values' : self._originalData, 'title' : 'Original signal (Time)'+' RMS='+self.rmsOrig+' m/s^2'}
        cleanDisplay = {'values' : self._cleanData, 'title' : 'Cleaned signal (Time)'+' RMS='+self.rmsClean+' m/s^2'}
        self._displayTime({0:originalDisplay, 1: cleanDisplay}, 's')

        meanDisplay = {'values' : self._meanFrame, 'title' : 'Mean frame (Time)'+' RMS='+self.rmsMean+' m/s^2'}
        cleanFrameDisplay = {'values' : self._cleanTimeFrame, 'title' : 'Cleaned signal frame (Time)'+' RMS='+self.rmsCleanSignal+' m/s^2'}
        self._displayTime({0:meanDisplay, 1: cleanFrameDisplay}, 'ms')

        origFreqDisplay = {'values' : self._originalDataFreq, 'title' : 'Original Signal (Freq)'}
        cleanFreqDisplay = {'values' : self._cleanDataFreq, 'title' : 'Cleaned Signal (Freq)'}
        self._displayFreq({0:origFreqDisplay, 1: cleanFreqDisplay})

        meanFreqFrameDisplay = {'values' : self._meanFrameFreq, 'title' : 'Mean Frame (Freq)'}
        cleanFreqFrameDisplay = {'values' : self._cleanTimeFrameFreq, 'title' : 'Cleaned signal Frame (Freq)'}
        self._displayFreq({0:meanFreqFrameDisplay, 1: cleanFreqFrameDisplay})

        cleanFreqFrame2Display = {'values' : self._cleanFreqFrame, 'title' : 'Cleaned Signal Stacked Frequency Frames'}
        self._displayFreq({0: cleanFreqFrame2Display})




    def processSignalFromFile(self, location):
        self._loadOriginal_File(location)
        self._calcMeanFrame()
        self._cleanSignal()
        self._stackCleanSignalFrames_Time_Freq()
        self._calcFreqSpectrums()

        self._rmsOriginal()
        self._rmsCleaned()
        self._rmsMeanF()
        self._rmsCleanedSF()
        self.displayAllData()

def execCalc(event):

    signal1 = Signal(inputTime.GetValue(), inputFrame.GetValue(), inputSkip.GetValue(), inputSkipFramesTime.GetValue(), inputFreMark.GetValue(), inputSingleRollTime.GetValue())
    signal1.processSignalFromFile('matavimai/'+ inputFile.GetValue())
    del signal1
    # Draw the plot to the screen
    plt.show()



app = wx.App(False)  # Create a new app, don't redirect stdout/stderr to a window.
frame = wx.Frame(None, wx.ID_ANY, title="Guoliu gedimai v0.6.7", size=(320, 370)) # A Frame is a top-level window.
frame.Show(True)     # Show the frame.
button = wx.Button(frame, label="Vykdyti", pos=(170, 270))
inputFile = wx.TextCtrl(frame,-1,pos=(180, 60), size=(110, 20), value=('m6.txt'))
inputTime = wx.TextCtrl(frame,-1,pos=(180, 90), size=(50, 20), value=('5'))
inputSkipFramesTime = wx.TextCtrl(frame,-1,pos=(180, 120), size=(50, 20), value=('25'))
inputFrame = wx.TextCtrl(frame,-1,pos=(180, 150), size=(50, 20), value=('1024'))
inputSingleRollTime = wx.TextCtrl(frame,-1,pos=(180, 180), size=(50, 20), value=('20'))
inputSkip = wx.TextCtrl(frame,-1,pos=(180, 210), size=(50, 20), value=('17'))
inputFreMark = wx.TextCtrl(frame,-1,pos=(180, 240), size=(50, 20), value=('0'))

label0 = wx.StaticText(frame, -1, 'Guoliu Gedimai v0.6.7' , pos=(30, 20))
font = wx.Font(16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
label0.SetFont(font)
label0.SetForegroundColour(wx.Colour(14,181,56));
frame.SetBackgroundColour(wx.Colour(225,225,225));
label1 = wx.StaticText(frame, -1, 'Pirmas matavimu failas' , pos=(15, 60))
label9 = wx.StaticText(frame, -1, 'Imtis (sekundes)' , pos=(15, 90))
label5 = wx.StaticText(frame, -1, 'Praleisti (sekundes)' , pos=(15, 120))
label3 = wx.StaticText(frame, -1, 'Tasku kiekis apsisukime' , pos=(15, 150))
label14 = wx.StaticText(frame, -1,'Apsisukimo trukme (ms)' , pos=(15, 180))
label2 = wx.StaticText(frame, -1, 'Praleisti eiluciu failuose' , pos=(15, 210))
label13 = wx.StaticText(frame, -1,'Atzyma dazniuose (Hz)', pos=(15, 240))
label10 = wx.StaticText(frame, -1,'Veiksmas', pos=(15, 270))
label12 = wx.StaticText(frame, -1,"Autorius: AurimasDGT", pos=(15, 300))
label12.SetForegroundColour(wx.Colour(173,88,88));

button.Bind(wx.EVT_BUTTON, execCalc)

app.MainLoop()