import numpy as np
import math
import matplotlib.pyplot as plt
import re
import wx
#import threading

class Signal:
    _fileCol = 3
    _lastFigure = 0
    _rolls = 0
    _rmsOrig = ''
    _rmsClean = ''
    _rmsMean = ''
    _rmsCleanSignal = ''
    _frameSize = 0
    _rolls= 0
    _limit = 0
    _skip = 0
    _limit = 0
    _freMark = 0.0
    _displayParams = 'g'
    _originalData = []
    _originalDataFreq = []
    _cleanData = []
    _cleanDataFreq = []
    _meanFrame = []
    _meanFrameFreq = []
    _cleanTimeFrame = []
    _cleanFreqFrame = [] #this one is made when each clean frame is converted to Freq and stacked (Freq frames stacked)
    _cleanTimeFrameFreq = [] # this one is made when full cleaned signal time interval is converted to freq spectrum
    _SingleRollTime = 0;

    def __init__(self, time, frameSize, skip, skipFramesTime, freMark, singleRollTime):
        Signal._SingleRollTime = int(singleRollTime)
        Signal._rolls = self.convertToRolls(time)
        Signal._frameSize = int(frameSize)
        Signal._skip = int(skip)
        skipFrames = self.convertToRolls(skipFramesTime)
        Signal._skip = Signal._skip + (skipFrames * Signal._frameSize)
        Signal._limit = Signal._rolls * Signal._frameSize
        Signal._limit = Signal._skip + Signal._limit
        Signal._freMark = float(freMark)
        Signal._displayParams = Signal._displayParams

    def convertToRolls(self, time):
        msTime = int(float(time)) * 1000
        return int(float(msTime / Signal._SingleRollTime))

    def _loadOriginal_File(self, location):
        fileData1 = open(location, "r")
        dialog = wx.ProgressDialog('Skaiciuoja duomenis', 'Prasome palaukti', Signal._limit, style=wx.PD_REMAINING_TIME)

        for itera in range(Signal._limit):
            line = fileData1.readline()
            if itera < Signal._skip:
                continue
            dataRaw = re.split("\t", line)
            data = float(dataRaw[self._fileCol-1])
            Signal._originalData.append(data)
            dialog.Update(itera)
        wx.CallAfter(dialog.Destroy)

    def _calcMeanFrame(self):

        bufIter = 0
        buffer = [0] * Signal._frameSize
        originalData = Signal._originalData
        frameSize = Signal._frameSize

        for iter in range(len(originalData)-frameSize):
            buffer[bufIter] = (buffer[bufIter] + originalData[iter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                bufIter=0
        Signal._meanFrame = buffer



    def _cleanSignal(self):

        bufIter = 0
        originalData = Signal._originalData
        frameSize = Signal._frameSize
        cleanData=[0]* (len(originalData)-frameSize)
        meanFrame = Signal._meanFrame

        for itera in range(len(originalData)-frameSize):
            cleanData[itera] = (originalData[itera] - meanFrame[bufIter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                bufIter=0
        Signal._cleanData = cleanData

    def _stackCleanSignalFrames_Time_Freq(self):
        bufIter = 0
        frameSize = Signal._frameSize
        cleanTimeFrame = [0] * frameSize
        cleanFreqFrame = [0]*((frameSize/2)+1)
        cleanFreqFrameTemp = [0]*((frameSize/2)+1)
        cleanData = Signal._cleanData
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
        Signal._cleanTimeFrame = cleanTimeFrame
        Signal._cleanFreqFrame = cleanFreqFrame

    def _calcFreqSpectrums(self):

        Signal._originalDataFreq = abs(np.fft.rfft(Signal._originalData)) /100
        for iter2 in range(len(Signal._originalDataFreq)):
            if Signal._originalDataFreq[iter2] < 0:
                Signal._originalDataFreq[iter2] = Signal._originalDataFreq[iter2] * -1
        Signal._cleanDataFreq = abs(np.fft.rfft(Signal._cleanData))
        Signal._meanFrameFreq = abs(np.fft.rfft(Signal._meanFrame))
        Signal._cleanTimeFrameFreq = abs(np.fft.rfft(Signal._cleanTimeFrame))

        #JS pradzia

    def _rmsOriginal (self):
        sum = 0
        originalData = Signal._originalData
        for itera in range (len(originalData)):
            sum += ( math.pow(originalData[itera], 2))
        aver = sum / len(originalData)

        Signal.rmsOrig = str(math.sqrt(aver))

    def _rmsCleaned (self):
        sum = 0

        cleanData = Signal._cleanData
        for itera in range (len(cleanData)):
            sum += ( math.pow(cleanData[itera], 2))

        aver = sum / len(cleanData)

        Signal.rmsClean = str(math.sqrt(aver))

    def _rmsMeanF (self):
        sum = 0

        meanData = Signal._meanFrame

        for itera in range (len(meanData)):
            sum += ( math.pow(meanData[itera], 2))

        aver = sum / len(meanData)

        Signal.rmsMean = str(math.sqrt(aver))

    def _rmsCleanedSF (self):
        sum = 0
        number = 0
        kiekN = 0
        cleanTFData = Signal._cleanTimeFrame
        for itera in range (len(cleanTFData)):
            sum += (cleanTFData[itera]*(math.sin(2*3.14*50*0.02)))
            if itera % 1024 == 0:
                number += sum ** 2
                kiekN+=1
                sum = 0
        Signal.rmsCleanSignal = str((number / kiekN) ** (1.0/2))


    def _displayTime(self, data):
        Signal._lastFigure+=1
        plt.figure(Signal._lastFigure)
        subplots = 211
        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            plt.plot(range(0, len(data[iter]['values'])), data[iter]['values'], Signal._displayParams)
            plt.title(data[iter]['title'])
            subplots+=1


    def _displayFreq(self, data):
        Signal._lastFigure+=1
        xAxisMarkerValues = ['0', '5', '10', '15', '20', '25 kHz']
        plt.figure(Signal._lastFigure)
        subplots = 211

        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)
            Lenght = float(len(data[iter]['values']))
            localFreMark = (Lenght / 25000) * Signal._freMark
            plt.bar(localFreMark, max(data[iter]['values']), width=0.8)
            plt.plot(range(0, int(Lenght)), data[iter]['values'], Signal._displayParams)
            plt.title(data[iter]['title'])
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')

            subplots+=1


    def displayAllData(self):
        originalDisplay = {'values' : Signal._originalData, 'title' : 'Original signal (Time)'+' RMS='+Signal.rmsOrig+' m/s^2'}
        cleanDisplay = {'values' : Signal._cleanData, 'title' : 'Cleaned signal (Time)'+' RMS='+Signal.rmsClean+' m/s^2'}
        Signal._displayTime(self, {0:originalDisplay, 1: cleanDisplay})

        meanDisplay = {'values' : Signal._meanFrame, 'title' : 'Mean frame (Time)'+' RMS='+Signal.rmsMean+' m/s^2'}
        cleanFrameDisplay = {'values' : Signal._cleanTimeFrame, 'title' : 'Cleaned signal frame (Time)'+' RMS='+Signal.rmsCleanSignal+' m/s^2'}
        Signal._displayTime(self, {0:meanDisplay, 1: cleanFrameDisplay})

        origFreqDisplay = {'values' : Signal._originalDataFreq, 'title' : 'Original Signal (Freq)'}
        cleanFreqDisplay = {'values' : Signal._cleanDataFreq, 'title' : 'Cleaned Signal (Freq)'}
        Signal._displayFreq(self, {0:origFreqDisplay, 1: cleanFreqDisplay})

        meanFreqFrameDisplay = {'values' : Signal._meanFrameFreq, 'title' : 'Mean Frame (Freq)'}
        cleanFreqFrameDisplay = {'values' : Signal._cleanTimeFrameFreq, 'title' : 'Cleaned signal Frame (Freq)'}
        Signal._displayFreq(self, {0:meanFreqFrameDisplay, 1: cleanFreqFrameDisplay})

        cleanFreqFrame2Display = {'values' : Signal._cleanFreqFrame, 'title' : 'Cleaned Signal Stacked Frequency Frames'}
        Signal._displayFreq(self, {0: cleanFreqFrame2Display})




    def processSignalFromFile(self, location):
        Signal._loadOriginal_File(self, location)
        Signal._calcMeanFrame(self)
        Signal._cleanSignal(self)
        Signal._stackCleanSignalFrames_Time_Freq(self)
        Signal._calcFreqSpectrums(self)

        Signal._rmsOriginal(self)
        Signal._rmsCleaned(self)
        Signal._rmsMeanF(self)
        Signal._rmsCleanedSF(self)
        Signal.displayAllData(self)

def execCalc(event):

    signal1 = Signal(inputTime.GetValue(), inputFrame.GetValue(), inputSkip.GetValue(), inputSkipFramesTime.GetValue(), inputFreMark.GetValue(), inputSingleRollTime.GetValue())
    signal1.processSignalFromFile('matavimai/'+ inputFile.GetValue())

    # Draw the plot to the screen
    plt.show()



app = wx.App(False)  # Create a new app, don't redirect stdout/stderr to a window.
frame = wx.Frame(None, wx.ID_ANY, title="Guoliu gedimai v0.6.5", size=(320, 370)) # A Frame is a top-level window.
frame.Show(True)     # Show the frame.
button = wx.Button(frame, label="Vykdyti", pos=(170, 270))
inputFile = wx.TextCtrl(frame,-1,pos=(180, 60), size=(110, 20), value=('m6.txt'))
inputTime = wx.TextCtrl(frame,-1,pos=(180, 90), size=(50, 20), value=('5'))
inputSkipFramesTime = wx.TextCtrl(frame,-1,pos=(180, 120), size=(50, 20), value=('25'))
inputFrame = wx.TextCtrl(frame,-1,pos=(180, 150), size=(50, 20), value=('1024'))
inputSingleRollTime = wx.TextCtrl(frame,-1,pos=(180, 180), size=(50, 20), value=('20'))
inputSkip = wx.TextCtrl(frame,-1,pos=(180, 210), size=(50, 20), value=('17'))
inputFreMark = wx.TextCtrl(frame,-1,pos=(180, 240), size=(50, 20), value=('0'))

label0 = wx.StaticText(frame, -1, 'Guoliu Gedimai v0.6.5' , pos=(30, 20))
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