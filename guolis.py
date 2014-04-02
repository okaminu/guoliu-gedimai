import numpy as np
from numpy import array, exp
import math
import matplotlib.pyplot as plt
import re
import wx
import copy
import os

# Vytauto testas
#import threading
class SignalParameters:

    FirstSignalColor = '#ff8000'
    SecondSignalColor = '#00c800'
    DifferenceSignalColor = '#8000ff'

    FirstSignalName = 'Pirmas'
    SecondSignalName = 'Antras'
    DifferenceSignalName = 'Skirtumas'


class Signal:

    signalNames = []
    signalData = []
    _lastTimeFigure = 0

    def initValues(self):
        self._fileCol = 0
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
        self._originalRMSData = []
        self._originalRMSSigma = []
        self._originalDataFreq = []
        self._cleanData = []
        self._cleanRMSData = []
        self._cleanRMSSigma = []
        self._cleanDataFreq = []
        self._meanFrame = []
        self._meanFrameFreq = []
        self._cleanTimeFrame = []
        self._cleanFreqFrame = [] #this one is made when each clean frame is converted to Freq and stacked (Freq frames stacked)
        self._cleanTimeFrameFreq = [] # this one is made when full cleaned signal time interval is converted to freq spectrum
        self._SingleRollTime = 0;
        self.isDrawLegend = 0
        self.rmsLocation = "statistics/RMS/"
        self.statisticsLocation = "statistics/"
        self.distanceLocation = "statistics/distance/"
        self.distanceTreshold = 0 #atstumas po kurio vel prades ieskoti kitos minimalios reiksmes
        self._hideFreq
        self.corrMode = 'full'
        self.freqSize = 25000
        self._hanningWindowSize = 0
        self._hanningAllow = 0
        self._coorLength = 0.01
        self._rollsOffset = 0.5

        self._originalDataCorr = []
        self._cleanDataCorr = []
        self._meanFrameCorr = []
        self._cleanTimeFrameCorr = []
        self._originalDataCeps = []
        self._cleanDataCeps = []
        self._meanFrameCeps = []
        self._cleanTimeFrameCeps = []
        self._exportImages = 0

    def __init__(self, range, frameSize, skip, freMark, rpm, isRangeTime, isSkipTime, hideFreq, hannSize, allowHanning, coorLength = 1, distanceTreshold = 100, fileCol = 0, isHalfRoll = 0):
        self._hideFreq = int(hideFreq)
        self.initValues()
        self._fileCol = int(fileCol)
        self.distanceTreshold = int(distanceTreshold)
        self._coorLength = float(coorLength) /100
        self._hanningAllow = int(allowHanning)
        self._hanningWindowSize = int(hannSize)
        self._SingleRollTime = int(1/float(int(rpm)/60)*1000)
        self._rolls = int(range)
        if(isRangeTime == 1):
            self._rolls = self.convertTimeToRolls(range)
        self._frameSize = int(frameSize)
        skipFrames = int(skip)
        if(isSkipTime == 1):
            skipFrames = self.convertTimeToRolls(skip)
        self._skip = self._skip + (skipFrames * self._frameSize)
        if(isHalfRoll == 1):
            self._skip = self._skip + int(self._rollsOffset * self._frameSize)
        self._limit = self._rolls * self._frameSize
        self._limit = self._skip + self._limit
        self._freMark = float(freMark)
        self._displayParams = self._displayParams

    # this is wrong and dirty on so many levels...
    @staticmethod
    def clearClassVariables():
        Signal.signalData = []
        Signal.signalNames = []

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
        frameSize = self._frameSize
        originalData = self._originalData
        size = len(originalData)
        if(size == 0):
            size = self._frameSize

        buffer = [0] * self._frameSize

        for iter in range(size):
            buffer[bufIter] = (buffer[bufIter] + originalData[iter]) / 2
            bufIter = bufIter+1
            if bufIter == frameSize:
                bufIter=0
        self._meanFrame = buffer


    def _cleanSignal(self):

        bufIter = 0
        originalData = self._originalData
        frameSize = self._frameSize
        size = len(originalData)
        meanFrame = self._meanFrame
        if(size == 0):
            size = self._frameSize

        cleanData=[0]* size


        for itera in range(size):
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

    def calculateHanningWindow(self, count):
        result= []
        for i in range(0, count, 1):
            result.append(0.5 * (1- np.cos(((2*3.14)*i)/count)))
        return np.array(result)

    def filterHanningWindowFrame(self, signal):
        han = self.calculateHanningWindow(len(signal))
        filtered = signal*han
        return filtered

    def filterHanningWindow(self, signal, windowSize, totalSize):
        size = int((len(signal) * windowSize) / totalSize)

        final = []

        if(size <= 0):
            return signal

        for start in range(len(signal)):
            final.append(0)
            sigSlice = signal[start:start+size]
            #
            #
            filterSlice = self.filterHanningWindowFrame(sigSlice)

            for i in range(len(filterSlice)):
                final[start] = final[start] + filterSlice[i]
            #
            #
            final[start] = final[start] / len(filterSlice)


        return final

    def _calcFreqSpectrums(self):

        self._originalDataFreq = abs(np.fft.rfft(self._originalData)) / 555.555555
        for iter2 in range(len(self._originalDataFreq)):
            if self._originalDataFreq[iter2] < 0:
                self._originalDataFreq[iter2] = self._originalDataFreq[iter2] * -1
        self._cleanDataFreq = abs(np.fft.rfft(self._cleanData)) / 555.555555
        self._meanFrameFreq = abs(np.fft.rfft(self._meanFrame))
        self._cleanTimeFrameFreq = abs(np.fft.rfft(self._cleanTimeFrame))


    def _calcHanningWindow(self):
        if(self._hanningAllow == 1):
            self._originalDataFreq = self.filterHanningWindow(self._originalDataFreq, self._hanningWindowSize, self.freqSize)
            self._cleanDataFreq = self.filterHanningWindow(self._cleanDataFreq, self._hanningWindowSize, self.freqSize)
        self._meanFrameFreq = self.filterHanningWindow(self._meanFrameFreq, self._hanningWindowSize, self.freqSize)
        self._cleanTimeFrameFreq = self.filterHanningWindow(self._cleanTimeFrameFreq, self._hanningWindowSize, self.freqSize)
        self._cleanFreqFrame = self.filterHanningWindow(self._cleanFreqFrame, self._hanningWindowSize, self.freqSize)
        return 0


    def getSingleCoorelation(self, signal1, signal2):
        fullCoorelation = np.correlate(signal1, signal2, mode='full')
        fullCoorelation = fullCoorelation / np.max(fullCoorelation)
        size = len(fullCoorelation)
        halfCoorelation = fullCoorelation[size / 2:]
        partialCoorelation = halfCoorelation[:(size / 2) * self._coorLength]
        return partialCoorelation

    def calcCorrelation(self, signal):
        self._originalDataCorr = self.getSingleCoorelation(signal._originalData, self._originalData)
        self._cleanDataCorr = self.getSingleCoorelation(signal._cleanData, self._cleanData)
        self._meanFrameCorr = self.getSingleCoorelation(signal._meanFrame, self._meanFrame)
        self._cleanTimeFrameCorr = self.getSingleCoorelation(signal._cleanTimeFrame, self._cleanTimeFrame)

    def _calcCepstrums(self):
        self._originalDataCeps = self.cepstrum(self._originalData)
        self._cleanDataCeps = self.cepstrum(self._cleanData)
        self._meanFrameCeps = self.cepstrum(self._meanFrame)
        self._cleanTimeFrameCeps = self.cepstrum(self._cleanTimeFrame)

    def cepstrum(self, signal):
        # arr = []
        # for i in range(0, 360):
        #     arr.append(math.sin(math.radians(i)))
        # temp = abs(np.fft.fft(signal))
        # for index, item in enumerate(temp):
        #     if(temp[index] == 0):
        #         temp[index] = 0.00001
        #
        # cepstrum = np.fft.ifft(np.log(temp))
        # return cepstrum

        table = signal
        pad_size = 0
        table = list(table)
        # table should be a real-valued table of FIR coefficients
        convolution_size = len(table)
        table += [0] * (convolution_size * (pad_size - 1))

        # compute the real cepstrum
        # fft -> abs + ln -> ifft -> real
        cepstrum = np.fft.ifft(map(lambda x: math.log(x), abs(np.fft.fft(table))))
        # because the positive and negative freqs were equal, imaginary content is neglible
        # cepstrum = map(lambda x: x.real, cepstrum)

        # window the cepstrum in such a way that anticausal components become rejected
        cepstrum[1                :len(cepstrum)/2] *= 2;
        cepstrum[len(cepstrum)/2+1:len(cepstrum)  ] *= 0;

        # now cancel the previous steps:
        # fft -> exp -> ifft -> real
        cepstrum = np.fft.ifft(map(exp, np.fft.fft(cepstrum)))
        return map(lambda x: x.real, cepstrum[0:convolution_size])

    def _calcDispersion(self, data):
        disp = 0
        temp = []

        for y in data:
            temp.append(float(y))

        mean = np.mean(temp)
        for x in temp:
            disp = disp + np.power(x - mean, 2)
        return round(disp / len(temp), 7)

    def _calcStandardDeviation(self, data):
        variance = self._calcDispersion(data)
        return round(np.sqrt(variance), 7)

    def _calcSigmaForEachPoint(self, data):
        disp = 0
        temp = []
        sigmaList = []

        for y in data:
            temp.append(float(y))

        mean = np.mean(temp)
        for x in temp:
            disp = np.power(x - mean, 2)
            variance = round(disp / len(temp), 7)
            sigma = round(np.sqrt(variance), 7)
            sigmaList.append(sigma)
        return sigmaList


    def _calcRms(self, data):
        sum=0
        for itera in range(len(data)):
            sum += math.pow(data[itera], 2)
        aver = sum / len(data)
        return str(math.sqrt(aver))

    def _rmsOriginal (self):
        return self._calcRms(self._originalData)

    def _rmsCleaned (self):
        return self._calcRms(self._cleanData)

    def _rmsMeanF (self):
        return self._calcRms(self._meanFrame)

    def _rmsCleanedSF (self):
        return self._calcRms(self._cleanTimeFrame)

    def calcRmsForEachRoll(self, signal):
        rmsInterval = []
        for itera in range(len(signal)):
            if (((itera+1) % self._frameSize) == 0) and (itera != 0):
                sliceOfData = signal[((itera+1) - self._frameSize):itera+1]
                rms = self._calcRms(sliceOfData)
                rmsInterval.append(rms)
        return rmsInterval

    def _calcRMSForTime(self):
        self._originalRMSData = self.calcRmsForEachRoll(self._originalData)
        self._cleanRMSData = self.calcRmsForEachRoll(self._cleanData)

    def _calcSigmaForRMS(self):
        self._originalRMSSigma = self._calcSigmaForEachPoint(self._originalRMSData)

    def _saveRMS(self, fileName):

        if(os.path.exists(self.rmsLocation) == 0):
            os.makedirs(self.rmsLocation)

        file = open(self.rmsLocation+fileName+".txt", 'w')
        file.writelines("Originalus: "+ self._rmsOriginal() +"\n")
        file.writelines("Centruotas: "+ self._rmsCleaned() +"\n")
        file.writelines("Originalo vidurkintas: "+ self._rmsMeanF() +"\n")
        file.writelines("Centruoto vidurkintas: "+ self._rmsCleanedSF() +"\n")

    def saveMaxFrequencyDistance(self, signalName):
        self.saveSignalMaxDistance('Originalus (Daznis)', self._originalDataFreq)
        self.saveSignalMaxDistance('Centruotas (Laikas)', self._cleanDataFreq)
        self.saveSignalMaxDistance('Originalo vidurkis (Daznis)', self._meanFrameFreq)
        self.saveSignalMaxDistance('Centruoto vidurkis (Daznis)', self._cleanTimeFrameFreq)
        self.saveSignalMaxDistance('Centruoto Signalo dazniu vidurkis', self._cleanFreqFrame)

    def saveSignalMaxDistance(self, signalName, signalData):
        originalFreqMax = self.twoMax(signalData)
        Length = float(len(signalData))
        fraction = (self.freqSize / Length)
        firstFreqIndex = originalFreqMax['firstIndex'] * fraction
        secondFreqIndex = originalFreqMax['secondIndex'] * fraction
        data = ['--- '+ signalName +' ---',
                'Didziausia: '+str(originalFreqMax['first']),
                'Antra didziausia:'+str(originalFreqMax['second']),
                'Pirmas Indeksas:'+str(firstFreqIndex),
                'Antras Indeksas:'+str(secondFreqIndex),
                'Atstumas:'+ str(abs(firstFreqIndex - secondFreqIndex))]
        self._saveData(self.distanceLocation, signalName, {'values':data})

    def twoMax(self, numbers):
        m1, m2 = None, None
        m1Index, m2Index, count = 0, 0, 0
        Length = float(len(numbers))
        fraction = (Length / self.freqSize)
        distanceExact = int(self.distanceTreshold * (Length / self.freqSize))
        skipCount = distanceExact
        skip = skipCount
        for val in numbers:
            if skip > skipCount:
                if (val) > m1 and (numbers[count-1]) < val and numbers[count+1] < val:
                    m2 = m1
                    m1 = val
                    m2Index = m1Index
                    m1Index = count
                    skip = 0
                elif (val) > m2 and (numbers[count-1]) < val and numbers[count+1] < val:
                    m2 = val
                    m2Index = count
            else:
                skip += 1
            count += 1

        return {'first': m1, 'second': m2, 'firstIndex': m1Index, 'secondIndex': m2Index}

    def _saveData(self, directory, fileName, data):

        if(os.path.exists(directory) == 0):
            os.makedirs(directory)

        file = open(directory+fileName+".txt", 'w')
        for iter in range(len(data['values'])):
            file.writelines(str(data['values'][iter]) +"\n")


    def _displayRMSSigma(self, data, displayParams, marking , grid, persist):
        self._lastFigure +=1
        self._drawDisplayTime(data, displayParams, self._lastFigure, marking, grid, persist, 'o')

    def _displayTime(self, data, displayParams, marking , grid, persist):
        Signal._lastTimeFigure+=1
        self._drawDisplayTime(data, displayParams, Signal._lastTimeFigure, marking, grid, persist)

    def _displayCepst(self, data, displayParams, marking , grid):
        self._lastFigure +=1
        self._drawDisplayTime(data, displayParams, self._lastFigure, marking, grid)

    def _drawDisplayTime(self, data, displayParams, figure, marking = 's', grid = 1, preserveFirstXAxis = 0, markerSimbol = ' '):
        plt.figure(figure)
        subplots = 211
        xAxisMarkerValues = []
        for iter in range(len(data)):
            if(len(data) > 1):
                plt.subplot(subplots)

            subplots+=1

            plt.title(data[iter]['title'])

            Lenght = float(len(data[iter]['values']))
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]

            lineStyle = '-'
            if preserveFirstXAxis == 0 or len(xAxisMarkerValues) == 0:
                xAxisMarkerValues = []
                for marker in range(len(xAxisMarkerPlacement)):
                    rolls = self.convertPointsToRolls(xAxisMarkerPlacement[marker])
                    if(marking == 's'):
                        time = round(float(self.convertRollsToTime(rolls)), 2)
                    elif(marking == 'ms'):
                        time = round(self.convertRollsToTime(rolls),2)
                    xAxisMarkerValues.append(time)
                xAxisMarkerValues[len(xAxisMarkerValues) -1] = str(xAxisMarkerValues[len(xAxisMarkerValues) -1]) + " s"
            else:
                size = float(max(data[iter]['values']))
                average = sum(float(value) for value in data[iter]['values'])/len(data[iter]['values']);
                plt.barh(average,Lenght, align='center', height=0.000001, linewidth = 1, edgecolor = 'b', color= 'b')
                markerSimbol = 'o'
                lineStyle = 'None'
                data[iter]['values'].insert(0, 0)

            plt.plot(range(0, len(data[iter]['values'])), data[iter]['values'], displayParams, marker = markerSimbol, linestyle = lineStyle)
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')

            Lenght = float(len(data[iter]['values']))

        if self._exportImages == 1:
            plt.savefig('image/'+str(figure)+' - '+ data[iter]['title']+'.png')

        if(grid==1):
                gridPart = (Lenght/8)
                gridMark = [gridPart * 1, gridPart * 2, gridPart * 3, gridPart * 4, gridPart * 5, gridPart * 6, gridPart * 7, gridPart * 8]
                for i in range(len(gridMark)):
                    height = abs(max(data[iter]['values'])) + abs(min(data[iter]['values']))
                    plt.bar(gridMark[i], height, width=1, edgecolor = '#000000', bottom= min(data[iter]['values']))
        if(self.isDrawLegend == 1):
            self.drawLegend()

    def _displayFreq(self, data, displayParams):
        self._lastFigure +=1
        xAxisMarkerValues = ['0', '3', '6', '9', '12', '15 kHz']
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

            #Max display
            freqMax = self.twoMax(data[iter]['values'])
            # plt.bar(freqMax['firstIndex'], max(data[iter]['values']), width=0.8, edgecolor = '#0066FF')
            # plt.bar(freqMax['secondIndex'], max(data[iter]['values']), width=0.8, edgecolor = '#0066FF')

            plt.plot(range(0, int(Lenght)), data[iter]['values'], displayParams)
            plt.title(data[iter]['title'])
            xAxisMarkerPlacement = [Lenght * 0, Lenght * 0.2, Lenght * 0.4, Lenght * 0.6, Lenght * 0.8, Lenght * 1]
            plt.xticks(xAxisMarkerPlacement, xAxisMarkerValues, ha='center')
            subplots+=1

        if self._exportImages == 1:
            plt.savefig('image/'+str(self._lastFigure)+' - '+ data[iter]['title']+'.png')

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

        if self._exportImages == 1:
            plt.savefig('image/'+str(self._lastFigure)+' - '+ data[iter]['title']+'.png')

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
        originalDisplay = {'values' : self._originalData, 'title' : 'Originalus (Laikas), RMS = '+ self._rmsOriginal()}
        originalRMSDisplay = {'values' : self._originalRMSData,
                              'title' : 'Originalus (Laikas) RMS, Dispersija = '+ str(self._calcDispersion(self._originalRMSData))
                                +', Sigma = '+ str(self._calcStandardDeviation(self._originalRMSData))}
        self._displayTime({0:originalDisplay, 1: originalRMSDisplay}, displayParams, 's', 0, 1)

        cleanDisplay = {'values' : self._cleanData, 'title' : 'Centruotas (Laikas), RMS = ' + self._rmsCleaned()}
        cleanRMSDisplay = {'values' : self._cleanRMSData,
                           'title' : 'Centruotas (Laikas) RMS, Dispersija = '+ str(self._calcDispersion(self._cleanRMSData))
                            +', Sigma = '+ str(self._calcStandardDeviation(self._cleanRMSData))}
        self._displayTime({0:cleanDisplay, 1: cleanRMSDisplay}, displayParams, 's', 0, 1)

        meanDisplay = {'values' : self._meanFrame, 'title' : 'Originalo vidurkis (Laikas), RMS = '+ self._rmsMeanF()}
        cleanFrameDisplay = {'values' : self._cleanTimeFrame, 'title' : 'Centruoto vidurkis (Laikas), RMS = '+ self._rmsCleanedSF()}
        self._displayTime({0:meanDisplay, 1: cleanFrameDisplay}, displayParams,'ms', 0, 0)


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
        self._displayCepst({0:originalCepstDisplay, 1: cleanCepstDisplay}, displayParams, 's', 0)

        meanCepstDisplay = {'values' : self._meanFrameCeps, 'title' : 'Originalo vidurkis (Laikas) Kepstras'}
        cleanFrameCepstDisplay = {'values' : self._cleanTimeFrameCeps, 'title' : 'Centruoto vidurkis (Laikas) Kepstras'}
        self._displayCepst({0:meanCepstDisplay, 1: cleanFrameCepstDisplay}, displayParams,'ms', 0)

    def displayAllCorrelations(self, displayParams):
        originalCorrDisplay = {'values' : self._originalDataCorr, 'title' : 'Originalus (Laikas) Koreliacija'}
        cleanCorrDisplay = {'values' : self._cleanDataCorr, 'title' : 'Centruotas (Laikas) Koreliacija'}
        self._displayCorr({0:originalCorrDisplay, 1: cleanCorrDisplay}, displayParams)

        meanCorrDisplay = {'values' : self._meanFrameCorr, 'title' : 'Originalo vidurkis (Laikas) Koreliacija'}
        cleanFrameCorrDisplay = {'values' : self._cleanTimeFrameCorr, 'title' : 'Centruoto vidurkis (Laikas) Koreliacija'}
        self._displayCorr({0:meanCorrDisplay, 1: cleanFrameCorrDisplay}, displayParams)

    def displayAllSigmas(self, displayParams):
        rmsSigmaDisplay = {'values' : self._originalRMSSigma, 'title' : 'Originalus (Laikas) RMS\'o Sigma'}
        self._displayRMSSigma({0:rmsSigmaDisplay}, displayParams,'ms', 0, 0)

    def displayAllData(self, color, signalName, isCorr = 0):
        self._saveRMS(signalName)

        displayParams = color
        self._lastFigure = Signal._lastTimeFigure
        # self.displayAllTime(displayParams)

        self.displayAllSigmas(displayParams)

        self.displayAllFreq(displayParams)

        self.displayAllCepstrums(displayParams)

        if(isCorr == 1):
            self.displayAllCorrelations(displayParams)

        self._lastFigure = Signal._lastTimeFigure

    def substract(self, num1, num2):
        return abs(num1) - abs(num2)

    def _substractSignals(self, signal2):
        leng = min({len(signal2._originalData), len(self._originalData)})
        for iter1 in range(leng):
            self._originalData[iter1] = self.substract(self._originalData[iter1], signal2._originalData[iter1])
        leng2 = min({len(signal2._cleanData), len(self._cleanData)})
        for iter2 in range(leng2):
            self._cleanData[iter2] = self.substract(self._cleanData[iter2], signal2._cleanData[iter2])
        leng3 = min({len(signal2._meanFrame), len(self._meanFrame)})
        for iter4 in range(leng3):
            self._meanFrame[iter4] = self.substract(self._meanFrame[iter4], signal2._meanFrame[iter4])
        leng4 = min({len(signal2._cleanTimeFrame), len(self._cleanTimeFrame)})
        for iter5 in range(leng4):
            self._cleanTimeFrame[iter5] = self.substract(self._cleanTimeFrame[iter5], signal2._cleanTimeFrame[iter5])
        leng5 = min({len(signal2._originalDataFreq), len(self._originalDataFreq)})
        for iter6 in range(leng5):
            self._originalDataFreq[iter6] = self.substract(self._originalDataFreq[iter6],
                                                           signal2._originalDataFreq[iter6])
        leng6 = min({len(signal2._cleanDataFreq), len(self._cleanDataFreq)})
        for iter7 in range(leng6):
            self._cleanDataFreq[iter7] = self.substract(self._cleanDataFreq[iter7], signal2._cleanDataFreq[iter7])
        leng7 = min({len(signal2._meanFrameFreq), len(self._meanFrameFreq)})
        for iter8 in range(leng7):
            self._meanFrameFreq[iter8] = self.substract(self._meanFrameFreq[iter8], signal2._meanFrameFreq[iter8])
        leng8 = min({len(signal2._cleanTimeFrame), len(self._cleanTimeFrameFreq)})
        for iter9 in range(leng8):
            self._cleanTimeFrameFreq[iter9] = self.substract(self._cleanTimeFrameFreq[iter9],
                                                             signal2._cleanTimeFrameFreq[iter9])
        leng9 = min({len(signal2._cleanFreqFrame), len(self._cleanFreqFrame)})
        for iter10 in range(leng9):
            self._cleanFreqFrame[iter10] = self.substract(self._cleanFreqFrame[iter10], signal2._cleanFreqFrame[iter10])

    def substractFrom(self, signal2):
        self._substractSignals(signal2)
        self._calcCepstrums()

    def processSignalFromFile(self, location):
        self._loadOriginal_File(location)
        self._alterTimeSignal()
        self._calcMeanFrame()
        self._cleanSignal()
        self._stackCleanSignalFrames_Time_Freq()
        self._calcFreqSpectrums()
        self._calcCepstrums()
        self._calcHanningWindow()
        self._calcRMSForTime()
        self._calcSigmaForRMS()

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
            inputRPM.GetValue(),
            rangeTime,
            skipTime,
            inputHideFreq.GetValue(),
            inputHanningSize.GetValue(),
            inputHanningAllow.GetValue(),
            inputCoorelLength.GetValue(),
            inputMaxTreshold.GetValue(),
            inputColumnSignal1.GetValue(),
            0
        )
    signal1.processSignalFromFile('matavimai/'+ inputFile.GetValue())
    signal1.addToLegend('Rezonansas', '#CCCCCC')

    signal1.setDrawLegend(1)
    signal1.displayAllTime(SignalParameters().FirstSignalColor)
    corrSignal = signal1
    if(inputFile2.GetValue() != ''):
        signal2 = Signal(
            inputRange.GetValue(),
            inputFrame.GetValue(),
            inputSkip.GetValue(),
            inputFreMark.GetValue(),
            inputRPM.GetValue(),
            rangeTime,
            skipTime,
            inputHideFreq.GetValue(),
            inputHanningSize.GetValue(),
            inputHanningAllow.GetValue(),
            inputCoorelLength.GetValue(),
            inputMaxTreshold.GetValue(),
            inputColumnSignal2.GetValue(),
            inputIsHalfRoll.GetValue()
        )

        signal2.processSignalFromFile('matavimai/'+ inputFile2.GetValue())
        signal1.addToLegend(SignalParameters().FirstSignalName, SignalParameters().FirstSignalColor)
        signal1.addToLegend(SignalParameters().SecondSignalName, SignalParameters().SecondSignalColor)
        signal1.addToLegend(SignalParameters().DifferenceSignalName, SignalParameters().DifferenceSignalColor)
        signalDiff = copy.deepcopy(signal1)
        signalDiff.substractFrom(signal2)
        signalDiff.calcCorrelation(signalDiff)
        signalDiff._calcRMSForTime()
        signalDiff._calcSigmaForRMS()

        signal2.setDrawLegend(1)
        signalDiff.setDrawLegend(1)
        signal2.displayAllTime(SignalParameters().SecondSignalColor)
        signalDiff.displayAllTime(SignalParameters().DifferenceSignalColor)

        signalDiff.displayAllData(SignalParameters().DifferenceSignalColor, SignalParameters().DifferenceSignalName)
        signal2.displayAllData(SignalParameters().SecondSignalColor, SignalParameters().SecondSignalName)
        signal2.saveMaxFrequencyDistance(SignalParameters().SecondSignalName)
        corrSignal = copy.deepcopy(signal2)
        del signalDiff
        del signal2

    signal1.calcCorrelation(corrSignal)
    signal1.displayAllData(SignalParameters().FirstSignalColor, SignalParameters().FirstSignalName, 1)
    signal1.saveMaxFrequencyDistance(SignalParameters().FirstSignalName);
    del signal1
    Signal.clearClassVariables()
    # Draw the plot to the screen
    plt.show()



appTitle = 'Guoliu Gedimai 1.6'
app = wx.App(False)  # Create a new app, don't redirect stdout/stderr to a window.
frame = wx.Frame(None, wx.ID_ANY, title=appTitle, size=(450, 560)) # A Frame is a top-level window.
frame.Show(True)     # Show the frame.
button = wx.Button(frame, label="Vykdyti", pos=(170, 460))
inputFile = wx.TextCtrl(frame,-1,pos=(200, 60), size=(110, 20), value=('m06s13.txt'))
inputFile2 = wx.TextCtrl(frame,-1,pos=(200, 90), size=(110, 20), value=(''))
inputSkip = wx.TextCtrl(frame,-1,pos=(200, 150), size=(50, 20), value=('50'))
inputRange = wx.TextCtrl(frame,-1,pos=(200, 180), size=(50, 20), value=('10'))
isSkipTime = wx.RadioButton(frame,label = 'sekundes',pos=(360, 150), style=wx.RB_GROUP)
isSkipPoint = wx.RadioButton(frame,label = 'apsisukimai',pos=(260, 150))
isSkipPoint.SetValue(1)
isInputTime = wx.RadioButton(frame,label = 'sekundes',pos=(360, 180), style=wx.RB_GROUP)
isInputPoint = wx.RadioButton(frame,label = 'apsisukimai',pos=(260, 180))
isInputPoint.SetValue(1)
inputFrame = wx.TextCtrl(frame,-1,pos=(200, 210), size=(50, 20), value=('1024'))
inputRPM = wx.TextCtrl(frame,-1,pos=(200, 240), size=(50, 20), value=('1800'))
inputFreMark = wx.TextCtrl(frame,-1,pos=(200, 270), size=(50, 20), value=('0'))
inputHideFreq = wx.TextCtrl(frame,-1,pos=(200, 300), size=(50, 20), value=('0'))
inputHanningSize = wx.TextCtrl(frame,-1,pos=(200, 330), size=(50, 20), value=('0'))
inputHanningAllow = wx.CheckBox(frame,-1,pos=(235, 360), size=(50, 20))
inputCoorelLength = wx.TextCtrl(frame,-1,pos=(200, 390), size=(50, 20), value=('25'))
inputMaxTreshold = wx.TextCtrl(frame,-1,pos=(200, 420), size=(50, 20), value=('500'))
inputIsHalfRoll = wx.CheckBox(frame,-1,pos=(200, 120), size=(50, 20))
inputIsHalfRoll.SetValue(0)

inputColumnSignal1 = wx.TextCtrl(frame,-1,pos=(320, 60), size=(50, 20), value=('1'))
inputColumnSignal2 = wx.TextCtrl(frame,-1,pos=(320, 90), size=(50, 20), value=('1'))
inputHanningAllow.SetValue(0)

label0 = wx.StaticText(frame, -1, appTitle , pos=(30, 20))
font = wx.Font(16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
label0.SetFont(font)
label0.SetForegroundColour(wx.Colour(14,181,56))
frame.SetBackgroundColour(wx.Colour(225,225,225))
label1 = wx.StaticText(frame, -1, 'Pirmas matavimu failas' , pos=(15, 60))
label6 = wx.StaticText(frame, -1, 'Antras matavimu failas' , pos=(15, 90))
label19 = wx.StaticText(frame, -1,"Ivertinti ritinio vieta", pos=(15, 120))
label5 = wx.StaticText(frame, -1, 'Praleisti' , pos=(15, 150))
label9 = wx.StaticText(frame, -1, 'Imtis' , pos=(15, 180))
label3 = wx.StaticText(frame, -1, 'Tasku kiekis apsisukime' , pos=(15, 210))
label14 = wx.StaticText(frame, -1,'Veleno greitis (RPM)' , pos=(15, 240))
label13 = wx.StaticText(frame, -1,'Rezonansas (Hz)', pos=(15, 270))
label2 = wx.StaticText(frame, -1,'Slept pirmus daznius (Hz)', pos=(15, 300))
label4 = wx.StaticText(frame, -1,'Haningo lango dydis (Hz)', pos=(15, 330))
label4 = wx.StaticText(frame, -1,'Haningo langas pirminiui signalui', pos=(15, 360))
label15 = wx.StaticText(frame, -1,'Koreliacijos ilgis (%)', pos=(15, 390))
label16 = wx.StaticText(frame, -1,'Maksimumu nuolydis (Hz)', pos=(15, 420))
label12 = wx.StaticText(frame, -1,"Autorius: AurimasDGT", pos=(15, 510))
label17 = wx.StaticText(frame, -1,"stulpelis", pos=(380, 60))
label18 = wx.StaticText(frame, -1,"stulpelis", pos=(380, 90))

label12.SetForegroundColour(wx.Colour(173,88,88));

button.Bind(wx.EVT_BUTTON, execCalc)

app.MainLoop()