import glob
import librosa
import csv
import numpy as np 
import pandas as pd

directory = '../../Training Dataset/'

TFlute = glob.glob(directory+'Training/flu/*.wav')
TPiano = glob.glob(directory+'Training/pia/*.wav')
TTrum = glob.glob(directory+'Training/tru/*.wav')
TVio = glob.glob(directory+'Training/vio/*.wav')

features = []

SAMPLE_RATE = 44100
lf = librosa.feature

for i in range(len(TFlute)):
	label = np.array([1])
	y, sr = librosa.load(TFlute[i],sr=SAMPLE_RATE)
	fe1 = lf.mfcc(y,sr=SAMPLE_RATE)
	fe2 = lf.zero_crossing_rate(y)[0]
	fe3 = lf.spectral_centroid(y)[0]
	fe1 = np.hstack((np.mean(fe1, axis=1)))
	fe2 = np.hstack((np.mean(fe2),np.std(fe2)))
	fe3 = np.hstack((np.mean(fe3),np.std(fe3)))
	features.append(np.hstack(([1],fe1,fe2,fe3)))
print("done 1")
for i in range(len(TPiano)):
	label = np.array([2])
	y, sr = librosa.load(TPiano[i],sr=SAMPLE_RATE)
	fe1 = lf.mfcc(y,sr=SAMPLE_RATE)
	fe2 = lf.zero_crossing_rate(y)[0]
	fe3 = lf.spectral_centroid(y)[0]
	fe1 = np.hstack((np.mean(fe1, axis=1)))
	fe2 = np.hstack((np.mean(fe2),np.std(fe2)))
	fe3 = np.hstack((np.mean(fe3),np.std(fe3)))
	features.append(np.hstack(([2],fe1,fe2,fe3)))
print("done 2")
for i in range(len(TTrum)):
	label = np.array([3])
	y, sr = librosa.load(TTrum[i],sr=SAMPLE_RATE)
	fe1 = lf.mfcc(y,sr=SAMPLE_RATE)
	fe2 = lf.zero_crossing_rate(y)[0]
	fe3 = lf.spectral_centroid(y)[0]
	fe1 = np.hstack((np.mean(fe1, axis=1)))
	fe2 = np.hstack((np.mean(fe2),np.std(fe2)))
	fe3 = np.hstack((np.mean(fe3),np.std(fe3)))
	features.append(np.hstack(([3],fe1,fe2,fe3)))
print("done 3")
for i in range(len(TVio)):
	label = np.array([0])
	y, sr = librosa.load(TVio[i],sr=SAMPLE_RATE)
	fe1 = lf.mfcc(y,sr=SAMPLE_RATE)
	fe2 = lf.zero_crossing_rate(y)[0]
	fe3 = lf.spectral_centroid(y)[0]
	fe1 = np.hstack((np.mean(fe1, axis=1)))
	fe2 = np.hstack((np.mean(fe2),np.std(fe2)))
	fe3 = np.hstack((np.mean(fe3),np.std(fe3)))
	features.append(np.hstack(([0],fe1,fe2,fe3)))


print("done 4")
with open(directory+'Datasetnotes.csv', 'w') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
	ro = ['Class']
	for i in range(20):
		ro.append('mfcc_mean'+str(i+1))
	ro.append('ZCR_mean')
	ro.append('ZCR_std')
	ro.append('SC_mean')
	ro.append('SC_std')
	filewriter.writerow(ro)
	for i in range(len(features)):
		filewriter.writerow(features[i])

