import glob
import librosa
import csv
import numpy as np 
import pandas as pd



SAMPLE_RATE = 44100
lf = librosa.feature

def getFeature(audio_path):
	features = []
	label = np.array([1])
	y, sr = librosa.load(audio_path,sr=SAMPLE_RATE)
	fe1 = lf.mfcc(y,sr=SAMPLE_RATE)
	fe2 = lf.zero_crossing_rate(y)[0]
	fe3 = lf.spectral_centroid(y)[0]
	fe1 = np.hstack((np.mean(fe1, axis=1)))
	fe2 = np.hstack((np.mean(fe2),np.std(fe2)))
	fe3 = np.hstack((np.mean(fe3),np.std(fe3)))
	features.append(np.hstack((fe1,fe2,fe3)))
	print(len(features[0]))
	return {"features": features[0]}

#print(getFeature('flute-c4.wav'))

