# import some libraries

import urllib
import scipy
import librosa
import librosa

import numpy as np
#from sklearn.decomposition import NMF


import numpy
import librosa, librosa.display
import warnings
warnings.simplefilter("ignore")
from scipy.signal import  correlate



import librosa, librosa.display
import warnings
warnings.simplefilter("ignore")

import scipy.sparse
import numpy.linalg as LA
from scipy.stats import entropy
from sys import exit
import pickle
#from __future__ import division
from numpy import polyfit, arange
from numpy import argmax, mean, diff, log, nonzero


class NMFBase():

	def __init__(self, X, rank=10, **kwargs):

		self.X = X
		self._rank = rank


		self.X_dim, self._samples = self.X.shape

	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			error = LA.norm(self.X - np.dot(self.W, self.H))
		else:
			error = None

		return error

	def kl_divergence(self):
		""" KL Divergence between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			V = np.dot(self.W, self.H)
			error = entropy(self.X, V).sum()
		else:
			error = None

		return error

	def initialize_w(self):
		""" Initalize W to random values [0,1]."""

		self.W = np.random.random((self.X_dim, self._rank))

	def initialize_h(self):
		""" Initalize H to random values [0,1]."""

		self.H = np.random.random((self._rank, self._samples))

	def update_h(self):
		"Override in subclasses"
		pass

	def update_w(self):
		"Override in subclasses"
		pass


	def check_non_negativity(self):

		if self.X.min()<0:
			return 0
		else:
			return 1

	def compute_factors(self, max_iter=100):
		if self.check_non_negativity():
			pass
		else:
			print("The given matrix contains negative values")
			exit()

		if not hasattr(self,'W'):
			self.initialize_w()

		if not hasattr(self,'H'):
			self.initialize_h()


		self.frob_error = np.zeros(max_iter)

		for i in range(max_iter):

			self.update_w()

			self.update_h()

			self.frob_error[i] = self.kl_divergence()

class NMF(NMFBase):


	def update_h(self):

		XtW = np.dot(self.W.T, self.X)
		HWtW = np.dot(self.W.T.dot(self.W), self.H ) + 2**-8
		self.H *= XtW
		self.H /= HWtW+30



	def update_w(self):

		XH = self.X.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
		self.W *= XH
		self.W /= WHtH

class NMFBaseIntW():

	def __init__(self, X,Wint, rank=10, **kwargs):

		self.X = X
		self._rank = rank
		self.W = Wint



		self.X_dim, self._samples = self.X.shape

	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			error = LA.norm(self.X - np.dot(self.W, self.H))
		else:
			error = None

		return error

	def kl_divergence(self):
		""" KL Divergence between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			V = np.dot(self.W, self.H)
			error = entropy(self.X, V).sum()
		else:
			error = None

		return error



	def initialize_h(self):
		""" Initalize H to random values [0,1]."""

		self.H = np.random.random((self._rank, self._samples))

	def update_h(self):
		"Override in subclasses"
		pass

	def update_w(self):
		"Override in subclasses"
		pass


	def check_non_negativity(self):

		if self.X.min()<0:
			return 0
		else:
			return 1

	def compute_factors(self, max_iter=100):
		if self.check_non_negativity():
			pass
		else:
			print("The given matrix contains negative values")
			exit()



		if not hasattr(self,'H'):
			self.initialize_h()


		self.frob_error = np.zeros(max_iter)

		for i in range(max_iter):



			self.update_h()

			self.frob_error[i] = self.kl_divergence()

class NMFInt(NMFBaseIntW):


	def update_h(self):

		XtW = np.dot(self.W.T, self.X)
		HWtW = np.dot(self.W.T.dot(self.W), self.H ) + 2**-8
		self.H *= XtW
		self.H /= HWtW+30

class NMFBaseIntWUpdtW():

	def __init__(self, X,Wint, rank=10, **kwargs):

		self.X = X
		self._rank = rank
		self.W = Wint



		self.X_dim, self._samples = self.X.shape

	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			error = LA.norm(self.X - np.dot(self.W, self.H))
		else:
			error = None

		return error

	def kl_divergence(self):
		""" KL Divergence between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			V = np.dot(self.W, self.H)
			error = entropy(self.X, V).sum()
		else:
			error = None

		return error



	def initialize_h(self):
		""" Initalize H to random values [0,1]."""

		self.H = np.random.random((self._rank, self._samples))

	def update_h(self):
		"Override in subclasses"
		pass

	def update_w(self):
		"Override in subclasses"
		pass


	def check_non_negativity(self):

		if self.X.min()<0:
			return 0
		else:
			return 1

	def compute_factors(self, max_iter=100):
		if self.check_non_negativity():
			pass
		else:
			print("The given matrix contains negative values")
			exit()



		if not hasattr(self,'H'):
			self.initialize_h()


		self.frob_error = np.zeros(max_iter)

		for i in range(max_iter):


			self.update_w()
			self.update_h()

			self.frob_error[i] = self.kl_divergence()

class NMFIntWUpdtW(NMFBaseIntWUpdtW):


	def update_h(self):

		XtW = np.dot(self.W.T, self.X)
		HWtW = np.dot(self.W.T.dot(self.W), self.H ) + 2**-8
		self.H *= XtW
		self.H /= HWtW+30



	def update_w(self):

		XH = self.X.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
		self.W *= XH
		self.W /= WHtH

def mix_frm_clean (x,length,noise):
  
  x = x[0:length]
  noise = noise[0:length] 
  mix = x + noise
  return mix , x

def spectorgram (x):
  X = abs(librosa.stft(x))
  magnitude, phase = librosa.magphase(X)
  
  return X , magnitude, phase

def plot_Spectogram (x,fs,title):
  V = librosa.stft(x)
  V_db = librosa.amplitude_to_db(abs(V))
  plt.figure(figsize=(14, 4))
  librosa.display.specshow(V_db, sr=fs, x_axis='time', y_axis='log',cmap='viridis')
  plt.colorbar()
  plt.title(title)

def train_Nmf(components , X):
  fpdnmf= NMF(X, rank=components)
  fpdnmf.compute_factors(6)
  W, H = fpdnmf.W , fpdnmf.H

  return W,H

def train_Custom_Nmf( X, W , H):

  fpdnmf= NMFIntWUpdtW(X,W, rank=W.shape[1])
  fpdnmf.compute_factors(6)
  W, H = fpdnmf.W , fpdnmf.H

  return W,H

def train_Custom_Nmf_No_UpdateW( X, W , H):

  fpdnmf= NMFInt(X,W, rank=W.shape[1])
  fpdnmf.compute_factors(6)
  W, H = fpdnmf.W , fpdnmf.H

  return W,H

def wiener(vocW,noiseW,Hmix,magnitudemix,phasemix):    
    magnitude_reconstructed_voc =np.matmul(vocW,Hmix[:vocW.shape[1],:])
    magnitude_reconstructed_noise = np.matmul(noiseW,Hmix[vocW.shape[1]:,:])
     
    #Gain function similar to wiener filter to enhance the speech signal
    wiener_gain_Voc = np.power(magnitude_reconstructed_voc,2) / (np.power(magnitude_reconstructed_voc,2) + np.power(magnitude_reconstructed_noise, 2))
    magnitude_estimated_clean = wiener_gain_Voc * magnitudemix

    #Reconstruct
    stft_reconstructed_clean = magnitude_estimated_clean * phasemix
    Vocal = librosa.istft(stft_reconstructed_clean)
    
    wiener_gain_Noise = np.power(magnitude_reconstructed_noise, 2) / (np.power(magnitude_reconstructed_voc,2) + np.power(magnitude_reconstructed_noise, 2))
    magnitude_estimated_Noise = wiener_gain_Noise * magnitudemix
    
    stft_reconstructed_Noise = magnitude_estimated_Noise * phasemix
    Noise = librosa.istft(stft_reconstructed_Noise)
    
    return Vocal , Noise

def sdr (voc,Vocal) :
  l = min(len(Vocal),len(voc))
  Vocal = Vocal[:l]
  voc = voc[:l]
  Vocl=np.reshape(Vocal,(l,1))

  vocl=np.reshape(voc,(l,1))
  normalized_voc = preprocessing.normalize(vocl)
  top = 1/normalized_voc.shape[0]*(sum(normalized_voc**2))
  
  normalized_Voc = preprocessing.normalize(Vocl)
  bottom = 1/normalized_Voc.shape[0]*(sum(normalized_Voc**2))

  Y = 10*math.log10(top/bottom)
  return Y

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    
    sd = a.std(axis = axis, ddof = ddof) 
    #print(m,"sd",sd)
    k = 10*math.log10(top/bottom)
    return k

def whmul(Wtan,Htan):
  T = np.matmul(Wtan,Htan)
  print (T.shape,Wtan.shape,Htan.shape)
  k = librosa.istft(T)
  return IPython.display.Audio(k,rate=fs)

def Trainint_Nmf(voc,noise):
  length = len(voc)
  noise = noise[0:length]
  mix , voc =  mix_frm_clean (voc,length,noise)
  vocSfft , vocMag , vocPhase = spectorgram (voc)
  #print (vocSfft.shape)
  noiseSfft , noiseMag , noisePhase = spectorgram (noise)
  #print (noiseSfft.shape)
  mixSfft , mixMag , mixPhase = spectorgram (mix)
  #print (mixSfft.shape)
  vocW , vocH = train_Nmf(20 ,vocSfft )
  #print (vocW.shape,vocH.shape)
  whmul(vocW,vocH)
  noiseW , noiseH = train_Nmf(10 ,noiseSfft )
  #print (noiseW.shape,noiseH.shape)
  whmul(noiseW,noiseH)
  
  WmixInt = np.concatenate((vocW, noiseW), axis=1)
  
  HmixInt = np.concatenate((vocH, noiseH), axis=0)
  
  HmixIntRand = np.random.rand(vocH.shape[0]+noiseH.shape[0],noiseH.shape[1])*0.0001
  
  Wmix,Hmix = train_Custom_Nmf_No_UpdateW(mixSfft, WmixInt , HmixInt)
  
  Vocal , Noise = wiener(vocW,noiseW ,Hmix,mixMag,mixPhase)
  
  return Vocal , Noise ,vocW , vocH, noiseW , noiseH

def Train_next_Nmf(voc,noise,mix, vocW , vocH, noiseW , noiseH):

  vocSfft , vocMag , vocPhase = spectorgram (voc)
 
  noiseSfft , noiseMag , noisePhase = spectorgram (noise)
  
  mixSfft , mixMag , mixPhase = spectorgram (mix)
  
  vocH = np.random.rand(vocW.shape[1],vocSfft.shape[1])*0.001
  noiseH = np.random.rand(noiseW.shape[1],noiseSfft.shape[1])*0.001
  
  vocW , vocH = train_Custom_Nmf(vocSfft, vocW , vocH)
  
  
  noiseW , noiseH = train_Custom_Nmf(noiseSfft,noiseW , noiseH)
  
  
  WmixInt = np.concatenate((vocW, noiseW), axis=1)
  
  HmixInt = np.concatenate((vocH, noiseH), axis=0)
  
  HmixIntRand = np.random.rand(vocH.shape[0]+noiseH.shape[0],noiseH.shape[1])*0.0001
  
  Wmix,Hmix = train_Custom_Nmf_No_UpdateW(mixSfft, WmixInt , HmixInt)
  
  Vocal , Noise = wiener(vocW,noiseW ,Hmix,mixMag,mixPhase)
  
  return Vocal , Noise ,vocW , vocH, noiseW , noiseH

def Test_Nmf(mix, vocW ,  noiseW ):

  
  mixSfft , mixMag , mixPhase = spectorgram (mix)
  
  vocH = np.random.rand(vocW.shape[1],mixSfft.shape[1])*0.001
  noiseH = np.random.rand(noiseW.shape[1],mixSfft.shape[1])*0.001
  
  
  WmixInt = np.concatenate((vocW, noiseW), axis=1)
  
  HmixInt = np.concatenate((vocH, noiseH), axis=0)
  
  HmixIntRand = np.random.rand(vocH.shape[0]+noiseH.shape[0],noiseH.shape[1])*0.0001
  
  Wmix,Hmix = train_Custom_Nmf_No_UpdateW(mixSfft, WmixInt , HmixIntRand)
  
  
  Vocal , Noise = wiener(vocW,noiseW ,Hmix,mixMag,mixPhase)
  
  return Vocal , Noise


def parabolic(f, x):

    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)



def freq_from_autocorr(sig, fs):

# Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]
# Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px




