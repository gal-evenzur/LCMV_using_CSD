import matplotlib.pyplot as plt
import soundfile as sf
import rir_generator as rir


c = 340;                     #Sound velocity (m/s)
fs = 8000;                  #Sample frequency (samples/s)
r = [2,1.5,2];            #Receiver position [ x y z ] (m)
s = [ 2 ,3.5,2 ];             #Source position [ x y z ] (m)
L = [ 5 ,4 ,6 ];               #Room dimensions [ x y z ] (m)
beta = 0.5;                #Reverberationtime (s)
nsample = 4096;              #Number of samples
mtype = 'hypercardioid';   #Type of microphone
order = -1;                  # −1 equals maximum reflection order!
dim = 3;                     # Room dimension
# orientation = [pi/2 0];      # Microphone orientation [azimuth elevation] in radians
hp_filter = 1;               # Enable high-pass filter


h,fs = sf.read('C:\project\h.wav')
h2 = rir.generate(c=340, fs=fs, r=r,s=s,L=L,reverberation_time=beta,nsample=4096,mtype=b"h").squeeze()
fig = plt.figure()
plt.plot(abs(h-h2))
fig.savefig(r'C:\project\fig.png')

print()

