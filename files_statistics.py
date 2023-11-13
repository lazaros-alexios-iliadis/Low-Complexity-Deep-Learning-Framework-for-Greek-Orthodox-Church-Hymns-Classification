import glob
import librosa
import numpy
import pandas as pd

paths = glob.glob('AudioDataset/hymn23/*.wav')
durations = [librosa.get_duration(filename=p) for p in paths]

stats = {
    'mean': numpy.mean(durations),
    'max': numpy.max(durations),
    'min': numpy.min(durations),
    'std': numpy.std(durations)
}

print(durations)
df = pd.DataFrame(durations)
df.to_csv("hymn23.csv")
