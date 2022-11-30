import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
import sys
import pdb
if __name__ == "__main__":
    indir = Path(sys.argv[1])
    _norm_scale = sys.argv[2]
    _snr = sys.argv[3]
    epsilon = 1e-5
    norm_scale = float(_norm_scale)
    snr_db = float(_snr)
    # indir = "../work/wav/Arthur_healthy"
    inwav = sorted(Path(indir).glob("**/*.wav"))
    
    output_dir = Path(indir.parent) / Path(indir.stem + "_normed_%s_snr_%s"%(_norm_scale.replace(".", "_"), _snr.replace(".", "_")))
    Path.mkdir(output_dir,  parents=False, exist_ok=True)
    
    for i, f in enumerate(inwav):
        y, sr = librosa.load(f, sr=48000)
        # Kevin
        # Make sure y_norm.max() <=0.8, or too loud conversion might be generated
        y, _ = librosa.effects.trim(y, top_db=22)
        y_norm = librosa.util.normalize(y, norm=norm_scale) 
        
        # NORMALIZATION
        power = y_norm ** 2
        signal_averagepow_db = 10*np.log10(np.mean(power))
        noise_db = signal_averagepow_db - snr_db
        noise_watts = 10 **(noise_db / 10)
        # y_norm = y
        # noise_range = y_norm.max() * noise_scale
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_watts), size=len(y))
        y_noisy = noise + y_norm
        
        sub = 0 
        for x in y_noisy:
            if x == 0:
                x = epsilon
                sub = sub + 1
        sf.write(output_dir /Path(f.stem + "_normed.wav"), y_noisy, samplerate=sr)
        
    print("Generate %d file at %s" %(i+1, str(output_dir)))
    