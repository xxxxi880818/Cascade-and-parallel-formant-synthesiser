# Cascade-and-parallel-formant-synthesiser
# Formant Synthesiser System

This project implements a simple formant synthesiser in Python using both **cascade** and **parallel** structures to generate vowel sounds.

## 📁 Files

- `Cascade_formant.py` – Calculates amplitude and phase for the cascade synthesiser
- `Parallel_formant.py` – Calculates amplitude and phase for the parallel synthesiser
- `Generate-files.py` – Plots frequency responses and saves audio files
- `Formant_synthesis.py` – Main script that runs the full synthesis
- `synthesiser_template.py` – Contains shared functions like impulse train generation and filter design

## ▶️ How to Run

1. Install dependencies:

   pip install numpy scipy soundfile matplotlib
   
2. Run the main script:

    python Formant_synthesis.py

   
This will generate .wav files and waveform plots for vowels /a/, /i/, and /u/.
