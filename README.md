# Long-Short-Term-Memory-for-Digital-Pre-Distortion-in-mmWave-Power-Amplifiers
## Problem statement

Modern wireless systems (5G/6G, mmWave) demand power amplifiers (PAs) to operate at high efficiency, pushing them into nonlinear regions that distort transmitted signals.
This degrades spectral efficiency, error vector magnitude. There are some conventional Digital Pre-distortion (DPD) methods to deal this include: volterra series, LUT-Based DPD,...
But these techniques face problems about high time complexity, inflexible with PA drift. I propose an LSTM-based DPD to address this problem by:
- Capturing long-term dependencies: LSTM’s recurrent structure models memory effects inherently.
- Learning complex nonlinearities: deep neural networks approximate arbitrary PA distortions better than polynomials.
- Robustness to noise: joint denoising and distortion correction via data-driven training
## Methodology
### Dataset preparation
I synthesize a nonlinear PA dataset with memory effects to train and evaluate the LSTM-DPD model, following these steps:

A. Input signal
- Waveform: 64-subcarrier OFDM with 256-QAM modulation.
- Bandwidth: 100 MHz.
- Normalization: Peak-to-average power ratio (PAPR) clipped to 8 dB.
- Length: 10,000 samples (80% train, 20% test).
B. Power Amplifier (PA) Modeling
- Nonlinearity:I use Saleh model because it is alidated for Class-AB and Doherty PAs and computationally efficient vs. Volterra  in 3GPP studies.
  The Saleh model mathematically captures two key distortions in PAs:
  + AM/AM Conversion: Nonlinear relationship between input and output amplitudes.
  + AM/PM Conversion: Phase shifts induced by input amplitude variations.
  ```python
  # Saleh Model Parameters
  alpha_a, beta_a = 2.1587, 1.1517  # AM/AM
  alpha_phi, beta_phi = 4.0033, 9.1040  # AM/PM

  def pa_model(x):
    """Apply Saleh model to complex signal x"""
    r = np.abs(x)
    amplitude = (alpha_a * r) / (1 + beta_a * r**2)
    phase = np.angle(x) + (alpha_phi * r**2) / (1 + beta_phi * r**4)
    return amplitude * np.exp(1j * phase)
  ```
- Memory Effects: modeling the PA’s frequency-dependent behavior using FIR filter where past inputs affect current outputs due to:
  + Thermal dynamics
  + Trapping effects in semiconductor substrates
  + Bias network impedance
  ```python
  memory_taps = np.array([0.7, 0.2, 0.1])  # Exponential decay
  y_memory = signal.lfilter(memory_taps, 1.0, pa_model(x))
  ```
- Noise: simulating real-world measurement noise in feedback paths (e.g., ADC quantization, channel noise).
  ```python
  noise_power = 10**(-snr_db/10)  # Convert SNR to linear power
  y_noisy = y_memory + np.sqrt(noise_power/2) * (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
  ```
### LSTM (Long Short-Term Memory)
 - LSTM is recurrent neural network (RNN) designed to process sequential data (e.g., time-series signals, text). Unlike traditional neural networks, LSTMs can "remember" long-term dependencies in data through a built-in memory mechanism.
 - LSTM is suitable for DPD because:
   + Memory Effects: PAs distort signals based on both current and past inputs (due to thermal dynamics, bias circuits, etc.). LSTM inherently models these temporal relationships.
   + Complex Nonlinearity: PA behavior isn’t a simple equation but a dynamic mix of effects. LSTMs learn these patterns without manual formula design.
- LSTM's mechanism:
  + Forget Gate: Decides which past PA distortions are irrelevant now.
  + Input Gate: Identifies new distortion patterns to remember.
  + Output Gate: Combines current and memorized data to predict the next correction.
```python
input_layer = Input(shape=(10, 2))  # 10 timesteps, I/Q channels
x = LSTM(128, return_sequences=False)(input_layer)
x = Dense(64)(x)
x = LeakyReLU(0.01)(x)
x = Dense(32)(x)
x = LeakyReLU(0.01)(x)
output_layer = Dense(2)(x)  # I/Q outputs
model = Model(inputs=input_layer, outputs=output_layer)
```
### Results
I test LSTM-DPD model for the case SNR = 25 dB 
- Normalized Mean Square Error: -23.79 dB
- Visulization of training loss curves, constellation diagrams, AM/AM characteristics and AM/PM distortion:
  ![https://github.com/KingdomNguyen/image_2/blob/main/Screenshot%202025-08-06%20215604.jpg](https://github.com/KingdomNguyen/image_2/blob/main/Screenshot%202025-08-06%20215604.jpg)

