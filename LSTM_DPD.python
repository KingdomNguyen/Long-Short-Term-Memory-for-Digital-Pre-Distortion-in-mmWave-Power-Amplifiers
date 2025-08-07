import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import signal

# 1. Data generation functions
def generate_pa_output(x, params):
    """Simulates PA nonlinear distortion using Saleh model"""
    alpha_a, beta_a, alpha_phi, beta_phi = params
    r = np.abs(x)
    amplitude = (alpha_a * r) / (1 + beta_a * r ** 2)
    phase = (alpha_phi * r ** 2) / (1 + beta_phi * r ** 2)
    y = amplitude * np.exp(1j * (np.angle(x) + phase))
    return y

def generate_mmwave_signal(num_samples, bandwidth=100e6, fs=1e9):
    """Generates multi-carrier mmWave signal"""
    t = np.arange(num_samples) / fs
    num_subcarriers = 64
    subcarrier_spacing = bandwidth / num_subcarriers
    x = np.zeros(num_samples, dtype=complex)
    
    for i in range(num_subcarriers):
        f = (i - num_subcarriers / 2) * subcarrier_spacing
        x += np.exp(1j * 2 * np.pi * f * t) * (np.random.randn() + 1j * np.random.randn())
    return x / np.std(x)

def add_memory_effect(y, memory_depth=3):
    """Adds memory effects using FIR filter"""
    taps = np.exp(-np.arange(memory_depth))
    taps = taps / np.sum(taps)
    y_real = signal.lfilter(taps, 1.0, np.real(y))
    y_imag = signal.lfilter(taps, 1.0, np.imag(y))
    return y_real + 1j * y_imag

def generate_dpd_dataset(num_samples=10000, train_ratio=0.8):
    """Generates dataset for DPD training"""
    pa_params = [2.1587, 1.1517, 4.0033, 9.1040]  # Saleh model parameters
    x = generate_mmwave_signal(num_samples)
    y = generate_pa_output(x, pa_params)
    y = add_memory_effect(y)

    # Add AWGN noise
    snr_db = 25
    noise_power = 10 ** (-snr_db / 10)
    y += np.sqrt(noise_power / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))

    # Normalize I/Q components
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = scaler.fit_transform(np.vstack([np.real(x), np.imag(x)]).T)
    y_scaled = scaler.transform(np.vstack([np.real(y), np.imag(y)]).T)

    # Train-test split
    split_idx = int(num_samples * train_ratio)
    x_train, x_test = x_scaled[:split_idx], x_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    return (x_train, y_train), (x_test, y_test), scaler

def prepare_lstm_data(x, y, time_steps=10):
    """Creates time-series sequences for LSTM training"""
    X, Y = [], []
    for i in range(len(x) - time_steps):
        X.append(x[i:i + time_steps])
        Y.append(y[i + time_steps])
    return np.array(X), np.array(Y)

# 2. Data preparation
(x_train, y_train), (x_test, y_test), scaler = generate_dpd_dataset(num_samples=10000)
time_steps = 10
X_train, Y_train = prepare_lstm_data(x_train, y_train, time_steps)
X_test, Y_test = prepare_lstm_data(x_test, y_test, time_steps)

# 3. LSTM model architecture
def build_lstm_dpd(L=128, N1=64, N2=32, N3=2):
    """Builds LSTM-based DPD model"""
    # Input layer
    input_layer = Input(shape=(time_steps, 2), name='input')

    # LSTM layer
    lstm_layer = LSTM(L, return_sequences=False, name='lstm')(input_layer)

    # Fully connected layers with LeakyReLU
    fc1 = Dense(N1, name='linear1')(lstm_layer)
    lr1 = LeakyReLU(alpha=0.01, name='leakyRelu1')(fc1)

    fc2 = Dense(N2, name='linear2')(lr1)
    lr2 = LeakyReLU(alpha=0.01, name='leakyRelu2')(fc2)

    # Output layer
    output_layer = Dense(N3, name='linearOutput')(lr2)

    return Model(inputs=input_layer, outputs=output_layer, name='DPD_LSTM')

# Model parameters
L = 128  # LSTM units
N1 = 64  # First dense layer neurons
N2 = 32  # Second dense layer neurons
N3 = 2   # I/Q output dimensions

# Initialize and compile model
model = build_lstm_dpd(L=L, N1=N1, N2=N2, N3=N3)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])
model.summary()

# 4. Model training
history = model.fit(X_train, Y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_test, Y_test),
                    verbose=1)

# 5. Model evaluation
Y_pred = model.predict(X_test)

def calculate_nmse(y_true, y_pred):
    """Calculates Normalized Mean Square Error (dB)"""
    return 10 * np.log10(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)) / np.mean(np.sum(y_true ** 2, axis=1)))

nmse = calculate_nmse(Y_test, Y_pred)
print(f"NMSE: {nmse:.2f} dB")

# 6. Visualization of results
plt.figure(figsize=(15, 10))

# Training/validation loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Constellation diagram
plt.subplot(2, 2, 2)
plt.scatter(Y_test[:, 0], Y_test[:, 1], s=1, label='Actual Output')
plt.scatter(Y_pred[:, 0], Y_pred[:, 1], s=1, label='Predicted Output')
plt.title('Constellation Diagram')
plt.xlabel('I Component')
plt.ylabel('Q Component')
plt.legend()

# AM/AM characteristics
r_in = np.sqrt(np.sum(X_test[:, -1, :] ** 2, axis=1))  # Input amplitude
r_out_actual = np.sqrt(np.sum(Y_test ** 2, axis=1))    # Actual output amplitude
r_out_pred = np.sqrt(np.sum(Y_pred ** 2, axis=1))      # Predicted output amplitude

plt.subplot(2, 2, 3)
plt.scatter(r_in, r_out_actual, s=1, label='Actual')
plt.scatter(r_in, r_out_pred, s=1, label='Predicted')
plt.title('AM/AM Characteristics')
plt.xlabel('Input Amplitude')
plt.ylabel('Output Amplitude')
plt.legend()

# AM/PM characteristics
phase_in = np.angle(X_test[:, -1, 0] + 1j * X_test[:, -1, 1])
phase_out_actual = np.angle(Y_test[:, 0] + 1j * Y_test[:, 1])
phase_out_pred = np.angle(Y_pred[:, 0] + 1j * Y_pred[:, 1])
phase_diff_actual = phase_out_actual - phase_in
phase_diff_pred = phase_out_pred - phase_in

plt.subplot(2, 2, 4)
plt.scatter(r_in, phase_diff_actual, s=1, label='Actual')
plt.scatter(r_in, phase_diff_pred, s=1, label='Predicted')
plt.title('AM/PM Characteristics')
plt.xlabel('Input Amplitude')
plt.ylabel('Phase Difference (rad)')
plt.legend()

plt.tight_layout()
plt.show()
