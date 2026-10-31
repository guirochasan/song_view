import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import os

# Configurações visuais (Acessibilidade)
plt.style.use('seaborn-v0_8-whitegrid')
COLORMAP = 'magma' # Paleta colorblind-safe recomendada para mapas de calor

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Cria os coeficientes de um filtro passa-banda Butterworth."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Aplica o filtro passa-banda aos dados de áudio."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_most_energetic_segment(y, sr, duration=5.0):
    """
    Lida com áudios de duração variável extraindo a janela de tempo (ex: 5s)
    com a maior energia acústica, removendo silêncios longos.
    """
    frame_length = int(sr * duration)
    if len(y) <= frame_length:
        return y # Retorna o áudio inteiro se for menor que a janela desejada
    
    # Calcula a energia por frames
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    # Encontra o frame com o pico de energia
    max_energy_idx = np.argmax(rms)
    # Converte o índice do frame de volta para amostras
    sample_idx = librosa.frames_to_samples(max_energy_idx, hop_length=512)
    
    # Define os limites de corte (centralizando o pico de energia)
    start = max(0, sample_idx - frame_length // 2)
    end = min(len(y), start + frame_length)
    
    # Ajusta o início se o fim ultrapassar o tamanho do array
    if end - start < frame_length:
        start = max(0, end - frame_length)
        
    return y[start:end]

def generate_visualizations(audio_path, output_dir="output"):
    """
    Processa o áudio e gera as 3 representações visuais, salvando-as no disco.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carregamento e Corte
    # sr=22050 é o padrão do librosa, adequado para maioria das vocalizações de aves
    y_raw, sr = librosa.load(audio_path, sr=22050) 
    y = extract_most_energetic_segment(y_raw, sr, duration=5.0)
    
    # 2. Filtragem (Aves geralmente vocalizam entre 1kHz e 8kHz)
    y_filtered = apply_bandpass_filter(y, 1000.0, 8000.0, sr)

    # --- RV 1: Forma de Onda (Comparação) ---
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='#1E8FFF', alpha=0.7)
    ax[0].set(title='Sinal Bruto (Raw)')
    ax[0].label_outer()
    
    librosa.display.waveshow(y_filtered, sr=sr, ax=ax[1], color='#FF9E1E', alpha=0.9)
    ax[1].set(title='Sinal Filtrado (Passa-banda 1kHz - 8kHz)', xlabel='Tempo (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rv1_waveform.png'), dpi=300)
    plt.close()

    # --- RV 2: Espectrograma Mel ---
    S = librosa.feature.melspectrogram(y=y_filtered, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap=COLORMAP)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Espectrograma Mel (Energia e Frequência)', xlabel='Tempo (s)', ylabel='Frequência (Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rv2_espectrograma.png'), dpi=300)
    plt.close()

    # --- RV 3: Autocorrelação (Periodicidade) ---
    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, win_length=192)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo', cmap=COLORMAP, ax=ax)
    ax.set(title='Tempograma / Autocorrelação (Padrão Rítmico)', xlabel='Tempo (s)', ylabel='BPM / Periodicidade')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rv3_autocorrelacao.png'), dpi=300)
    plt.close()
    
    print(f"Visualizações geradas com sucesso na pasta '{output_dir}'.")

# Exemplo de uso (comente ao integrar com outros scripts):
# generate_visualizations('caminho_para_seu_audio_de_passaro.wav')