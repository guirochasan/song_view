import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def aplicar_filtro_passa_banda(y, sr, lowcut=1000, highcut=8000, order=5):
    """Aplica um filtro digital para isolar a frequência do canto das aves (1kHz - 8kHz)."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, y)

def extrair_janela_energia(y, sr, window_sec=5.0):
    """
    Padroniza o tamanho do áudio para garantir que o gráfico tenha sempre a mesma largura.
    """
    window_samples = int(window_sec * sr)
    
    # Preenchimento (zero-padding) para áudios curtos
    if len(y) < window_samples:
        padding = window_samples - len(y)
        y = np.pad(y, (0, padding), mode='constant')
        return y
        
    # Identificação do trecho de maior intensidade para áudios longos
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    max_rms_frame = np.argmax(rms)
    center_sample = max_rms_frame * 512
    
    start = max(0, center_sample - (window_samples // 2))
    end = start + window_samples
    
    # Ajuste de borda
    if end > len(y):
        end = len(y)
        start = end - window_samples
        
    return y[start:end]

def configurar_processamento(input_dir):
    """Permite ao usuário definir as configurações de execução via terminal."""
    print("=== Configuração do Dataset ===")
    
    # Seleção de formato
    extensoes = []
    while True:
        resp_fmt = input("Quais formatos deseja processar? (1: Apenas .wav, 2: Apenas .mp3, 3: Ambos): ").strip()
        if resp_fmt == '1':
            extensoes = ["*.wav"]
            break
        elif resp_fmt == '2':
            extensoes = ["*.mp3"]
            break
        elif resp_fmt == '3':
            extensoes = ["*.wav", "*.mp3"]
            break
        print("Opção inválida.")
    
    # Coleta arquivos baseados na seleção
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not arquivos:
        print(f"Nenhum arquivo encontrado com a extensão selecionada em '{input_dir}'.")
        return []

    print(f"\nForam encontrados {len(arquivos)} arquivos de áudio compatíveis.")
    
    # Limite de arquivos
    while True:
        try:
            limite_str = input("Quantos arquivos deseja processar? (Digite 0 para processar TODOS): ").strip()
            limite = int(limite_str)
            if limite < 0:
                print("Por favor, digite um número não negativo.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Digite um número inteiro.")
    
    # Aplica o limite se for maior que 0
    if limite > 0:
        arquivos = arquivos[:limite]
        print(f"\nLimitando o processamento aos primeiros {len(arquivos)} arquivos.")
    else:
        print("\nProcessando TODOS os arquivos encontrados.")
        
    return arquivos

def processar_dataset(input_dir="dataset_aves", output_dir="plots_espectrogramas"):
    """
    Controla o fluxo principal: configuração interativa, iteração e geração dos gráficos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    arquivos_audio = configurar_processamento(input_dir)
    
    if not arquivos_audio:
        return

    # Estilo limpo e legível
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for caminho in arquivos_audio:
        nome_arquivo = os.path.basename(caminho)
        print(f"Processando: {nome_arquivo}...")
        
        # 1. Carregamento (Librosa ignora metadados e trata mp3/wav transparentemente)
        y_bruto, sr = librosa.load(caminho, sr=None)
        
        # 2. Padronização de tamanho fixa para 5 segundos
        y_janela = extrair_janela_energia(y_bruto, sr, window_sec=5.0)
        
        # 3. Filtragem
        y_limpo = aplicar_filtro_passa_banda(y_janela, sr)
        
        # 4. Cálculo
        S = librosa.feature.melspectrogram(y=y_limpo, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 5. Plotagem com dimensões e escalas estritas
        fig, ax = plt.subplots(figsize=(8, 4))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', 
                                       sr=sr, fmax=8000, cmap='viridis', ax=ax)
        
        ax.set_xlim([0, 5.0]) 
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensidade (dB)')
        ax.set_title(f'Espectrograma Mel: {nome_arquivo}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        
        plt.tight_layout()
        
        # 6. Salvamento
        nome_saida = os.path.splitext(nome_arquivo)[0] + "_spec.png"
        plt.savefig(os.path.join(output_dir, nome_saida), dpi=300)
        plt.close(fig)
        
    print(f"\nSucesso! {len(arquivos_audio)} espectrogramas foram salvos na pasta '{output_dir}'.")

if __name__ == "__main__":
    # Garante que a pasta de entrada exista antes de tentar processar
    if not os.path.exists("Amazilia_versicolor"):
        print("A pasta 'Amazilia_versicolor' não existe. Criando...")
        os.makedirs("Amazilia_versicolor")
        print("Coloque seus áudios dentro de 'Amazilia_versicolor' e execute novamente.")
    else:
        processar_dataset(input_dir="Amazilia_versicolor", output_dir="plots_espectrogramas")