# song_view
# ECVD: Análise Exploratória de Dados de Bioacústica

Este documento descreve o fluxo de trabalho para o Estudo de Caso de Visualização de Dados (ECVD) focado no processamento e análise visual de vocalizações de pássaros.

## 1. Escopo e Desafio
O objetivo é processar áudios de duração variável (contendo cantos de aves e ruído ambiente) e extrair representações visuais acessíveis e compreensíveis. O desafio técnico reside na limpeza do sinal (filtragem) e na demonstração visual da periodicidade das vocalizações, facilitando a criação de um dataset higienizado para modelos de IA.

## 2. Fluxo de Processamento (Pipeline)

Abaixo apresentamos o diagrama de blocos do processo, desde a entrada do áudio bruto até a geração do relatório em PDF.

```mermaid
graph TD
    A[Dataset Xeno-canto<br>Áudios de Duração Variável] --> B(Pré-processamento<br>Python/Librosa)
    
    subgraph Processamento de Áudio
    B --> C{Normalização e<br>Corte por Energia}
    C -->|Áudio Padronizado| D[Filtragem Passa-Banda]
    end
    
    subgraph Mapeamento e Visualização
    D --> E[Forma de Onda<br>Raw vs Filtered]
    D --> F[Espectrograma Mel<br>Mapa de Calor]
    D --> G[Autocorrelação<br>Periodicidade]
    end
    
    subgraph Produto Final
    E --> H(Gerador de PDF<br>fpdf2)
    F --> H
    G --> H
    H --> I[Relatório ECVD.pdf<br>com Placeholders de Texto]
    end
```
