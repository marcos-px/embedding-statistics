import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

embedding = np.load('reg_20250322213429_embedding.npy')

def analyze_embedding(embedding):
    stats_dict = {
        'Dimensões': len(embedding),
        'Média': np.mean(embedding),
        'Desvio Padrão': np.std(embedding),
        'Mínimo': np.min(embedding),
        'Máximo': np.max(embedding),
        'Mediana': np.median(embedding),
        'Quantidade de Zeros': np.sum(embedding == 0),
        'Quantidade de Números numpy': np.sum(np.isfinite(embedding)),
        'Assimetria Numpy': np.mean(np.abs(embedding - np.mean(embedding))),
        'Assimetria stats': stats.skew(embedding),
        'Curtose stats': stats.kurtosis(embedding),
        'Q1 np': np.percentile(embedding, 25),
        'Q3 np': np.percentile(embedding, 75),
        'IQR stats': stats.iqr(embedding),
        'Amplitude np': np.ptp(embedding),
        'Variância np': np.var(embedding),
        'Desvio Padrão np': np.std(embedding),
        'Entropia stats': stats.entropy(np.abs(embedding), base=2),
        'Soma np': np.sum(embedding),
        'Soma Quadrática np': np.sum(embedding**2),
        'Soma Absoluta np': np.sum(np.abs(embedding)),
        'Soma Absoluta Quadrática np': np.sum(np.abs(embedding)**2),        
    }

    #views
    plt.figure(figsize=(15, 10))

    #histogram
    plt.subplot(2, 2, 1)
    plt.hist(embedding, bins=50, edgecolor='black')
    plt.title('Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    #Boxplot
    plt.subplot(2, 2, 2)
    plt.boxplot(embedding)
    plt.title('Embedding Boxplot')
    plt.ylabel('Value')

    #Density Plot
    plt.subplot(2, 2, 3)
    sns.kdeplot(embedding, fill=True)
    plt.title('Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Q-Q Normality Plot
    plt.subplot(2, 2, 4)
    stats.probplot(embedding, dist="norm", plot=plt)
    plt.title('Q-Q Normality Plot')

    plt.tight_layout()
    plt.savefig('reg_20250322213429_embedding.png')

    #Normality Test
    _, p_shapiro_value = stats.shapiro(embedding)
    stats_dict['Shapiro-Wilk p-value'] = p_shapiro_value

    return stats_dict

results = analyze_embedding(embedding)

print("Statistic Embedding's Analysis")
for key, value in results.items():
    print(f"{key}: {value}")

pd.DataFrame.from_dict(results, orient='index', columns=['Value']).to_csv('reg_20250322213429_embedding.csv')

print("Done!")
