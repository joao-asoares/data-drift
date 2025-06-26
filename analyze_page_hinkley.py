import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from studd.studd_batch import STUDD
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT

def analyze_page_hinkley_workflow(X, y, delta, window_size):
    """
    Workflow modificado
    """
    ucdd = STUDD(X=X, y=y, n_train=window_size)
    ucdd.initial_fit(model=RF(), std_model=LR())
    
    # Criar detector Page Hinkley com os mesmos parâmetros
    std_detector = PHT(delta=delta/2, threshold=2)
    
    print(f"Page Hinkley Parameters:")
    print(f"  - Delta: {delta/2}")
    print(f"  - Threshold: 2")
    print(f"  - Min instances: {std_detector.min_instances}")
    print(f"  - Alpha: {std_detector.alpha}")
    
    # Listas para armazenar os valores da métrica
    page_hinkley_values = []
    std_errors = []
    iterations = []
    detected_changes = []
    
    # Simular o processo de detecção
    iter_count = window_size
    
    while ucdd.datastream.has_more_samples():
        Xi, yi = ucdd.datastream.next_sample()
        
        # Predições dos modelos
        teacher_pred = ucdd.base_model.predict(Xi)
        student_pred = ucdd.student_model.predict(Xi)
        
        # Erro entre professor e aluno
        std_err = int(teacher_pred != student_pred)
        std_errors.append(std_err)
        
        # Adicionar ao detector
        std_detector.add_element(std_err)
        
        # Capturar valor atual da métrica Page Hinkley
        page_hinkley_values.append(std_detector.sum)
        iterations.append(iter_count)
        
        # Verificar se houve detecção de mudança
        if std_detector.detected_change():
            detected_changes.append(iter_count)
            print(f"Mudança detectada na iteração {iter_count}, Page Hinkley = {std_detector.sum}")
        
        iter_count += 1
    
    return {
        'page_hinkley_values': page_hinkley_values,
        'std_errors': std_errors,
        'iterations': iterations,
        'detected_changes': detected_changes,
        'threshold': 2,
        'delta': delta/2
    }

def plot_page_hinkley_analysis(results):
    """
    Plotar análise completa do Page Hinkley
    """
    # Gráfico principal: Page Hinkley Sum vs Threshold
    plt.figure(figsize=(14, 8))
    
    plt.plot(results['iterations'], results['page_hinkley_values'], 'b-', linewidth=1.5, label='Page Hinkley Sum', alpha=0.8)
    plt.axhline(y=results['threshold'], color='r', linestyle='--', linewidth=2, label=f'Threshold = {results["threshold"]}')
    
    # Destacar área acima do threshold
    above_threshold = np.array(results['page_hinkley_values']) >= results['threshold']
    if any(above_threshold):
        plt.fill_between(results['iterations'], results['page_hinkley_values'], results['threshold'], 
                        where=above_threshold, alpha=0.3, color='red', label='Região de Drift')
    
    # Marcar pontos de detecção
    for i, change_point in enumerate(results['detected_changes']):
        idx = change_point - results['iterations'][0]
        if idx < len(results['page_hinkley_values']):
            plt.plot(change_point, results['page_hinkley_values'][idx], 'ro', markersize=10, 
                    label=f'Detecção {i+1}' if i == 0 else "")
            plt.annotate(f'Drift {i+1}\n({change_point}, {results["page_hinkley_values"][idx]:.3f})', 
                        xy=(change_point, results['page_hinkley_values'][idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Iterações', fontsize=12)
    plt.ylabel('Page Hinkley Sum', fontsize=12)
    plt.title('Análise da Métrica Page Hinkley - Detecção de Drift\nThreshold=2, Delta=0.1', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('page_hinkley_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Segundo gráfico: Análise detalhada dos erros
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Erros ao longo do tempo
    plt.subplot(1, 2, 1)
    plt.plot(results['iterations'], results['std_errors'], 'g-', linewidth=1, alpha=0.7, label='Erro Professor-Aluno')
    
    # Marcar pontos de detecção
    for i, change_point in enumerate(results['detected_changes']):
        plt.axvline(x=change_point, color='red', linestyle=':', alpha=0.8, linewidth=2)
        if i == 0:
            plt.axvline(x=change_point, color='red', linestyle=':', alpha=0.8, linewidth=2, label='Detecções de Drift')
    
    plt.xlabel('Iterações')
    plt.ylabel('Erro (0 ou 1)')
    plt.title('Erros entre Professor e Aluno')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Histograma dos valores Page Hinkley
    plt.subplot(1, 2, 2)
    plt.hist(results['page_hinkley_values'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=results['threshold'], color='r', linestyle='--', linewidth=2, label=f'Threshold = {results["threshold"]}')
    plt.xlabel('Page Hinkley Values')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Valores Page Hinkley')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('page_hinkley_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return None

def main():
    # Carregar dados
    merged_combined_data = pd.read_parquet("merged_embeddings.parquet")
    
    # Preparar colunas
    column = []
    for i in range(merged_combined_data.shape[1] - 1):
        column.append(i)
    column.append("target")
    merged_combined_data.columns = column
    
    # Parâmetros
    delta = 0.2
    window_size = 100
    
    y = merged_combined_data.target.values
    X = merged_combined_data.drop(['target'], axis=1)
    
    print("=== ANÁLISE DO PAGE HINKLEY ===")
    print(f"Tamanho do dataset: {len(y)}")
    print(f"Delta usado: {delta}")
    print(f"Delta para Page Hinkley: {delta/2}")
    print(f"Threshold: 2")
    print(f"Tamanho da janela: {window_size}")
    print("=" * 40)
    
    # Executar análise
    results = analyze_page_hinkley_workflow(X, y, delta, window_size)
    
    # Análise dos resultados
    print(f"\nRESULTADOS:")
    print(f"Número total de iterações: {len(results['page_hinkley_values'])}")
    print(f"Número de detecções de drift: {len(results['detected_changes'])}")
    print(f"Pontos de detecção: {results['detected_changes']}")
    
    # Verificar se houve ultrapassagem do threshold
    max_page_hinkley = max(results['page_hinkley_values']) if results['page_hinkley_values'] else 0
    print(f"Valor máximo da métrica Page Hinkley: {max_page_hinkley:.4f}")
    print(f"Threshold configurado: {results['threshold']}")
    
    # Análise de retreinamento
    if len(results['detected_changes']) > 0:
        print(f"\n Está entrando na condição de retreinamento!")
        print(f"Motivo: A métrica Page Hinkley ultrapassou o threshold de {results['threshold']}")
        print(f"Isso aconteceu {len(results['detected_changes'])} vez(es)")
        
        for i, change_point in enumerate(results['detected_changes']):
            idx = change_point - results['iterations'][0]
            if idx < len(results['page_hinkley_values']):
                value = results['page_hinkley_values'][idx]
                print(f"  - Detecção {i+1}: Iteração {change_point}, Page Hinkley = {value:.4f}")
    else:
        print(f"\nNÃO está entrando na condição de retreinamento.")
        print(f"Motivo: A métrica Page Hinkley (máx: {max_page_hinkley:.4f}) nunca ultrapassou o threshold de {results['threshold']}")
    
    errors_sum = sum(results['std_errors'])
    error_rate = errors_sum / len(results['std_errors']) if results['std_errors'] else 0
    print(f"Total de erros professor-aluno: {errors_sum}")
    print(f"Taxa de erro: {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    # Plotar resultados
    plot_page_hinkley_analysis(results)
    
    return results

if __name__ == "__main__":
    results = main()
