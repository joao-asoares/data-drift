# studd

Unsupervised Concept Drift Detection or Model Monitoring

- nomes: João Pedro Lamaison , João Marcos de Assis

Resumo da Análise

1. Está entrando na condição de retreinamento?  
   SIM
   A métrica Page Hinkley atingiu 2.2099 na iteração 819
   Isso ultrapassou o threshold configurado de 2.0
   O threshold é o limite definido no código: std_detector = detector(delta=delta, threshold=2)
   Parâmetros:
   Threshold: 2.0
   Delta: 0.1
   Detecção: 1 vez na iteração 819
   Taxa de erro professor-aluno: 8.82%
   Porque:
   O Page Hinkley monitora erros entre predições do professor vs aluno
   Quando a métrica ultrapassa threshold=2, detecta drift
   Isso triggera o retreinamento de ambos os modelos
   O pico no gráfico de erros representa o momento onde a métrica Page Hinkley cruzou o threshold, ativando a condição de retreinamento.
