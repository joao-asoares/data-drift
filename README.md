# studd

Unsupervised Concept Drift Detection or Model Monitoring

- nomes: João Pedro Lamaison , João Marcos de Assis

Resumo da Análise

1. Está entrando na condição de retreinamento?  
   SIM
   Mudança detectada na iteração 201, Page Hinkley = 2.284849009799091
   Mudança detectada na iteração 435, Page Hinkley = 2.0386160742999544
   Mudança detectada na iteração 819, Page Hinkley = 2.254530640711381

   Isso ultrapassou o threshold configurado de 2.0
   O threshold é o limite definido no código: std_detector = detector(delta=delta, threshold=2)
   Parâmetros:
   Threshold: 2.0
   Delta: 0.1
   Número de detecções de drift: 3
   Pontos de detecção: [201, 435, 819]  
   Taxa de erro: 0.0893 (8.93%)
   Porque:
   O Page Hinkley monitora erros entre predições do professor vs aluno
   Quando a métrica ultrapassa threshold=2, detecta drift
   Isso triggera o retreinamento de ambos os modelos
   O pico no gráfico de erros representa o momento onde a métrica Page Hinkley cruzou o threshold, ativando a condição de retreinamento.
