# Risultato Paper OPS-SAT

|      **Modello**       | **Accuracy** | **Precision** | **Recall** | **F1** | **MCC** | **AUC_PR** | **AUC_ROC** | **N_score** |
| :--------------------: | :----------: | :-----------: | :--------: | :----: | :-----: | :--------: | :---------: | ----------- |
|     **Supervised**     |              |               |            |        |         |            |             |             |
|     **LinearSVC**      |    0.926     |     0.911     |   0.726    | 0.808  |  0.771  |   0.949    |    0.976    | 0.867       |
| **LogisticRegression** |    0.924     |     0.92      |   0.708    |  0.8   |  0.764  |   0.949    |    0.976    | 0.867       |
|        **FCNN**        |     0.96     |     0.926     |   0.885    | 0.905  |  0.88   |   0.963    |    0.982    | 0.903       |
|      **AdaBoost**      |    0.934     |     0.89      |   0.788    | 0.836  |  0.797  |   0.923    |    0.962    | 0.841       |
|      **RF+ICCS**       |    0.964     |     0.98      |    0.85    |  0.91  |  0.891  |   0.949    |    0.976    | 0.867       |
|     **Linear+L2**      |    0.902     |     0.969     |   0.558    | 0.708  |  0.69   |   0.889    |    0.95     | 0.814       |
|       **XGBOD**        |    0.968     |     0.953     |   0.894    | 0.922  |  0.903  |   0.969    |    0.99     | 0.912       |
|    **Unsupervised**    |              |               |            |        |         |            |             |             |
|      **MO_GAAL**       |    0.896     |     0.939     |   0.549    | 0.693  |  0.669  |   0.771    |    0.849    | 0.699       |
|       **AnoGAN**       |    0.594     |     0.296     |   0.655    | 0.408  |  0.19   |   0.403    |    0.651    | 0.239       |
|      **SO_GAAL**       |     0.89     |     0.937     |   0.522    |  0.67  |  0.649  |   0.858    |    0.919    | 0.761       |
|      **IForest**       |    0.701     |     0.297     |   0.292    | 0.295  |  0.105  |   0.347    |    0.635    | 0.301       |
|        **KNN**         |    0.849     |     0.78      |   0.407    | 0.535  |  0.489  |   0.658    |    0.852    | 0.593       |
|       **OCSVM**        |    0.837     |     0.721     |   0.389    | 0.506  |  0.447  |   0.659    |    0.788    | 0.655       |
|        **ABOD**        |    0.845     |     0.782     |   0.381    | 0.512  |  0.472  |   0.644    |    0.843    | 0.584       |
|        **INNE**        |     0.83     |     0.689     |   0.372    | 0.483  |  0.418  |   0.624    |    0.801    | 0.646       |
|        **ALAD**        |    0.819     |     0.667     |   0.301    | 0.415  |  0.361  |   0.537    |     0.7     | 0.451       |
|        **LMDD**        |    0.822     |      1.0      |   0.168    | 0.288  |  0.37   |   0.624    |    0.765    | 0.663       |
|        **SOD**         |    0.826     |     0.611     |   0.513    | 0.558  |  0.453  |   0.621    |    0.797    | 0.549       |
|        **COF**         |    0.834     |     0.667     |   0.442    | 0.532  |  0.449  |   0.603    |    0.774    | 0.593       |
|        **LODA**        |     0.83     |     0.689     |   0.372    | 0.483  |  0.418  |   0.549    |    0.692    | 0.522       |
|       **LUNAR**        |    0.819     |     0.743     |    0.23    | 0.351  |  0.313  |   0.541    |    0.797    | 0.46        |
|       **CBLOF**        |    0.802     |     0.569     |   0.292    | 0.386  |  0.304  |    0.45    |    0.574    | 0.372       |
|        **DIF**         |    0.788     |      1.0      |   0.009    | 0.018  |  0.084  |   0.494    |    0.805    | 0.522       |
|        **VAE**         |    0.794     |     0.532     |   0.292    | 0.377  |  0.283  |   0.446    |    0.687    | 0.513       |
|        **GMM**         |    0.783     |     0.482     |   0.239    |  0.32  |  0.225  |   0.426    |    0.713    | 0.389       |
|      **DeepSVDD**      |    0.788     |     0.509     |   0.239    | 0.325  |  0.241  |   0.344    |    0.55     | 0.336       |
|        **PCA**         |    0.779     |     0.464     |    0.23    | 0.308  |  0.21   |   0.373    |    0.612    | 0.363       |
|       **COPOD**        |    0.767     |      0.4      |   0.177    | 0.245  |  0.147  |   0.328    |    0.627    | 0.257       |
|        **SOS**         |    0.758     |     0.364     |   0.177    | 0.238  |  0.125  |   0.308    |    0.524    | 0.274       |
|        **ECOD**        |    0.767     |     0.396     |   0.168    | 0.236  |  0.14   |    0.34    |    0.637    | 0.345       |

# Risultati XGBOD su OPS_SAT

Varie versioni

| Modalità       | Accuracy | Precision | Recall | F1    | MCC   | AUC_PR | AUC_ROC | N_score |
| -------------- | -------- | --------- | ------ | ----- | ----- | ------ | ------- | ------- |
| \+Mod e Param  | 0.97     | 0.945     | 0.912  | 0.928 | 0.909 | 0.973  | 0.992   | 0.92    |
| EarlyStop(M+P) | 0.97     | 0.971     | 0.885  | 0.926 | 0.909 | 0.969  | 0.99    | 0.912   |
| Totale         | 0.97     | 0.971     | 0.912  | 0.928 | 0.909 | 0.973  | 0.992   | 0.92    |
| Più Modelli    | 0.968    | 0.944     | 0.903  | 0.923 | 0.903 | 0.974  | 0.991   | 0.92    |
| Con Param      | 0.964    | 0.943     | 0.885  | 0.913 | 0.891 | 0.972  | 0.991   | 0.912   |
| Senza Param    | 0.962    | 0.935     | 0.885  | 0.909 | 0.886 | 0.977  | 0.992   | 0.912   |
| Con Grid       | 0.947    | 0.989     | 0.761  | 0.86  | 0.839 | 0.898  | 0.945   | 0.969   |

# Risultati Rocket su OPS_SAT

Varie versioni

| **Modalità**             | **Accuracy** | **Precision** | **Recall** | **F1** | **MCC** | **AUC_PR** | **AUC_ROC** | **N_score** |
| ------------------------ | ------------ | ------------- | ---------- | ------ | ------- | ---------- | ----------- | ----------- |
| **Rocket +  RidgeCV**    | 0.977        | 0.972         | 0.92       | 0.945  | 0.932   | 0.962      | 0.984       | 0.929       |
| **Rocket + LogisticReg** | 0.974        | 0.963         | 0.912      | 0.936  | 0.92    | 0.964      | 0.986       | 0.92        |
| **Rocket + RidgeReg**    | 0.864        | 0.936         | 0.389      | 0.55   | 0.554   | 0.871      | 0.94        | 0.92        |
| **Rocket + KNN**         | 0.845        | 0.746         | 0.416      | 0.534  | 0.478   | 0.61       | 0.804       | 0.522       |
| **Unsupervised**         | 0.834        | 0.963         | 0.23       | 0.371  | 0.424   | 0.726      | 0.772       | 0.646       |
| **Supervised**           |              |               |            |        |         |            |             |             |

`RidgeClassifierCV` → è il modello usato nella demo del paper di rocket ed ho riscontrato metriche migliori

## Riesecuzione del Paper

| **Dataset**           | **Accuracy**        |
| --------------------- | ------------------- |
| Coffee                | 1                   |
| Computers             | 0.64                |
| Adiac                 | 0.6342710997442456  |
| ArrowHead             | 0.7885714285714286  |
| BeetleFly             | 0.8                 |
| CinCECGTorso          | 0.7898550724637681  |
| CBF                   | 0.9711111111111111  |
| ChlorineConcentration | 0.5901041666666667  |
| GunPoint              | 0.9866666666666667  |
| Ham                   | 0.7714285714285715  |
| HandOutlines          | 0.9351351351351351  |
| InlineSkate           | 0.36727272727272725 |
| Lightning2            | 0.6229508196721312  |
| Mallat                | 0.9283582089552239  |
| MiddlePhalanxTW       | 0.5584415584415584  |

## Risultati Rocket su Dataset BakeOff (paper)

Sono i risultati della mia implementazione (presa dallo stesso paper ma leggermente modificata) con alcuni asset dei dataset di "BakeOff".

| **Dataset**           | **Accuracy** | **Precision** | **Recall** | **F1** | **MCC** | **AUC_PR** | **AUC_ROC** |
| --------------------- | ------------ | ------------- | ---------- | ------ | ------- | ---------- | ----------- |
| Coffee                | 1.0          | 1.0           | 1.0        | 1.0    | 1.0     | 1.0        | 1.0         |
| Computers             | 0.704        | 0.713         | 0.704      | 0.701  | 0.417   | 0.358      | 0.753       |
| Adiac                 | 0.777        | 0.823         | 0.777      | 0.761  | 0.773   | 0.79       | 0.978       |
| ArrowHead             | 0.771        | 0.809         | 0.771      | 0.756  | 0.69    | 0.88       | 0.942       |
| BeetleFly             | 0.85         | 0.885         | 0.85       | 0.847  | 0.734   | 0.331      | 1.0         |
| CinCECGTorso          | 0.823        | 0.831         | 0.823      | 0.822  | 0.767   | 0.914      | 0.963       |
| CBF                   | 0.992        | 0.992         | 0.992      | 0.992  | 0.988   | 1.0        | 1.0         |
| ChlorineConcentration | 0.782        | 0.778         | 0.782      | 0.774  | 0.633   | 0.813      | 0.902       |
| GunPoint              | 1.0          | 1.0           | 1.0        | 1.0    | 1.0     | 0.315      | 1.0         |
| Ham                   | 0.657        | 0.658         | 0.657      | 0.655  | 0.314   | 0.37       | 0.734       |
| HandOutlines          | 0.916        | 0.917         | 0.916      | 0.915  | 0.817   | 0.963      | 0.96        |
| InlineSkate           | 0.705        | 0.708         | 0.705      | 0.7    | 0.404   | 0.865      | 0.838       |
| Lightning2            | 0.738        | 0.743         | 0.738      | 0.733  | 0.473   | 0.829      | 0.805       |
| Mallat                | 0.943        | 0.951         | 0.943      | 0.944  | 0.936   | 0.977      | 0.994       |
| MiddlePhalanxTW       | 0.552        | 0.516         | 0.552      | 0.523  | 0.419   | 0.438      | 0.822       |

---

# Imp

Per valutare i dataset ho dovuto aggiungere i parametri per valutare la multidimensionalità, cioè quando dobbiamo classificare più di due classi (classificazione binaria).

Quindi se dobbiamo classificare più classi bisogna usare un array multidimensionale per questo in caso contrario poniamo `predict_proba` in modo monodimensionale.

Utilizziamo `predict_proba` al posto di `decision_function`

```python
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
```

Per valutare invece un dataset monodimensionale basta aggiungere `[:, 1]` dopo `predict_proba`

```python
y_proba = model.predict_proba(features_test)[:,1]
```

![image.png](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/fe96bd76-28f7-47ed-89a3-bde1a0a930d4/e7890c20-9c12-4c32-ba53-38772f4b0f12)

---

Ho sostituito `LogisticRegression` con `RidgeClassifierCV` dato che è molto più veloce.

Esempio con dataset *InlineSkate* con algoritmo:

- `RidgeClassifierCV` ha finito in $$10.9s$$
- `LogisticRegression` in $$2min,5s$$

Per aggiustare il codice ho dovuto aggiungere:

```python
y_proba = softmax(model.decision_function(features_test), axis=1)
```

per convertire i margini di decisione in probabilità → dato che stiamo lavorando con più classi e `decision_function` lavora soprattutto con classi binarie

?descriptionFromFileType=function+toLocaleUpperCase()+{+[native+code]+}+File&mimeType=application/octet-stream&fileName=Risultati.md&fileType=undefined&fileExtension=md