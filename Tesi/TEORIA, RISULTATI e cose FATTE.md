- [[#1	TEORIA, RISULTATI e cose FATTE|1	TEORIA, RISULTATI e cose FATTE]]
	- [[#1	TEORIA, RISULTATI e cose FATTE#1.1	Risultati delle Varie Prove ::XGBOD::|1.1	Risultati delle Varie Prove ::XGBOD::]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.1	\+Modelli e Parametri|1.1.1	\+Modelli e Parametri]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.2	Codice Modelli Supervisionati + Parametri XGBOD|1.1.2	Codice Modelli Supervisionati + Parametri XGBOD]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.3	XGBOD|1.1.3	XGBOD]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.4	XGBOD Parametri|1.1.4	XGBOD Parametri]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.5	Modelli Supervisionati|1.1.5	Modelli Supervisionati]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.6	Modelli Supervisionati + Parametri|1.1.6	Modelli Supervisionati + Parametri]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.7	Early Stopping|1.1.7	Early Stopping]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.8	XGBOD + ESN|1.1.8	XGBOD + ESN]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.9	Bathc Processing|1.1.9	Bathc Processing]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.10	Cross Validation|1.1.10	Cross Validation]]
		- [[#1.1	Risultati delle Varie Prove ::XGBOD::#1.1.11	Lista parametri con `Grid`|1.1.11	Lista parametri con `Grid`]]
	- [[#1	TEORIA, RISULTATI e cose FATTE#1.2	Domande Utili|1.2	Domande Utili]]
		- [[#1.2	Domande Utili#1.2.1	Uso di Standard Scaler|1.2.1	Uso di Standard Scaler]]
- [[#2	::Altra TEORIA::|2	::Altra TEORIA::]]
	- [[#2	::Altra TEORIA::#2.1	TimeSeries|2.1	TimeSeries]]
		- [[#2.1	TimeSeries#2.1.1	Analisi delle Time Series:|2.1.1	Analisi delle Time Series:]]
	- [[#2	::Altra TEORIA::#2.2	Threshold|2.2	Threshold]]
		- [[#2.2	Threshold#2.2.1	Come funziona il Threshold:|2.2.1	Come funziona il Threshold:]]
	- [[#2	::Altra TEORIA::#2.3	Algoritmi Convuluzionali|2.3	Algoritmi Convuluzionali]]
		- [[#2.3	Algoritmi Convuluzionali#2.3.1	Struttura di una Rete Convoluzionale:|2.3.1	Struttura di una Rete Convoluzionale:]]
	- [[#2	::Altra TEORIA::#2.4	Rocket|2.4	Rocket]]
		- [[#2.4	Rocket#2.4.1	::Come Funziona ROCKET:::|2.4.1	::Come Funziona ROCKET:::]]
		- [[#2.4	Rocket#2.4.2	Vantaggi di ROCKET:|2.4.2	Vantaggi di ROCKET:]]
		- [[#2.4	Rocket#2.4.3	Spiegazione Codice GitHub_Paper|2.4.3	Spiegazione Codice GitHub_Paper]]
		- [[#2.4	Rocket#2.4.4	**Codice**|2.4.4	**Codice**]]
		- [[#2.4	Rocket#Spiegazione del Codice:|Spiegazione del Codice:]]
	- [[#2	::Altra TEORIA::#2.5	::Teoria Paper Rocket::|2.5	::Teoria Paper Rocket::]]
		- [[#2.5	::Teoria Paper Rocket::#2.5.1	Funzionamento di ROCKET|2.5.1	Funzionamento di ROCKET]]
		- [[#2.5	::Teoria Paper Rocket::#2.5.2	Parametri di ROCKET|2.5.2	Parametri di ROCKET]]
		- [[#2.5	::Teoria Paper Rocket::#2.5.3	Vantaggi di ROCKET|2.5.3	Vantaggi di ROCKET]]
	- [[#2	::Altra TEORIA::#2.6	Risultati Ottenuti → Rocket su OP-SAT|2.6	Risultati Ottenuti → Rocket su OP-SAT]]
- [[#3	Rockad|3	Rockad]]
		- [[#2.6	Risultati Ottenuti → Rocket su OP-SAT#3.1.1	**Funzionamento di ROCKAD**|3.1.1	**Funzionamento di ROCKAD**]]
- [[#4	Funzionamento Paper|4	Funzionamento Paper]]


# 1	TEORIA, RISULTATI e cose FATTE

1. Ri-esecuzione in locale degli algoritmi
   1. Mancano qualcuno di quelli supervisionati → non funzionano
2. **XGBOD**
3. Ho modificato i valori di `X_train` e `X_test` con lo standard scaler di `sklearn.preprocessing`
4. **Ricerca degli iperparametri** → per trovare i migliori parametri
   1. `GridSearchCV`
5. Scelta delle *Caratteristiche* → NON FATTO
6. **Modifica dei Parametri** → Riduzione complessità del modello `n_estimators` e `max_depth`
7. **Lista degli estimatori** → modelli passati all'algoritmo per la parte non supervisionata
8. Ho aggiunto delle **metriche su Tempo** e uso della **Memoria**
   1. Rispettivamente tempo e memoria usata in fase di addestramento e predizione

> Il modello XGBoost di tipo supervisionato, si sviluppa con un processo iterativo di addestramento di alberi decisionali deboli (alberi decisionali poco profondi e quindi poco accurati), questi vengono combinati tra di loro portando un miglioramento progressivo delle prestazioni del modello.

> XGBoost è composto da pochi passi ma ripetuti iterativamente: come primo passo vengono calcolati i residui, la differenza tra le previsioni iniziali e i valori reali, questi sono i valori che vogliamo ridurre. Con questi valori il modello addestra un insieme di alberi decisionali deboli dove ognuno cerca di correggere questi valori migliorando le previsioni del modello precedente. Tutti gli alberi vengono aggiunti al modello complessivo di XGBoost (supervisionato) , che aggiorna le sue previsioni combinando tutti gli alberi precedentemente costruiti.

> Per regolare tutto questo processo sono applicate internamente tecniche di limitazione e regolazione per evitare un overfitting del modello. All'interno di XGBoost è presente anche una metrica chiamata \textit{tasso di apprendimento} che permette di decidere quanto un albero incide sul risultato finale minimizzando così gli errori di percorso.

Utilizzo di ::pipeline:: → Consentono di concatenare vari passaggi di preprocessing e addestramento del modello in un'unica sequenza ripetibile → ::quindi posso applicare più preprocessi diversi in un unica soluzione sequenziale grazie alla pipeline::

1. Una pipeline è composta da una serie di step, ognuno dei quali rappresenta una trasformazione sui dati.

## 1.1	Risultati delle Varie Prove ::XGBOD::

| Modalità       | Accuracy | Precision | Recall | F1    | MCC   | AUC_PR | AUC_ROC | N_score |
| -------------- | -------- | --------- | ------ | ----- | ----- | ------ | ------- | ------- |
| \+ModParamDiv  | 0.97     | 0.945     | 0.912  | 0.928 | 0.909 | 0.973  | 0.992   | 0.92    |
| EarlyStop(M+P) | 0.97     | 0.971     | 0.885  | 0.926 | 0.909 | 0.969  | 0.99    | 0.912   |
| Più Modelli    | 0.968    | 0.944     | 0.903  | 0.923 | 0.903 | 0.974  | 0.991   | 0.92    |
| \+Mod e Param  | 0.968    | 0.962     | 0.885  | 0.922 | 0.903 | 0.974  | 0.991   | 0.92    |
| Scaled         | 0.968    | 0.953     | 0.894  | 0.922 | 0.903 | 0.969  | 0.99    | 0.912   |
| Con Param      | 0.964    | 0.943     | 0.885  | 0.913 | 0.891 | 0.972  | 0.991   | 0.912   |
| Senza Param    | 0.962    | 0.935     | 0.885  | 0.909 | 0.886 | 0.977  | 0.992   | 0.912   |
| XGBOD + ESN    | 0.962    | 0.927     | 0.894  | 0.91  | 0.886 | 0.966  | 0.989   | 0.912   |
| Con Grid       | 0.947    | 0.989     | 0.761  | 0.86  | 0.839 | 0.898  | 0.945   | 0.969   |
| Totale         | 0.97     | 0.989     | 0.912  | 0.928 | 0.909 | 0.977  | 0.992   | 0.969   |

(::O::) → Migliori valori

(::O::) → Secondi migliori valori

### 1.1.1	\+Modelli e Parametri

|                                                               | Accuracy | Precision | Recall | F1    | MCC   | AUC_PR | AUC_ROC | N_score |
| ------------------------------------------------------------- | -------- | --------- | ------ | ----- | ----- | ------ | ------- | ------- |
| n_estimator=100<br>max_depth=3<br>learning_rate=0.1           | 0.968    | 0.944     | 0.903  | 0.923 | 0.903 | 0.974  | 0.991   | 0.92    |
| n_estimators=50         max_depth=3         learning_rate=0.1 | 0.968    | 0.962     | 0.885  | 0.922 | 0.903 | 0.974  | 0.991   | 0.92    |
| n_estimators=50        max_depth=3         learning_rate=0.2  | 0.964    | 0.935     | 0.894  | 0.914 | 0.892 | 0.973  | 0.99    | 0.912   |
| n_estimators=25         max_depth=3         learning_rate=0.2 | 0.968    | 0.962     | 0.885  | 0.922 | 0.903 | 0.972  | 0.99    | 0.929   |
| n_estimators=10        max_depth=3         learning_rate=0.5  | 0.966    | 0.936     | 0.903  | 0.919 | 0.898 | 0.969  | 0.99    | 0.928   |
| n_estimators=100        max_depth=3         learning_rate=0.2 | 0.97     | 0.945     | 0.912  | 0.928 | 0.909 | 0.973  | 0.992   | 0.92    |

(::O::) → Migliori metriche

(::O::) →Miglior Rapporto efficienza-metriche

+ ### Aggiunta di Metriche di Efficienza

   **Tempo di Addestramento**

```python
import time

start_time = time.time()
model.fit(X_train)
training_time = time.time() - start_time
print(f"Tempo di addestramento: {training_time} secondi")
```

   **Tempo di Inferenza**

```python
start_time = time.time()
y_predicted = model.predict(X_test)
inference_time = time.time() - start_time
print(f"Tempo di inferenza: {inference_time} secondi")
```

   **Uso della Memoria**

```python
from memory_profiler import memory_usage

# Funzione per l'addestramento del modello
def train_model():
    model.fit(X_train)

# Monitoraggio dell'uso della memoria durante l'addestramento
mem_usage = memory_usage(train_model)
print(f"Uso della memoria durante l'addestramento: {max(mem_usage)} MiB")

# Funzione per l'inferenza del modello
def inference_model():
    return model.predict(X_test)
```

### 1.1.2	Codice Modelli Supervisionati + Parametri XGBOD

```python
from pyod.models.xgbod import XGBOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM

# Definizione dei modelli unsupervised
unsupervised_models = [ KNN(),
                       LOF(),
                       ABOD(),
                       OCSVM()
                      ]
# Inizializza e addestra XGBOD
model = XGBOD( estimator_list=unsupervised_models,
              n_estimators = 50,
               max_depth = 3,
               learning_rate = 0.1,
               n_jobs = -1,
               random_state = SEED
             )

model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)

# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)
print(model, metrics)
```

Ho modificato:

- `n_estimators` → ho diminuito il numero di estimatori ossia il numero di alberi del modello
- `max_depth` → profondità di ciascun albero
- `learning_rate` → tasso di apprendimento del modello
- Se aumento necessita di un numero minore di estimatori (alberi)
- `random_state` → seme per la riproducibilità
- `n_jobs = -1` → imposta il numero di core della CPU utilizzati (Per velocizzare la compuazione)

#### 1.1.2.1	Codici Tutto

> `Import` **da fare prima di ogni codice**

```python
from pyod.models.xgbod import XGBOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM
import numpy as np
```

### 1.1.3	XGBOD

```python
from pyod.models.xgbod import XGBOD

# Inizializza e addestra XGBOD
model = XGBOD()
model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)
# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)

# Stampa i risultati
print(model, metrics)
```

### 1.1.4	XGBOD Parametri

```python
from pyod.models.xgbod import XGBOD

# Inizializza e addestra XGBOD
model = XGBOD(estimator_list=unsupervised_models,
              n_estimators=100,
              max_depth=3,
              learning_rate=0.2,
              n_jobs=-1,
              random_state=SEED
            )
model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)
# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)

# Stampa i risultati
print(model, metrics)
```

### 1.1.5	Modelli Supervisionati

```python
# Definizione dei modelli unsupervised
unsupervised_models = [ KNN(),
                       LOF(),
                       ABOD(),
                        OCSVM()
                    ]
# Inizializza e addestra XGBOD
model = XGBOD(estimator_list=unsupervised_models)

model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)
# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)

# Stampa i risultati
print(model, metrics)
```

### 1.1.6	Modelli Supervisionati + Parametri

```python
# Definizione dei modelli unsupervised
unsupervised_models = [ KNN(),
                       LOF(),
                       ABOD(),
                        OCSVM()
                    ]

# Inizializza e addestra XGBOD
model = XGBOD(estimator_list=unsupervised_models,
              n_estimators=100,
              max_depth=3,
              learning_rate=0.2,
              n_jobs=-1,
              random_state=SEED
            )

model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)

# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)
print("")
print(metrics)
```

### 1.1.7	Early Stopping

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Definizione dei modelli unsupervised
unsupervised_models = [ KNN(),
                       LOF(),
                       ABOD(),
                        OCSVM()
                    ]

# Divisione del dataset di allenamento per avere un set di validazione
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=SEED)

# Inizializzazione del modello
model = XGBOD(estimator_list=unsupervised_models, n_estimators=50, max_depth=3, learning_rate=0.2, n_jobs=-1, random_state=SEED)

best_score = -np.inf
patience = 10       # Numero di volte che il modello cercherà di migliorarsi
patience_counter = 0
n_iterations = 100      # Numero massimo di cicli del'allenamento

for i in range(n_iterations):  # Numero massimo di iterazioni
    model.fit(X_train_sub, y_train_sub)
    
    # Predizione sul set di validazione
    y_val_pred = model.predict(X_val)
    val_score = accuracy_score(y_val, y_val_pred)
    
    # Controllo early stopping
    if val_score > best_score:
        best_score = val_score
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at iteration {i}")
            break
    model.n_estimators += 1  # Incrementa il numero di stimatori per la prossima iterazione

# Predizione sul set di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)

# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)
print("")
print(metrics)
```

### 1.1.8	XGBOD + ESN

```python
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Definizione dei modelli unsupervised
unsupervised_models = [
    KNN(),
    LOF(),
    ABOD(),
    OCSVM()
]

# Creazione del reservoir
reservoir = Reservoir(units=1000, sr=0.95)  # sr: raggio spettrale
# Creazione del nodo di output per il readout
readout = Ridge(ridge=1e-5)
# Connessione del reservoir al readout per creare l'ESN
reservoir >> readout

# Pipeline di preprocessing
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('reservoir', reservoir)
])

# Trasformazione dei dati di addestramento e test con ESN
# Addestramento del modello
readout.fit(reservoir.run(X_train_scaled), X_train_scaled)  # Si allena il readout sugli stati del reservoir

# Predizione per il rilevamento di anomalie
X_train_transformed = reservoir.run(X_train_scaled)
X_test_transformed = reservoir.run(X_test_scaled)

# Creazione del modello XGBOD con parametri specificati
model = XGBOD(estimator_list=unsupervised_models, n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42)
# Uso le trasformazioni di ESN con il modello XGBOD
model.fit(X_train_transformed, y_train)

# Predizione sui dati di test
y_pred = model.predict(X_test_transformed)
y_predicted_score = model.decision_function(X_test_transformed)

# Valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)

# Stampa i risultati
print(f"Model: {model}")
print(f"Metrics: {metrics}")
```

### 1.1.9	Bathc Processing

```python
# Dividi il dataset in batch
n_batches = 10  # Specifica il numero di batch che vuoi
X_train_batches = np.array_split(X_train_scaled, n_batches)
y_train_batches = np.array_split(y_train, n_batches)

# Definizione dei modelli unsupervised
unsupervised_models = [ KNN(),
                       LOF(),
                       ABOD(),
                        OCSVM()
                    ]

# Inizializza i modelli per ciascun batch
models = []
for X_batch, y_batch in zip(X_train_batches, y_train_batches):
    # Inizializza e addestra il modello
    model = XGBOD(estimator_list=unsupervised_models,
                  n_estimators=100,
                  max_depth=3,
                  learning_rate=0.2,
                  n_jobs=-1,
                  random_state=SEED
                )
    model.fit(X_batch, y_batch)
    models.append(model)

# Prevedi gli outlier nel dataset di test e combinalo
y_pred_scores = np.zeros_like(X_test_scaled[:, 0], dtype=float)
for model in models:
    y_pred_scores += model.decision_function(X_test_scaled)

# Media dei punteggi di decisione
y_pred_scores /= n_batches
y_pred = (y_pred_scores > np.mean(y_pred_scores)).astype(int)

# Esegui la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_pred_scores)
print("")
print(metrics)
```

### 1.1.10	Cross Validation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score


# Preprocessing and model pipeline
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', XGBOD(n_estimators=50, max_depth=3, learning_rate=0.1))
])

# Cross-validation with pipeline
scores = cross_val_score(pipeline, X_train_scaled, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Cross-validation scores: {scores}")
print(f"Mean ROC AUC score: {np.mean(scores)}")

# Train and evaluate model
pipeline.fit(X_train_scaled, y_train)
y_pred = pipeline.predict(X_test_scaled)
y_predicted_score = pipeline.decision_function(X_test_scaled)

metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)
print(pipeline.named_steps['classifier'], metrics)
```

### 1.1.11	Lista parametri con `Grid`

```python
from sklearn.model_selection import RandomizedSearchCV
from pyod.models.xgbod import XGBOD
import numpy as np

# Definizione della griglia di parametri
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

# Inizializza il modello
model = XGBOD()

# Randomized search con meno iterazioni e parallelizzazione
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# Migliori parametri trovati
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Riaddestramento del modello con i migliori parametri
model = XGBOD(**best_params)
model.fit(X_train_scaled, y_train)

# Prevedi gli outlier nel dataset di test
y_pred = model.predict(X_test_scaled)
y_predicted_score = model.decision_function(X_test_scaled)

# Eseguiamo la valutazione delle metriche
metrics = evaluate_metrics(y_test, y_pred, y_predicted_score)

# Stampa i risultati
print(model, metrics)
```

## 1.2	Domande Utili

- Cosa devo fare della parte sperimentale?
- Come devo efficientare un algoritmo? (Ho modificato parametri e provato più tecniche)
   - Devo usare ESN come nel paper di Valerio?
- è sufficiente la parte di efficientamento che ho fatto?
- Per testare gli algoritmi su raspberry?

### 1.2.1	Uso di Standard Scaler

In breve

Utilizziamo lo **Standard Scaler** con gli algoritmi supervisionati per:

1. **Uniformare le Scale**: Garantire che tutte le caratteristiche abbiano scale simili, evitando che una caratteristica domini il processo di addestramento.
2. **Convergenza più Veloce**: Facilitare l'ottimizzazione accelerando la convergenza di algoritmi come il gradiente discendente.
3. **Migliore Performance**: Migliorare la performance di algoritmi che dipendono dalle distanze (es. SVM, KNN).
4. **Prevenire Overfitting**: Rendere il modello più robusto a nuovi dati, riducendo la sensibilità ai valori estremi.
5. **Interpretazione dei Coefficienti**: Rendere comparabili i coefficienti in modelli lineari.

Questi benefici aiutano a migliorare l'accuratezza, la stabilità e la velocità del modello.

---

# 2	Altra TEORIA

## 2.1	TimeSeries

> Le **Time Series** (o serie temporali) sono una sequenza di dati raccolti o registrati a intervalli di tempo successivi. Ogni punto nella serie è associato a un timestamp che indica il momento in cui il dato è stato raccolto. Le serie temporali sono comunemente utilizzate in vari campi per analizzare come una quantità cambia nel tempo.

- **Continuo o Discreto**: I dati possono essere raccolti in maniera continua (es. temperatura ogni secondo) o discreta (es. chiusura giornaliera di una borsa valori).
- **Trend**: La componente che rappresenta il comportamento a lungo termine della serie.
- **Stagionalità**: La componente che riflette le variazioni periodiche (es. vendite più alte durante le festività).
- **Rumore**: La componente casuale che rappresenta le fluttuazioni imprevedibili.

### 2.1.1	Analisi delle Time Series

L'analisi delle serie temporali comprende una varietà di tecniche per comprendere, modellare e prevedere i dati nel tempo. Alcune tecniche comuni includono:

- **Decomposizione**: Separare la serie nei suoi componenti (trend, stagionalità, rumore).
- **Smoothing**: Applicare metodi come la media mobile per ridurre il rumore.
- **Modellazione**: Utilizzare modelli come ARIMA, Exponential Smoothing, o tecniche di Machine Learning per fare previsioni future.

## 2.2	Threshold

> Il **threshold** (o soglia) è un valore di riferimento utilizzato in vari algoritmi di rilevamento delle anomalie e tecniche di classificazione per decidere quando una predizione dovrebbe essere classificata come positiva o negativa, oppure quando un evento dovrebbe essere considerato anomalo.

### 2.2.1	Come funziona il Threshold

1. #### **Rilevamento delle Anomalie**:

   Quando si usa il threshold nel contesto del rilevamento delle anomalie, si calcola un punteggio di anomalia per ogni punto dati. Questo punteggio riflette quanto il dato sia lontano dal comportamento normale. Se il punteggio supera un certo valore di soglia, il dato viene considerato anomalo.

   - **Calcolo del Punteggio di Anomalia**: Gli algoritmi generano un punteggio che misura la "stranezza" di ogni dato.
   - **Impostazione del Threshold**: Si sceglie un valore soglia che separa i dati normali dagli anomali. Ad esempio, un punteggio di anomalia superiore a 0.8 potrebbe indicare un'anomalia.
2. #### **Classificazione Binaria**:

   In un modello di classificazione binaria, il threshold viene utilizzato per determinare il punto di divisione tra le due classi. Generalmente, il modello produce una probabilità di appartenenza a una delle due classi. Il valore di soglia definisce quale probabilità considerare come positiva.

   - **Probabilità**: Supponiamo che un modello di classificazione dia una probabilità tra 0 e 1 che un'osservazione appartenga alla classe positiva.
   - **Impostazione del Threshold**: Comunemente si usa un threshold di 0.5, dove valori superiori a 0.5 vengono classificati come positivi e valori inferiori come negativi. Il threshold può essere regolato in base alle esigenze del problema (ad esempio, per aumentare il recall o la precision).

```python
# Supponiamo di avere punteggi di anomalia per un set di dati
anomaly_scores = [0.1, 0.2, 0.3, 0.9, 0.4, 0.8]

# Impostiamo un threshold
threshold = 0.5

# Classifichiamo le anomalie
anomalies = [score > threshold for score in anomaly_scores]
print(anomalies)  # Output: [False, False, False, True, False, True]
```

> **Precision vs. Recall**: La scelta del threshold influenza il bilanciamento tra precision e recall

## 2.3	Algoritmi Convuluzionali

> Gli **algoritmi convoluzionali**, noti anche come **reti neurali convoluzionali** (CNN), sono una classe di reti neurali profonde particolarmente efficaci nell'elaborazione di dati con una struttura a griglia, come immagini e serie temporali. Sono progettati per riconoscere pattern e caratteristiche locali nei dati utilizzando operazioni di convoluzione.

1. **Convoluzioni**:
   - **Filtri**: Le CNN utilizzano filtri (o kernel) che passano sopra l'input (come un'immagine) per estrarre caratteristiche locali. Ogni filtro produce una mappa di attivazione che rappresenta la risposta del filtro a diverse parti dell'input.
   - **Stride**: Determina di quanti pixel il filtro si sposta sopra l'input.
   - **Padding**: Aggiunge bordi attorno all'input per mantenere le dimensioni dopo l'applicazione del filtro.
2. **Pooling**:
   - **Max Pooling**: Riduce le dimensioni dell'input prendendo il valore massimo in una finestra specifica.
   - **Average Pooling**: Riduce le dimensioni dell'input prendendo la media dei valori in una finestra specifica.
   - **Scopo**: Pooling aiuta a ridurre la dimensionalità e a rendere l'algoritmo meno sensibile alle traslazioni.
3. **Strati Completamente Connessi**:
   - **Dense Layers**: Dopo una serie di strati convoluzionali e di pooling, i dati vengono appiattiti e passati attraverso uno o più strati completamente connessi per la classificazione o la regressione.
4. **Attivazione**:
   - **ReLU (Rectified Linear Unit)**: Funzione di attivazione non lineare comunemente utilizzata che introduce non linearità, permettendo alla rete di apprendere pattern complessi.

### 2.3.1	Struttura di una Rete Convoluzionale

Una CNN tipica potrebbe avere la seguente struttura:

1. **Strato di Input**: Riceve l'immagine o il dato grezzo.
2. **Strati Convoluzionali**: Applicano i filtri per estrarre caratteristiche locali.
3. **Strati di Pooling**: Riduzione dimensionale.
4. **Strati Aggiuntivi**: Ulteriori strati convoluzionali e di pooling per estrarre caratteristiche più profonde.
5. **Strato di Flattening**: Appiattisce i dati per prepararli per gli strati completamente connessi.
6. **Strati Completamente Connessi**: Per la classificazione finale o altre previsioni.

---

## 2.4	Rocket

> **ROCKET** (Random Convolutional Kernels) è una tecnica di classificazione delle serie temporali eccezionalmente veloce e accurata. È stata progettata per essere semplice da implementare e scalabile per grandi set di dati

### 2.4.1	Come Funziona ROCKET

1. **Generazione di Kernel Convoluzionali Casuali**:
   - ROCKET genera un gran numero di kernel convoluzionali con parametri scelti casualmente, come lunghezza del kernel, pesi e bias.
   - Questi kernel sono usati per ::filtrare le serie temporali in ingresso::, producendo una serie di caratteristiche (features).
2. **Convoluzione e Pooling**:
   - Ogni kernel convoluzionale viene applicato alla serie temporale, generando una mappa di attivazione.
   - ROCKET utilizza operazioni di pooling, come max pooling e il percentuale di pooling (ad esempio, il 75° percentile), per ottenere statistiche riassuntive da queste mappe di attivazione.
3. **Creazione delle Caratteristiche**:
   - Le statistiche riassuntive ottenute dal passaggio di pooling sono utilizzate come caratteristiche (features) per la classificazione.
   - Questa procedura viene ripetuta per tutti i kernel convoluzionali, generando un gran numero di caratteristiche.
4. **Classificazione**:
   - Le caratteristiche estratte vengono utilizzate come input per un modello di classificazione lineare, come il Ridge Classifier.
   - Questo modello apprende a distinguere tra le diverse classi delle serie temporali in base alle caratteristiche generate.

### 2.4.2	Vantaggi di ROCKET

- **Velocità**: Grazie all'uso di kernel convoluzionali casuali e a un modello di classificazione lineare, ROCKET è estremamente veloce da addestrare.
- **Accuratezza**: Nonostante la semplicità e la velocità, ROCKET ha dimostrato di essere molto accurato in vari compiti di classificazione delle serie temporali.
- **Scalabilità**: È in grado di gestire grandi set di dati grazie alla sua efficienza computazionale.

```python
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

# Inizializzazione e adattamento di ROCKET
rocket = Rocket()
X_train_transformed = rocket.fit_transform(X_train)
X_test_transformed = rocket.transform(X_test)

# Utilizzo di un modello di classificazione, come RidgeClassifier
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transformed, y_train)

# Predizione
y_pred = classifier.predict(X_test_transformed)
```

> Kernel → Generare un kernel nel contesto delle reti neurali convoluzionali significa creare un filtro che verrà utilizzato per analizzare una parte della serie temporale o immagine

**Come Funziona:**

1. **Parametri del Kernel**:
   - **Lunghezza**: La dimensione del kernel, ovvero quanti punti dati coprirà in un'operazione.
   - **Pesi**: I valori all'interno del kernel che vengono moltiplicati per i valori dei dati d'ingresso.
   - **Bias**: Un valore aggiuntivo sommato al risultato dell'operazione di convoluzione.
   - **Dilatazione**: Determina quanto distanti sono i campioni presi dai dati d'ingresso durante l'applicazione del kernel.
   - **Padding**: Aggiunge zeri ai bordi dei dati d'ingresso per mantenere le dimensioni dopo l'applicazione del kernel.
2. **Processo di Convoluzione**:
   - **Applicazione del Kernel**: Si passa il kernel sui dati d'ingresso, spostandolo di un passo (stride) ogni volta, e si calcola un valore per ogni posizione del kernel.
   - **Estrazione delle Caratteristiche**: Il risultato dell'applicazione del kernel è una mappa di attivazione che evidenzia dove il pattern rappresentato dal kernel è presente nei dati d'ingresso.
3. **Generazione Casuale**:
   - Nel caso di ROCKET, i parametri del kernel sono scelti in modo casuale, creando così una varietà di kernel che possono catturare diversi pattern nei dati.
   - Questo permette di estrarre una vasta gamma di caratteristiche dai dati d'ingresso, che sono poi utilizzate per l'addestramento di un modello di classificazione.

### 2.4.3	Spiegazione Codice GitHub_Paper

Questo codice è parte dell'implementazione dell'algoritmo ROCKET, che è progettato per la classificazione delle serie temporali utilizzando kernel convoluzionali casuali. Ecco una spiegazione di come funziona il codice:

#### 2.4.3.1	Funzioni Principali

1. **generate_kernels**:
   - Genera kernel convoluzionali casuali con parametri come lunghezza del kernel, pesi, bias, dilatazione e padding.
   - I kernel generati vengono utilizzati per convolvere le serie temporali e ottenere caratteristiche utili.
2. **apply_kernel**:
   - Applica un kernel a una serie temporale.
   - Calcola due statistiche: `PPV` (Proportion of Positive Values) e `MAX` (massimo valore ottenuto dalla convoluzione).
3. **apply_kernels**:
   - Applica tutti i kernel generati alle serie temporali in input.
   - Estrae caratteristiche per ciascuna serie temporale.

### 2.4.4	**Codice**

+ Codice trovato nel PAPER

```python
# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import numpy as np
from numba import njit, prange

@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):

    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X
```

   ### Spiegazione del Codice:

   1. **Genera Kernel Convoluzionali**:
      - La funzione `generate_kernels` crea kernel convoluzionali casuali con varie lunghezze, pesi, bias, dilatazioni e padding.
   2. **Applica i Kernel ai Dati**:
      - La funzione `apply_kernel` applica un singolo kernel alla serie temporale.
      - La funzione `apply_kernels` applica tutti i kernel generati alle serie temporali, producendo un set di caratteristiche (features) per ciascuna serie temporale.
   3. **Modello di Rilevamento delle Anomalie**:
      - Usa Isolation Forest per rilevare anomalie. Isolation Forest è un algoritmo di machine learning non supervisionato utilizzato per individuare dati anomali all'interno di un dataset.
      - Il modello viene addestrato sui dati trasformati e scala le caratteristiche usando StandardScaler per migliorare le prestazioni del modello.
   4. **Thresholding**:
      - Ottiene i punteggi di anomalia e applica una soglia per identificare le anomalie. I dati che superano questa soglia vengono considerati anomalie.
+ Altro Codice

```python
import numpy as np
from numba import njit, prange


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)
    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]
        _weights = np.random.normal(0, 1, _length)
        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()
        biases[i] = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        a1 = b1
    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0
    _max = np.NINF
    end = (input_length + padding) - ((length - 1) * dilation)
    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1
    return _ppv / output_length, _max


@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel=True,
      fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64)            # 2 features per kernel
    for i in prange(num_examples):
        a1 = 0          # for weights
        a2 = 0          # for features
        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2
            _X[i, a2:b2] = apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])
            a1 = b1
            a2 = b2
    return _X


def train_step(actual_data, num_kernels, seed):
    # np.random.seed(seed)
    input_length = actual_data.shape[1]
    kernels = generate_kernels(input_length, num_kernels)
    data = actual_data.astype(np.float64)
    modified_x = apply_kernels(data, kernels)
    return modified_x
```

---

## 2.5	::Teoria Paper Rocket::

> Rocket è un algoritmo che si basa su kernel convoluzionali che filtrano le serie temporali in ingresso

**::Kernel::** → matrice di pesi usata per eseguire operazioni di filtraggio per estrarre caratteristiche specifiche. Opera tramite moltiplicazioni

**::Pooling::** → Queste operazioni aiutano a ridurre la dimensionalità dei dati convoluti e a migliorare la robustezza e l'efficienza del modello.

- Max Pooling → seleziona il valore massimo all'interno di una finestra di dati

> I principali vantaggi dell'utilizzo di ROCKET per la classificazione delle serie temporali comprendono:

1. **Velocità e scalabilità**: ROCKET è eccezionalmente veloce, in grado di apprendere da grandi insiemi di dati (ad esempio, 1 milione di serie temporali) in un tempo significativamente più breve rispetto ai metodi tradizionali. Ad esempio, può raggiungere un'accuratezza simile a quella di Proximity Forest in poco più di un'ora, mentre Proximity Forest richiede più di 16 ore per la stessa quantità di dati. Una variante ristretta di ROCKET può imparare dallo stesso set di dati in meno di un minuto, il che lo rende circa 100 volte più veloce.
2. **Alta precisione**: ROCKET ha dimostrato prestazioni competitive rispetto ai metodi più avanzati, ottenendo il miglior punteggio medio su 85 dataset “bake off”. Ha dimostrato di superare diversi classificatori esistenti, tra cui BOSS, Shapelet Transform e HIVE-COTE, oltre a metodi più recenti come InceptionTime e TS-CHIEF .
3. **Flessibilità con i parametri del kernel**: ROCKET utilizza un numero molto elevato di kernel convoluzionali casuali con diverse configurazioni (lunghezze, dilatazioni e padding casuali). Questa varietà consente di catturare un'ampia gamma di caratteristiche dai dati delle serie temporali senza la necessità di un'estesa messa a punto, il che indica la robustezza del metodo.
4. **Complessità di addestramento lineare**: La ::complessità:: di addestramento di ROCKET è ::lineare sia per la lunghezza delle serie temporali sia per il numero di esempi di addestramento::, il che lo rende particolarmente adatto a grandi insiemi di dati.
5. **Parallelizzazione**: ROCKET è stato progettato per essere naturalmente parallelo, consentendogli di sfruttare più core di CPU o GPU, il che ne aumenta ulteriormente la velocità e l'efficienza.

Nel complesso, ROCKET combina velocità, precisione e scalabilità, rendendolo uno strumento potente per le attività di classificazione delle serie temporali.

> ROCKET supera in modo significativo i metodi tradizionali in termini di efficienza computazionale sotto diversi punti di vista:

1. **Tempo di addestramento**: ROCKET richiede un tempo di addestramento notevolmente inferiore rispetto ai classificatori tradizionali. Ad esempio, quando è stato testato sul dataset “bake off” con il più grande set di formazione (ElectricDevices, con 8926 esempi di formazione), ROCKET ha impiegato solo 6 minuti e 33 secondi, mentre MrSEQL ha richiesto 31 minuti, Proximity Forest 1 ora e 35 minuti e TS-CHIEF 2 ore e 24 minuti. Questa tendenza continua con altri set di dati, dove ROCKET mostra costantemente tempi di addestramento più rapidi.
2. **Scalabilità**: ROCKET è stato progettato per gestire in modo efficiente set di dati di grandi dimensioni. Può apprendere da 1 milione di serie temporali in poco più di un'ora, ottenendo un'accuratezza simile a quella di Proximity Forest, che richiede più di 16 ore per la stessa quantità di dati. 
Grazie a:
   1. Elaborazione Parallela
   2. Partizionamento dei Dati → Divide i dati in parti più piccole e gestibili, consentendo un accesso e una manipolazione più veloci
   3. Indicizzazione dei Dati → Implementa tecniche di indicizzazione per migliorare la velocità di accesso ai dati, riducendo il tempo necessario per recuperare informazioni specifiche.
   4. Architetture Distribuite
   5. Elaborazione in Tempo Reale → Supporta l'elaborazione in tempo reale dei dati, utile per applicazioni che richiedono risposte immediate
3. **Complessità lineare**: La complessità dell'addestramento di ROCKET è lineare rispetto alla lunghezza della serie temporale e al numero di esempi di addestramento. Questa scalatura lineare lo rende particolarmente efficiente per i dataset di grandi dimensioni, a differenza di molti metodi tradizionali che possono presentare una scalatura quadratica o peggiore.
4. **Utilizzo delle risorse**: ROCKET può essere parallelizzato in modo efficace, consentendogli di utilizzare più core di CPU o GPU, il che ne aumenta l'efficienza di calcolo. Questa capacità è particolarmente vantaggiosa quando si lavora con grandi insiemi di dati, in quanto può ridurre significativamente il tempo di elaborazione.
5. **Riduzione della regolazione degli iperparametri**: A differenza di molti metodi tradizionali che richiedono un'ampia regolazione degli iperparametri, ROCKET semplifica questo processo utilizzando un gran numero di kernel convoluzionali casuali senza la necessità di apprendere i pesi dei kernel. Ciò riduce il carico computazionale associato all'addestramento e alla regolazione del modello.

Nel complesso, il design di ROCKET consente di ottenere un'elevata accuratezza riducendo al minimo il tempo di addestramento e il consumo di risorse, rendendolo un'alternativa più efficiente ai tradizionali metodi di classificazione delle serie temporali.

ROCKET (Random Convolutional Kernel Transform) è un metodo innovativo per la classificazione delle serie temporali che utilizza un approccio basato su kernel convoluzionali casuali. Ecco una spiegazione dettagliata su come funziona ROCKET e sui suoi parametri principali:

### 2.5.1	Funzionamento di ROCKET

1. **Generazione di Kernel Convoluzionali Casuali**:
   - ROCKET ::utilizza un gran numero di kernel convoluzionali casuali::. A differenza delle reti neurali convoluzionali tradizionali, dove i pesi dei kernel vengono appresi durante l'addestramento, in ROCKET i pesi sono generati casualmente e rimangono fissi.
   - Ogni kernel ha parametri casuali, inclusi la lunghezza, la dilatazione e il padding, il che consente di ::catturare una vasta gamma di caratteristiche dalle serie temporali.::
2. **Applicazione dei Kernel**:
   - Ogni kernel viene applicato a ciascuna serie temporale per generare una ::mappa delle caratteristiche::. L'operazione di convoluzione comporta un prodotto scalare mobile tra un kernel e una serie temporale di input.
   - La convoluzione ::produce un vettore di caratteristiche:: che rappresenta l'output del kernel applicato alla serie temporale.
3. **Estrazione delle Caratteristiche**:
   - ROCKET estrae le caratteristiche da ciascuna serie temporale utilizzando i kernel casuali. Queste caratteristiche vengono poi utilizzate come ::input per un classificatore::, tipicamente una *regressione logistica*, che è scalabile e veloce per grandi dataset.
4. **Classificazione**:
   - Una volta che le caratteristiche sono state estratte, un classificatore (come la regressione logistica) viene ::addestrato su queste caratteristiche:: per effettuare la classificazione delle serie temporali.

### 2.5.2	Parametri di ROCKET

1. **Numero di Kernel (k)**:
   - Il principale iperparametro di ROCKET è il numero di kernel convoluzionali da utilizzare. Il valore predefinito è 10.000, ma può essere modificato in base alle esigenze. Un ::numero maggiore:: di kernel tende a ::migliorare l'accuratezza della classificazione::, ma aumenta anche il tempo di calcolo.
   - La complessità del trasformazione è lineare rispetto a k, quindi un numero maggiore di kernel comporta un aumento proporzionale del tempo di calcolo .
2. **Lunghezza del Kernel**:
   - Ogni kernel ha una lunghezza casuale, che ::determina quanto della serie temporale viene considerato durante la convoluzione::. La lunghezza può influenzare la capacità del kernel di catturare caratteristiche specifiche delle serie temporali.
3. **Dilatazione**:
   - La dilatazione è un parametro che ::controlla la distanza tra i punti campionati nel kernel::. ROCKET utilizza una varietà di dilatazioni per ogni kernel, il che consente di catturare caratteristiche a diverse scale temporali.
4. **Padding**:
   - Il padding determina come vengono ::gestiti i bordi della serie temporale:: durante la convoluzione. ROCKET applica padding casuale, il che significa che i kernel possono essere applicati in modo flessibile alle serie temporali.
5. **Bias**:
   - Ogni kernel può avere un termine di bias casuale, che aggiunge un ::ulteriore livello di flessibilità:: nella rappresentazione delle caratteristiche.

### 2.5.3	Vantaggi di ROCKET

- **Efficienza Computazionale**: ROCKET è progettato per essere estremamente veloce e scalabile, in grado di gestire grandi dataset in tempi ridotti.
- **Robustezza**: La varietà di kernel e la loro natura casuale consentono a ROCKET di generalizzare bene su nuovi problemi senza necessità di un'ampia messa a punto degli iperparametri.
- **Semplicità**: Con un solo iperparametro principale (il numero di kernel), ROCKET riduce la complessità associata alla messa a punto rispetto ad altri metodi di classificazione delle serie temporali.

In sintesi, ROCKET combina un approccio innovativo con kernel casuali e un design efficiente per fornire un metodo potente e veloce per la classificazione delle serie temporali.

## 2.6	Risultati Ottenuti → Rocket su OP-SAT

| Tipo               | **Accuracy** | **Precision** | **Recall** | **F1** | **MCC** | **AUC_PR** | **AUC_ROC** | **N_Score** |
| ------------------ | ------------ | ------------- | ---------- | ------ | ------- | ---------- | ----------- | ----------- |
| Unsupervised       | 0.792        | 0.541         | 0.177      | 0.219  | 0.414   | 0.65       | 0.487       |             |
| Supervised         | \-           | \-            | \-         | \-     | \-      | \-         | \-          | \-          |
| XGBOD              | 0.862        | 0.803         | 0.469      | 0.0892 | 0.843   | 0.723      | 0.907       | 0.664       |
| KNN                | 0.826        | 0.684         | 0.345      | 0.459  | 0.399   | 0.556      | 0.817       | 0.531       |
| LogisticRegression | 0.934        | 0.855         | 0.832      | 0.843  | 0.801   | 0.905      | 0.968       | 0.841       |
| RidgeRegression    | 0.819        | 0.758         | 0.221      | 0.342  | 0.342   | 0.54       | 0.779       | 0.504       |
| Total              | 0.934        | 0.855         | 0.832      | 0.843  | 0.843   | 0.905      | 0.968       | 0.841       |

# 3	Rockad

### 3.1.1	**Funzionamento di ROCKAD**

1. **Trasformazione Rocket**:
   - Converte i dati originali in un insieme di caratteristiche (feature space) adatto per il rilevamento di anomalie, utilizzando trasformazioni convolutive basate su kernel.
2. **Power Transformation**:
   - Applica una trasformazione non lineare per migliorare la distribuzione dei dati e ridurre la skewness.
3. **Ensemble di stimatori basati su Nearest Neighbors (NN)**:
   - Costruisce più modelli utilizzando il bootstrap aggregating (bagging).
   - Ogni modello è addestrato su un sottocampione diverso del dataset trasformato.
4. **Calcolo dello score di anomalia**:
   - Ogni stima fornisce un punteggio. I punteggi finali sono ottenuti mediando le previsioni di tutti gli stimatori.
5. **Rilevamento finale**:
   - Utilizza un classificatore **NearestNeighborOCC** per etichettare i punti dati come normali (1) o anomali (-1).

---

# 4	Funzionamento Paper

ROCKAD (Random Convolutional Kernel Transform Anomaly Detector) è un metodo per il rilevamento delle anomalie nelle serie temporali. L'idea chiave è quella di usare ROCKET, un classificatore di serie temporali, come estrattore di feature non supervisionato per trasformare i dati e addestrare un singolo KNN o un ensemble di rilevatori di anomalie KNN.

Ecco come funziona ROCKAD in dettaglio:

- **Estrazione di feature:** ROCKAD utilizza ROCKET per generare un insieme di kernel convoluzionali con parametri casuali. Questi kernel vengono spostati lungo le serie temporali per estrarre due feature: la somma ponderata massima degli elementi dei kernel e la proporzione di somme ponderate positive.
- **Trasformazione:** Le feature estratte da ROCKET vengono poi trasformate utilizzando un power transformer.
- **Rilevamento di anomalie:** Infine, un KNN o un ensemble di KNN viene addestrato sulle feature trasformate per dedurre un punteggio di anomalia per le serie temporali.

ROCKAD ha due varianti principali:

- **ROCKAD(1):** Utilizza un singolo KNN per il rilevamento delle anomalie.
- **ROCKAD(n):** Utilizza un ensemble di n KNN per il rilevamento delle anomalie.

I parametri principali di ROCKAD sono:

- **K:** Il numero di kernel convoluzionali generati da ROCKET. Il valore predefinito è 10.000.
- **n:** Il numero di estimatori KNN nell'ensemble ROCKAD(n). Il valore predefinito è 10.
- **k:** Il numero di vicini più vicini utilizzati per calcolare il punteggio di anomalia. Il valore predefinito è 5.

I parametri possono essere ottimizzati per migliorare le prestazioni di ROCKAD su set di dati specifici.

La scelta tra ROCKAD(1) e ROCKAD(n) dipende dalle dimensioni del set di dati e dalle risorse computazionali disponibili.

ROCKAD ha dimostrato di essere un metodo efficace per il rilevamento delle anomalie nelle serie temporali univariate e multivariate.

NearestNeighborOCC è una classe definita nel codice sorgente fornito che implementa un rilevatore di valori outlier basato sull'algoritmo del vicino più prossimo.1

Dato che ROCKAD usa un ensemble di KNN per il rilevamento delle anomalie, NearestNeighborOCC serve a fornire un metodo aggiuntivo per valutare l'anomalia di una serie temporale. Invece di basarsi solo sulla distanza media dai k vicini più prossimi, NearestNeighborOCC calcola un punteggio di anomalia basato sul rapporto tra due distanze:

- La distanza tra la serie temporale in esame e il suo vicino più prossimo.
- La distanza tra il vicino più prossimo e il suo vicino più prossimo.

Se questo rapporto è inferiore o uguale a 1, la serie temporale viene classificata come normale. Altrimenti, viene classificata come anomala.2

In sostanza, NearestNeighborOCC introduce un ulteriore livello di analisi per la classificazione delle anomalie, affiancando il metodo di ensemble di KNN utilizzato da ROCKAD. L'utilizzo di metodi multipli può contribuire a una maggiore robustezza e accuratezza nel rilevamento di anomalie.

[[Risultati]]
