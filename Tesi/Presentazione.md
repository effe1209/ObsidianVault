# 1	Introduzione

Con l'esplorazione dello spazio tramite il lancio di molti satelliti, un aspetto fondamentale è il rilevamento delle anomalie a bordo.

Fino ad adesso i satelliti necessitava di un **grande** numero di **comunicazioni** con la base per mandare le **segnalazioni.**

Queste spesso risultavano essere **false anomalie** causate da un’analisi statica e **soglie fisse**, non avendo quindi un reale riscontro con **l’ambiente dinamico dello spazio** che rendeva inefficienti questi controlli.

Tramite l'intelligenza artificiale siamo riusciti a portare un **rilevamento** delle anomalie **adattivo** in base ai cambiamenti dell'ambiente.

Nonostante i suoi vantaggi questo approccio porta con se nuove sfide e nuovi problemi, come il **consumo di banda** nel caso di un algoritmo allenato nella base operativa e spedito successivamente o la **scarsa potenza di calcolo** dei processori satellitari che rendono difficile l'addestramento dei modelli a bordo.

---

# 2	Obbiettivo

L'obbiettivo consiste nel trovare un algoritmo che possa essere **eseguito a bordo** dei satelliti e quindi che abbia un buon **compromesso** tra efficacia ed efficienza.

La realizzazione di questo obiettivo porterebbe ad un **aumento** della **vita dei satelliti** e ad un **consumo minore di banda** dato il numero minore e la maggiore precisione delle comunicazioni.

Per il nostro scopo partendo dall'analisi dei dati reali contenuti nei dataset OPS_SAT e NASA, **confrontiamo** i **modelli presenti** fino ad oggi per il rilevamento delle anomalie e il modello **ROCKET**, che non è stato ancora utilizzato in questo contesto. Abbiamo scelto questo modello dato il **basso costo computazionale** e le **performance dichiarate** nel suo paper.

Lo scopo di questo confronto è capire se ROCKET può essere una **valida alternativa** agli algoritmi utilizzati attualmente a bordo dei satelliti e se potrebbe essere utile anche come **base per ricercatori** futuri.

---

# 3	Background - 2

**DESCRIZIONE DATASET**

Come detto in precedenza, siamo partiti dai dataset, in particolare da **OPS_SAT**, che possiede dati formattati in maniera più **semplice** permettendone un **utilizzo facilitato**. OPS_SAT possiede anche i dati etichettati.

Al contrario il dataset **NASA** non ha i dati etichettati e presenta dei **dati rumorosi** e molto **variabili** data la provenienza da **più missioni**, questo dataset rappresenta un contesto **più realistico**.

**TIPI DI ANOMALIE**

All'interno di questi dataset possiamo trovare tre diversi tipi di anomalie, quelle **puntuali**, quelle **contestuali** e quelle **collettive**.

- **Point** Anomalies: ossia sono **singoli punti** che si discostano dal resto dei dati
- **Contextual** Anomalies: sono dati che in un **contesto** specifico risultano anomali
- **Collective** Anomalies: sono **gruppi** di dati che presi tutti insieme rappresentano un comportamento anomalo.

**MISURE DI VALUTAZIONE**

Nei test effettuati che vedremo abbiamo utilizzato delle **metriche** di valutazione che ci permettono di **confrontare** gli algoritmi in modo **oggettivo**.

In particolare ci interessano:

- l'accuratezza
- La precisione
- Il Recupero
- La metrica **F1**: che rappresenta la **media tra precisione e richiamo**

---

# 4	Background - 3

**XGBOD**

Uno degli algoritmi più utilizzati per il rilevamento delle anomalie è XGBOD.

XGBOD è un modello per il rilevamento delle anomalie **supervisionato** che si compone in tre fasi: nella prima vengono **applicati algoritmi** di rilevamento di anomalie ai dati per ottenere **punteggi** di anomalie; nella seconda questi punteggi sono **filtrati** per tenere solo quelli **utili** e per finire viene allenato **XGBoost** con questi punteggi per fare le **previsioni** sui dati.

XGBoost è composto da un processo **iterativo** che addestra **alberi** decisionali deboli, ovvero poco profondi.

Dalla tabella possiamo vedere che il miglior risultato è quello che utilizza **più modelli** e **iperparametri modificati**.

Durante il processo sono state applicate tecniche di regolazione e limitazione per evitare un overfitting del modello.

![image.png](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/8b7eb68a-96a9-4247-b3e9-bd1dcbe7d4eb)

---

# 5	Metodi - 1

ROCKET (::RandOm Convolutional KErnel Transform::) è un algoritmo **convoluzionale** che lavora con le **timeseries**, ossia una sequenza di dati registrati ad **intervalli di tempo** consecutivi.

ROCKET è stato selezionato per la sua grande capacità di **estrarre caratteristiche** importanti dalle serie temporali e per **l’efficienza** di utilizzo.

ROCKET ha un funzionamento piuttosto semplice, si divide in due parti fondamentali:

1. Nella prima i dati vengono appiattiti e fatti passare attraverso **kernel** convoluzionali **casuali**, questi sono paragonabili a dei **filtri** che applicati ai dati restituiscono una **serie di features**.
2. Nella seconda sono utilizzate **tecniche di pooling** che servono per ridurre la **dimensione** delle features e rendere l'algoritmo più stabile;

![image.png|607x458|400](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/c63963da-ae74-4304-b47e-6e206b7644ec)

Questo processo lo possiamo osservare nella prima figura, dove abbiamo la timeseries alla quale applichiamo ROCKET e in uscita abbiamo la serie di features.

Nella seconda figura invece vediamo che ROCKET scorre la timeseries segmentandola in **finestre** di dimensione uguale tra loro.

Le **caratteristiche estratte** da ciascuna finestra vengono successivamente utilizzate da un **classificatore**, che elabora le informazioni per fornire una **predizione sull’intera serie temporale**.

![image.png|655x937|225](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/3407c495-c6ad-46cb-9fec-f0973126a472)

Per avere quanti **più dati** possibili a nostra disposizione, e trovare il miglior compromesso, sono stati utilizzati **algoritmi di classificazione diversi** con approcci **unsupervised** o **supervised**.

---

# 6	Metodi - 2

Il secondo algoritmo che andiamo ad analizzare è ROCKAD, questo è un modello creato per il rilevamento delle anomalie, ed è **basato su due parti** fondamentali: nella prima abbiamo l'utilizzo di ROCKET come **estrattore di caratteristiche** e nella seconda viene **addestrato un KNN** o un insieme combinato di essi.

Alla fine di questi passaggi come risultato otteniamo un serie di punteggi di anomalia che utilizziamo per addestrare l'algoritmo **NearestNeighborOCC**, che calcola a sua volta un punteggio di anomalia basato sul **rapporto tra due distanze**.

Il risultato è usato dallo stesso NearestNeighborOCC per effettuare la predizione finale sulla timeseries.

NearestNeighborOCC aggiunge un ulteriore passaggio di analisi per la classificazione, il quale permette di aumentare la **robustezza** e **l’accuratezza** nel rilevamento delle anomalie.

Per utilizzare il dataset NASA è stato necessario utilizzare una tecnica chiamata **overlapping** (come vediamo in figura), che permette di avere più segmenti con gli stessi dati di partenza, **sovrapponendo tra loro le sequenze**.

![image.png|425](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/57b25d57-7f2d-4128-9f70-935db5a08855)

---

# 7	Test OPS_SAT

Entrambi gli algoritmi sono stati testati sul dataset OPS_SAT con varie modalità di esecuzione nel caso di **ROCKET** con più **classificatori diversi** e nel caso di **ROCKAD** invece abbiamo effettuato test con **iperparametri variabili**.

Dalla tabella possiamo osservare che tramite **ROCKET** otteniamo **risultati migliori** rispetto a **ROCKAD** non solo per quanto riguarda le metriche ma in modo ancora più **marcato** per quanto riguarda il **tempo di esecuzione** passando da 7 secondi a circa 2 minuti e 12 secondi.

Il miglior compromesso tra efficacia ed efficienza per quanto riguarda **ROCKET** lo otteniamo con il classificatore **RidgeClassifierCV**, con anche delle metriche molto alte.

Per quanto riguarda **ROCKAD** le migliori metriche sono state ottenute con un numero di **estimatori** pari a $$10$$ e un numero di **kernel** uguale a $$10.000$$, ottenendo dei valori discreti e dei tempi di esecuzione comunque accettabili.

![image.png](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/637864a0-131f-4e7f-aa6a-1f9c6dbc2bc1)

![image.png](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/56decb67-c6f5-4b37-b593-7791ac99de31)

---

# 8	Test NASA

Per i test effettuati con NASA, i risultati differiscono per la scelta del numero di **kernel**, del numero di **n_neighbors** e del valori di **OFFSET**.

Data la costruzione del dataset, suddiviso in **canali separati**, è stato necessario **conservare** e **sommare** i **valori utili** per calcolare le metriche globali.

*Per ROCKET*, possiamo avere 3 possibili soluzioni efficaci, queste portano con sé vantaggi e svantaggi, come in alcune una **velocità di esecuzione molto elevata** ed è un **modello** molto **semplice** e **poco espressivo**.

O in altri casi il **tempo di esecuzione aumenta**, ma di pari passo **aumentano** anche **l'espressività**, la **robustezza** e la **stabilità** del modello.

*Per quanto riguarda ROCKAD*, avremo anche qui due possibili soluzioni con metriche simili, ma rispetto a ROCKET il **tempo di esecuzione cambia in modo ancora più marcato.**

![image.png](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/688ba6b7-596f-41ae-ab2b-9ee7ac2f45aa)

---

# 9	Risultati OPS_SAT

**Contrariamente** a quanto affermato nei paper, ROCKET e ROCKAD **non hanno un efficienza così elevata**.

Infatti nella fase di test, non è stato possibile estrarre le metriche con un valore di STEP **inferiore a 250**, dato che portava ad un esecuzione che avrebbe richiesto **troppo tempo** per terminare.

> Come grafici significativi consideriamo quelli relativi a ROCKAD che consentono di notare meglio i segni distintivi

Il grafico in **arancione** rappresenta la relazione tra i kernel e la metrica F1,  infatti F1 ha un valore costante per tutti i valori dei kernel.  Questo andamento è presente sia per ROCKET che per ROCKAD.

Il grafico **blu** invece indica il tempo di esecuzione in base al numero di kernel, mostrando un **aumento** significativo del **tempo** all'aumentare del numero di kernel.

Anche per ROCKET vale l'andamento crescente visto nel grafico ma con tempi minori.

![GraficoF1_ROCKAD_OPS_SAT.png|500](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/b119d3f6-7304-4732-ad3d-e9de803612d3)

![GraficoTempoEsecuzione_ROCKAD_OPS_SAT.png|500](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/c5f16bb8-f2d1-4c36-9a97-8ee4ac3b9ba5)

---

# 10	Risultati NASA

Dai test effettuati su NASA con un valore di OFFSET **maggiore di** $$50$$**** riscontriamo due problemi: i **tempi** di esecuzione molto **lunghi** e il non poter aumentare il numero di **nodi del KNN**, per mancanza di vicini.

Come possiamo vedere dal grafico arancione la metrica F1 con il variare del numero di kernel **aumenta** leggermente **su valori intermedi** ma resta **uguale** per un numero di kernel pari a **1000** e **10.000**, questo andamento è l'opposto di quello che troviamo in **ROCKET** tranne che per il valore degli estremi che restano uguali.

Nel grafico blu invece sono riportati gli andamenti dei tempi di esecuzione, infatti entrambi i modelli mostrano un **tempo di esecuzione direttamente proporzionale al numero di kernel.**

La differenza maggiore tra i due è il tempo di esecuzione, dove **ROCKET impiega meno della metà del tempo di ROCKAD**, questo infatti cresce in maniera significativa avendo un **valore 7 volte maggiore nel punto finale rispetto a quello iniziale.**

![GraficoF1_ROCKAD.png|500](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/f37ba6ba-db66-4ff8-89ec-66a18f1d58a9)

**F1 di ROCKAD**

![GraficoTempoEsecuzione_ROCKAD.png|500](https://res.craft.do/user/full/e6d22d93-9332-d70a-138e-da405beabb00/doc/7d34d857-cc01-4f98-878a-f9ed30b8abfc/199f5d62-8bc0-43ce-8c69-f22552090f28)

**ROCKAD**

---

# 11	Conclusioni

In conclusione possiamo affermare che **l’efficienza** di ROCKET e ROCKAD è **dipendente** dalla lunghezza della **sotto sequenza**.

In particolare, per il dataset NASA, **l’efficienza dipende** anche dal valore di **OFFSET**, che permette l’overlapping.

Inoltre osserviamo che i valori di **STEP inferiori a 250 sono sconsigliati**, mentre valori di **OFFSET inferiori a 50** portano ad un **aumento significativo del tempo di esecuzione**.

Bisogna anche menzionare il problema relativo alla classificazione riguardante il dataset NASA: infatti con un valore di OFFSET maggiore di quello indicato su alcuni canali potremmo avere una **mancanza di nodi vicini** per il classificatore KNN.

Nonostante i problemi menzionati questi risultati ci fanno **ben sperare** nell’applicazione futura di ROCKET per il rilevamento delle anomalie, soprattutto per il suo utilizzo in maniera supervised, dato il suo **basso costo** computazionale e la **capacità di rilevare** correttamente la maggior parte delle anomalie.

Non possiamo affermare le stesse conclusioni per ROCKAD, che durante i test ha dimostrato varie limitazioni relative al costo computazionale più elevato, che portava ad un **tempo di esecuzione** **elevato** e metriche non molto alte.

ROCKAD rimane comunque consigliato nell’utilizzo con dati non etichettati.

---
## 11.1	ciao
### 11.1.1	cdpkvd
#### 11.1.1.1	sdvd

