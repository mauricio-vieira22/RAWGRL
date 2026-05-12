# NetROML `__v0__` – Asignación de Recursos WiFi con GNN + REINFORCE

Versión inicial autocontenida del pipeline de entrenamiento.

## Fundamento Teórico

### Definición de Variables

Consideramos tiempo ranurado (slotted), indexado por $t \in \mathbb{N}$. Supongamos que hay $N$ APs indexados por $n \in \{1, \ldots, N\}$. El índice $u \in \mathbb{N}$ indexará a los clientes a medida que vayan llegando al sistema. El tiempo de arribo de cada cliente será $t_u$ y cada uno tendrá una duración $\Delta_u$.

Durante su conexión y por tanto en tiempo $t \in \{t_u, t_u+1, \ldots, t_u+\Delta_u\}$, el cliente $u$ tendrá ganancias desde los APs dadas por el vector $h_{u,t} \in \mathbb{R}^N$. La totalidad de las ganancias se puede representar como una matriz:

$$H_t \in \mathbb{R}^{U_t \times N},$$

siendo $U_t$ el total de clientes existentes en tiempo $t$. Es importante notar que la fila $v$ corresponde a las ganancias de cierto cliente $v$, que no guarda relación con el índice $u$: este último indexa todos los clientes que han existido en el sistema, mientras que $v$ identifica al cliente del total conectados en el momento $t$.

Más allá de lo anterior, se puede interpretar lo anterior como un grafo bipartito con pesos $H_t$ entre clientes y APs, cuya matriz de adyacencia sería:

$$\begin{pmatrix} 0 & H_t \\ H_t^\top & 0 \end{pmatrix}.$$

A su vez, el $n$-ésimo AP tiene configurado en tiempo $t$ un canal $c_{n,t} \in \{1, \ldots, C\}$ y una potencia $P_{n,t} \in \{P_1, \ldots, P_L\}$ (asumiendo un conjunto discreto de potencias posibles). Retomando el grafo que definimos recién, estas dos variables las podríamos interpretar como una señal sobre el grafo:

$$X_t \in \mathbb{R}^{(U_t+N) \times 2}.$$

En particular, sobre los nodos correspondientes a los APs tenemos $c_{n,t}$ y $P_{n,t}$ y en el resto por ahora rellenamos con $0$.

Discutamos ahora cómo es el modelo de conexión entre los APs y los clientes. En tiempo $t$, el cliente $u$ está conectado al AP $a_{u,t}$. En este caso podemos pensar para cada $t$ en otro grafo bipartito $A_t$ entre clientes y APs, pero esta vez binario y tal que el grado de los nodos correspondientes a clientes es siempre $1$. Su matriz de adyacencia estará dada por:

$$\begin{pmatrix} 0 & A_t \\ A_t^\top & 0 \end{pmatrix}$$

con

$$A_t \in \{0,1\}^{U_t \times N}.$$

### Dinámica

La evolución de la fila $u$ de $A_t$ depende de cuál escenario de sticky client consideremos:

- **Full sticky**: el cliente busca el mejor AP al momento de conectarse y nunca cambia de AP luego de eso. En este caso $a_{u,t_u}$ depende de $X_{t_u}$ y $H_{t_u}$ (al menos de las filas correspondientes a sus ganancias), lo cual será cierto en todos los escenarios. Después de su nacimiento y hasta su finalización, se cumple que $a_{u,t+1} \leftarrow a_{u,t}$.

- **Sticky**: sólo en caso que la potencia desde el AP asignado baje de cierto umbral, el cliente elige nuevamente el mejor AP. En este caso, y para los tiempos posteriores a su nacimiento, $a_{u,t}$ depende de $X_t$ y $H_t$ (para verificar que la potencia desde su AP no haya caído por debajo del umbral, y en caso contrario elegir al mejor nuevo AP) y de $a_{u,t-1}$ (en ese caso se repite la asignación, al igual que el último escenario).

- **Sticky lite**: en caso que la potencia desde otro AP al asignado supere la potencia del AP asignado, el cliente se cambia a este nuevo AP. Este caso es muy similar al anterior, salvo que la decisión de cambio del cliente depende de todas las ganancias y no únicamente de la que tiene con su AP. Por tanto, las variables de las que dependen son las mismas.

Una vez establecida la conexión vamos a suponer que la ganancia es independiente paso a paso. Esto no es del todo cierto, pero por construcción lo vamos a hacer de esta forma. Esto implica que la matriz $H_t$ es independiente de $H_{t-1}$ (al menos en las entradas correspondientes a clientes que se mantienen entre $t-1$ y $t$).

De todas formas, el mayor problema es con la evolución en el número de clientes, y por tanto en el tamaño de las matrices involucradas. Salvo que la duración de cada cliente sea exponencial y el proceso de arribo Poisson, no podemos suponer que la distribución en $t$ del tamaño de las matrices depende únicamente de cuántos clientes había en $t-1$. Dicho de otra forma, en tiempo $t+1$ no podemos sortear ni saber cuántos y cuáles clientes van a terminar únicamente con $H_t$, $X_t$ y $A_t$. En particular, deberíamos guardar también cuánto tiempo le resta de conexión a cada cliente conectado actualmente. Es similar para los arribos, en particular el momento en que llegan nuevos clientes, pues no podemos sortear de forma independiente para cada slot si habrá un nuevo arribo (salvo, nuevamente, si los arribos son del tipo Poisson).

Un modelo mínimamente más realista sería sortear tras cada nuevo arribo un valor para el tiempo hasta el siguiente arribo (y ya dijimos que las ganancias correspondientes a este nuevo cliente sí serán independientes de lo que ya existe y los pasos anteriores). Esto último implica también guardar una variable temporal que nos indique cuánto falta para el próximo arribo, así como las ganancias y duración del cliente correspondiente. En caso que se habilite la llegada de varios clientes por slot, también habría que agregar un sorteo para la cantidad de clientes que llegarían.

Por tanto, el proceso será realmente Markoviano si al proceso $(H_t, X_t, A_t)$ le agregamos $\epsilon_{t,u}$ que representa cuántos slots le restan a cada usuario activo $u$ en tiempo $t$. Esta variable se puede considerar como una señal sobre el grafo $A_t$:

$$\epsilon_t \in \mathbb{N}^{(U_t+N) \times 1},$$

donde en los nodos correspondientes a los APs simplemente rellenaremos con cero, similar a lo que hicimos antes con $X_t$. Finalmente, será necesaria una última variable $\delta_t$ que indique cuánto resta para la llegada del/los siguiente/s cliente/s, que se descontará en cada slot y se volverá a sortear cuando llegue a cero.

En suma, el proceso queda definido como Markoviano si consideramos las cuatro variables $(H_t, X_t, A_t, \epsilon_t, \delta_t)$. De todas formas, si bien algún tipo de consideración como la anterior será necesaria para las simulaciones, estas últimas dos variables típicamente no se conocen en tiempo $t$. Esto significa que nuestro proceso de Markov será parcialmente observable.

### Reinforcement Learning

Por slot, el desempeño del sistema estará dado por el rate total, el cual se puede obtener como una función de las variables ya mencionadas:

$$r_t = F(H_t, X_t, A_t) = \sum_{v=1}^{U_t} r_v(H_t, X_t, A_t),$$

donde la función $r_v(\cdot)$ corresponde al rate obtenido por el cliente $v$ dada la situación actual del sistema, caracterizado por $(H_t, X_t, A_t)$. Nuestra variable de acción será precisamente $X_t$, que incluye la potencia y el canal de cada AP.

Es importante notar aquí que los ajustes en $X_t$ se podrán hacer cada cierta cantidad $T$ de slots. Por lo tanto, el objetivo de nuestras decisiones será:

$$\underset{\{X_0,\, X_T,\, \ldots,\, X_{T(\Gamma-1)}\}}{\arg\max} \sum_{\tau=0}^{\Gamma-1} \sum_{s=0}^{T-1} F\!\left(H_{\tau T+s},\, X_{\tau T},\, A_{\tau T+s}\right).$$

Por tanto, el reward que obtendremos en cada paso de nuestra decisión será:

$$R_\tau = \sum_{s=0}^{T-1} F\!\left(H_{\tau T+s},\, X_{\tau T},\, A_{\tau T+s}\right),$$

y aplicando algún algoritmo de reinforcement learning (RL) deberíamos llegar a una política $\pi(a \mid s)$ que resuelva:

$$\max_\pi\; \mathbb{E}\!\left[\sum_{\tau=0}^{\infty} \gamma^\tau R_\tau\right].$$

Hay una diferencia importante respecto al caso de RL estándar y la definición del MDP. En las secciones anteriores nos convencimos de que el sistema es un MDP en $t$. Para que la teoría siga funcionando, deberíamos estar convencidos de que el sistema es un MDP en $\tau$ (es decir, cada tiempo $t = \tau T$). En este caso es cierto pues la política queda fija en los pasos intermedios, y el sistema evoluciona como describimos antes y por tanto los cambios en los $T$ pasos intermedios siguen siendo markovianos.

De todas formas, el sistema es parcialmente observable pues no conocemos ni $\epsilon_{\tau T}$ ni en qué slots llegarán los nuevos clientes (i.e.\ las distintas realizaciones de $\delta_t$). Se podría de todos modos experimentar con cuánto pierde el algoritmo de RL por no poder observar estas variables.

## Estructura de archivos

```text
__v0__/
├── train.py                    ← ENTRY POINT del entrenamiento
├── README.md                   ← Documentación
│
├── data/                       ← Preprocesamiento y estructuras
│   ├── clases.py               ← Dataclasses: ClientEvent, Block, Distribution
│   ├── preprocessing.py        ← Pipeline end-to-end de preprocessing
│   ├── step1_crear_dataset.py
│   ├── step2_gain_e_imputer.py
│   ├── step3_mapear_a_blocks.py
│   └── step4_block_to_distribution.py
│
├── simulation/                 ← Core físico y generador de grafos
│   ├── arrival_departure_model.py ← Modelo Poisson/Exponencial
│   ├── wifi_physics.py         ← Física WiFi (grilla, RSSI, SINR, rate)
│   └── graph_builder.py        ← Constructor de HeteroData por timestep
│
└── model/                      ← Componentes de RL y GNN
    ├── network_graph_env.py    ← Entorno Gymnasium
    └── gnn_model.py            ← Arquitectura GNN
```

## Uso rápido

```bash
cd /Users/mauriciovieirarodriguez/project/NetROML/__v0__

# Usando el CSV de Step2 ya generado (recomendado – mucho más rápido)
python train.py \
    --step2_csv ../preprocesamiento_de_datos/output/dataset_990_step2.csv \
    --episodes 200 \
    --timesteps 100

# Con todos los parámetros
python train.py \
    --building_id 990 \
    --step2_csv ../preprocesamiento_de_datos/output/dataset_990_step2.csv \
    --episodes 500 \
    --timesteps 200 \
    --arrival_rate 2.0 \
    --mean_dur 10.0 \
    --hidden 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --seed 314 \
    --save_dir saved_models
```

## Salidas

| Archivo | Descripción |
|---|---|
| `saved_models/best_model.pt` | Mejor modelo según G acumulado |
| `saved_models/final_model.pt` | Modelo al final del entrenamiento |
| `saved_models/model_epN.pt` | Checkpoint cada 50 episodios |
| `training_metrics.csv` | G, rate, loss, grad_norm, lr, tiempo por episodio |

## Flujo interno

```
preprocessing.run_pipeline()
        │
        ▼  ArrivalDepartureModel.simulate_all_events()  [Poisson + Exp]
list[ClientEvent]
        │
        ▼  crear_grilla() → convertir_grilla_a_tensor()
(n_ev, T, n_APs, 2) ganancias [dB]
        │
        ▼  obtener_grilla_RSSIs(potencias_APs)
(n_ev, T, n_APs, 2) RSSI [dBm]
        │
        ▼  asignaciones_AP()   [sticky client, prioridad 5GHz]
(n_ev, T, 3) [ap_idx, band_idx, RSSI]
        │
        ▼  calcular_sinr()  [interferencia co-canal y co-banda]
(n_ev, T) lineal
        │
        ▼  calcular_rate() + actualizar_avg_rate()
Reward = Σ rate activos en el timestep
        │
        ▼  construir_grafo_timestep()
HeteroData → GNN → [ch_logits, pwr_logits] → Categorical → acción
```

## Notas técnicas

- **Interferencia correcta**: sólo entre APs que comparten canal **y** la misma banda de recepción del cliente.
- **Sticky client**: un cliente mantiene su AP/banda actual a menos que la señal baje del `umbral_conexion` (−85 dBm por defecto).
- **5 GHz priorizado**: si RSSI 5G ≥ −70 dBm se prefiere esa banda.
- **Promedio running** del rate para feature de nodo Cliente: `avg_n = ((n−1)·avg_{n−1} + r_nuevo) / n`.
