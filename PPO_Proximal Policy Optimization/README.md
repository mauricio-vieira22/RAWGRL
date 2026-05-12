# NetROML `__v1__` – Asignación de Recursos WiFi con GNN + PPO (Actor-Critic)

Esta versión mejora la arquitectura de la red agregando una cabeza de **Critic** y reemplaza el algoritmo de entrenamiento de **REINFORCE a Proximal Policy Optimization (PPO)** utilizando un loop de entrenamiento customizado (estilo CleanRL) compatible de manera nativa con las estructuras heterogéneas de PyTorch Geometric.

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

## Novedades en `__v1__` respecto a `__v0__`

- **Algoritmo PPO**: Se implementa Surrogate Loss con Clipping (`clip_coef=0.2`), un MSE Value Loss para el Critic, y un Entropy Bonus para mantener la exploración activa.
- **Ventajas GAE**: Se utiliza *Generalized Advantage Estimation* (`gae_lambda=0.95`) calculando predicciones de Valor $V(s)$.
- **Arquitectura Actor-Critic (`gnn_model.py`)**: El GNN ahora devuelve tres valores `(channel_logits, power_logits, state_value)`, donde `state_value` es una evaluación escalar del estado del entorno implementada mediante un `global_mean_pool` de PyG sobre los features de todos los APs.
- **Rollouts y Epochs**: `train.py` ahora corre asimilando una trayectoria y posteriormente optimizando el modelo en varios minibatchs en lugar de actualizar todo una única vez por episodio.

## Estructura de archivos

```text
__v1__/
├── train.py                    ← ENTRY POINT del entrenamiento (Loop PPO)
├── evaluate.py                 ← Evaluación determinística post-entrenamiento
├── README.md                   ← Documentación
│
├── data/                       ← Preprocesamiento y estructuras
│   └── ... (Igual a __v0__)
│
├── simulation/                 ← Core físico y generador de grafos
│   └── ... (Igual a __v0__)
│
└── model/                      
    ├── network_graph_env.py    ← Entorno Gymnasium de HeteroData
    └── gnn_model.py            ← Arquitectura GNN (Actor-Critic)
```

## Uso rápido

```bash
cd /Users/mauriciovieirarodriguez/project/NetROML/__v1__

# Para correr un entrenamiento PPO base:
python train.py \
    --building_id 814 \
    --episodes 200 \
    --timesteps 100 \
    --lr 3e-4 \
    --update_epochs 4 \
    --clip_coef 0.2 \
    --ent_coef 0.01
```

Otras opciones ajustables desde CLI para afinar PPO:
- `--minibatch_size`: Tamaño del mini-batch de grafos durante las actualizaciones PPO (default: 64)
- `--vf_coef`: Factor de impacto del Value Crtitic Loss (default: 0.5)
- `--max_grad_norm`: Nivel de Gradient Clipping (default: 0.5)

## Salidas y Métricas

| Archivo | Descripción |
|---|---|
| `saved_models/best_model.pt` | Mejor modelo Actor-Critic según el sumatorio del Rate acumulado |
| `saved_models/final_model.pt`| Modelo al finalizar los episodios |
| `training_metrics.csv`       | Detalles paso a paso: Policy Loss, Value Loss, Entropy, GradNorm, Rate y Tiempo |

## Evaluación

Una vez entrenado el modelo, puedes validarlo viendo su comportamiento determinístico:
```bash
python evaluate.py --building_id 814 --model_path ./saved_models/best_model.pt
```
