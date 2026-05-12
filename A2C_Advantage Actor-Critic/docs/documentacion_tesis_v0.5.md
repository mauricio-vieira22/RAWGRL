# NetROML Versión 0.5 — Documentación Técnica y Arquitectural a Nivel Tesis

Este documento provee una especificación formal y exhaustiva de las actualizaciones arquitecturales, correcciones semánticas y formulaciones matemáticas introducidas en la versión `__v0.5__` del simulador de asignación de recursos WiFi (NetROML). Esta versión actúa como un puente fundamental crítico entre la prueba de concepto preliminar de REINFORCE (`__v0__`) y la representación interdependiente final planificada (`__v1__`).

---

## 1. Motivación y Antecedentes

La versión `__v0__` demostró la viabilidad de utilizar Redes Neuronales de Grafos Heterogéneas (HGNN) en conjunto con un optimizador por gradiente de políticas (Policy Gradient) para solucionar la ineficiencia espectral de la especificación 802.11. Sin embargo, el análisis retrospectivo del algoritmo develó tres vectores de mejora inexorables para garantizar estabilidad durante la convergencia prolongada:
1. Alta varianza en los gradientes producto del uso de REINFORCE puro con un baseline global retrospectivo.
2. Sensibilidad ante hiperparámetros estáticos no generalizables, en particular en la normalización de la Ecuación de Capacidad de Shannon-Hartley.
3. Inconsistencias semánticas originadas en el pipeline de preprocesamiento de características de canal al representar "la ausencia de línea de visión física" (desconexión o atenuación absoluta) con valores sentinelas incompatibles (`-inf` y `NaN` interactuando).

---

## 2. Refactorización del Pipeline de Canal Físico

### 2.1. Problema de la Representación Semántica Categórica
En telecomunicaciones, cuando un Access Point (AP) se encuentra térmicamente fuera de cobertura para un Cliente Móvil, dicho enlace físico no posee un valor subyacente. En `__v0__`, el pipeline de precesamiento inyectaba `-inf` (producto de la sustracción del Nivel de Ruido o RSSI Ficticio contra la potencia `P_TX`), lo cual representaba formalmente una atenuación absoluta logarítmica ($- \infty$ dBm). 
Sin embargo, al migrar al dominio de optimización tensorial (PyTorch), operaciones de agrupación, max-pooling o funciones log-lineales corrían el riesgo matemático de inducir `NaN` (Not a Number) de segundo grado, corrompiendo silenciosamente el grafo de cómputo del Optimizador.

### 2.2. Solución y Convención Unificada
A partir de `__v0.5__`, se establece normativamente a `NaN` como representador global de "Carencia Fiel de Señal".
* En el paso 2 del pipeline computacional (`step2_gain_e_imputer.py`), se eliminó `fill_value=-np.inf` sustituyéndolo sistémicamente por `np.nan`.
* En el encapsulamiento de Bloques Discretos de Pandas (`step3_mapear_a_blocks.py`), las operaciones `.max()` se condicionaron estrictamente a `min_count=1`. Esto obedece a las variaciones arquitectónicas subyacentes en `pandas`; aseguramos que el máximo de un segmento completamente vacío derive normativamente en `NaN` en vez de interpolarse asumiendo un piso flotante nulo (`0.0`).
* **Soporte Retrospectivo Estricto:** Dado el excesivo costo de CPU en re-compilar bloques `.joblib` de `__v0__`, las lecturas heredadas interponen un co-procesamiento "On-The-Fly" donde todo valor Negativo Infinito detectado pre-compilación tensorial es casteado a la nueva representación ($NaN$).

$$ Tensor_{_{filtrado}} \leftarrow \text{nan\_to\_num}(Tensor_{_{_{crudo}}}, \, \text{neginf} \leftarrow \text{NaN}) $$

---

## 3. Limitación Numérica y Normalización Topológica Adaptativa

En las bases fundacionales (`__v0__`), la red asimiló las percepciones de anchos de banda esperados $C$ basándose explícitamente en el Teorema de Shannon-Hartley. 

$$ C = \log_2(1 + \text{SINR}_{_{\text{lineal}}}) \quad [\text{Bits/s/Hz}] $$

Sin embargo, para garantizar una convergencia estable, las Redes Neuronales de Grafos exigen de entradas (Features) normalizadas en intervalos estandarizados (idealmente $[0, 1]$ o $\mathcal{N}(0, 1)$). La versión iterativa normalizaba la entrada dividiendo rígidamente sobre el factor $10.0$, asumiendo el ideal hipotético de un $\text{SINR}$ ininterrumpido de $30 \text{dB}$.

En `__v0.5__`, reemplazamos esta heurística ingenieril dependiente del entorno ("Magic Number") por un diseño escalable y agnóstico mediante vectorización **Min-Max Adaptativa temporalizada**.

$$ x_{_{\text{norm}}}^{t} = \frac{x^{t}}{\max(x^{t}, \, \epsilon)} \quad \text{donde } \epsilon = 1 \!\times\! 10^{-6}$$

Este sistema evalúa la recompensa y el *rate media* dinámicamente per-iteración en el batch de usuarios interconectados mediante `torch.clamp`, eliminando drásticamente el sesgo de variabilidad. Esto prevé divisiones por cero (0) en instantes fenológicos ( $t = 0$ ) cuando las conexiones no están establecidas, garantizando rigor formal y generalidad si los entornos varían radicalmente la métrica del piso de ruido en el futuro.

---

## 4. Evolución de Política Optimizada: Avance al algoritmo Advantage Actor-Critic (A2C)

La reforma principal de esta variante es la sustitución del núcleo de inteligencia matemática de Decisión de Markov (MDP). `__v0__` fundamentó su optimizador de REINFORCE en el uso del simple Teodrama Fundacional del Gradiente de Política donde la política $\pi_{\theta}$ se actualizaba ponderando sobre un multiplicador de Retorno Global Discounted $G_t$ con un baseline promedio móvil lento. El principal dolor a nivel de tesis o experimentación científica es que dicho componente posee altísima varianza endógena.

**El Framework Advantage Actor-Critic Asincrónico (A2C) define una arquitectura bifurcada en la cual la GAT (Graph Attention Network) expide dos outputs segregados combinando valor y decisiones:**

### 4.1. La Cabeza del Actor (Actor Head)
Corresponde a la capa conectada de la subred del Agente con parámetros de aprendizaje $\theta$. Es responsable directo de deducir, decodificar, y despachar los logits vectoriales $\pi_{\theta}(\text{ Canal } | s_t)$ y $\pi_{\theta}(\text{ Potencia } | s_t)$. Representa las acciones directas de control de gestión espectral AP independientes entre sí.

### 4.2. La Cabeza del Crítico (Critic Head)
Integra un estrato neuronal adicional parametrizado convencionalmente por vector de fase $\phi$ que persigue aproximarse ciegamente a la función de valor teórica del grafo en cuestión $V^{\pi}(s)$.
Considerando un estado $s_t$ como un cúmulo jerárquico de Access Points $H \in \mathbb{R}^{n \times d}$, el nodo Actor de valor Crítico consolida mediante un Average Global Graph-Pooling todos los *Embeddings* terminales.

$$ s_{_{\text{global\_emb}}} =  \text{MeanPool}(\{h_1, h_2, ..., h_n\}) $$
$$ \hat{V}_{\phi}(s_t) = W_3 \cdot \max(0, W_2 \cdot s_{_{\text{global\_emb}}} + b_2) + b_3 $$

Esto permite estipular una métrica que predice cuánta Eficiencia Espectral (Reward) el entorno va a exprimir en total independientemente a las decisiones individuales subsiguientes. 

### 4.3. Restructuración Ecuacional e Integración A2C
El "Actor" se entrena mediante el uso normativizado del vector paramétrico de *Advantage* (Ventaja) evaluada analíticamente como el error temporal transaccional de lo estimado empíricamente contra lo presagiado deductivamente.

$$ A_t = G_t - \hat{V}_\phi(s_t)$$

El vector ventajoso es derivado luego por normalización iterativa:
$$ \tilde{A}_t = \frac{A_t - \mu(A_t)}{\sigma(A_t) + \epsilon} $$

La pérdida holística multicriterio se formula formalmente definiendo al Error Cuadrado Medio (MSE) como regulador del Crítico y una sub-política sumatoria regularizadora de Bonificación de Entropía ($\mathcal{H}$) para exacerbar empíricamente la exploración algorítmica temprana al impedir divergencias hacia máximos locales sub-óptimos:

$$ \mathcal{L}_{_{actor}} = -\frac{1}{T} \sum_{t=0}^{T} \tilde{A}_t \log \pi_\theta(a_t|s_t) $$
$$ \mathcal{L}_{_{critic}} = \frac{1}{T} \sum_{t=0}^{T} \left( G_t - \hat{V}_\phi(s_t) \right)^2 $$
$$ \mathcal{L}_{_{total}} = \mathcal{L}_{_{actor}} + c_{_{v}} \mathcal{L}_{_{critic}} - c_{_{e}} \mathcal{H}(\pi_\theta) $$

*(donde $c_v$ rige el coeficiente factor crítico (0.5 por convención técnica) y $c_e$ penaliza decaimiento entrópico general [0.01]).*

---

## 5. Abordaje Pragmático de Mantenibilidad Computacional

1. **Isolation del Inspector de Typings (PyG Patch)**: La versión originaria anclaba un Hotfix rudimentario al inicio de `train.py` para evadir el `AttributeError: '_name'` inducida por la incapacidad de introspección nativa del intérprete Python 3.13 sobre librerías del inspector PyTorch Geometric. 
Implementamos un marco de mitigación inyectándolo globalmente a través del módulo subyacente `utils/compat.py`, robusteciendo el flujo productivo lógico y aislando los fallos originados por incompatibilidades dependientes agenas al núcleo físico programático.
2. **Desacoplamiento Estocástico Selectivo:** Hemos desacoplado el loop inferencial introduciendo el testeo evaluativo sin intervención iterativa (`torch.no_grad()`) en semillas seud-aleatorias rigurosas distantes del entrenamiento `(+9999)`.
Esto le proporciona a las arquitecturas la posibilidad de auditar y garantizar si la GNN ha consolidado patrones de control sistémicos o sí solo está superponiendo sobre-ajustamientos (*overfitting*) a las particularidades paramétricas iterativas. El log sistemático de PyTorch ahora salvaguardará en el disco única y exclusivamente las iteraciones maestras estipuladas por su máximo empírico cruzado ($G_{Val}$) y no por su retorno ruidoso nominal de entrenamiento.
