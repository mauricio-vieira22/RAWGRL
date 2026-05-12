"""
clases.py – Entidades Estructurales y Tipos de Datos de NetROML.

Este módulo define las estructuras de datos fundamentales (Dataclasses) que orquestan 
el flujo de información entre el pipeline de preprocesamiento, el simulador físico 
y los algoritmos de aprendizaje por refuerzo.

Jerarquía de Datos:
    - ClientEvent: Abstracción de la sesión temporal (ciclo de vida) de un agente.
    - Block: Instantánea espacial de las propiedades físicas del canal (Ganancia/RSSI).
    - Distribution: Trayectoria completa de un cliente basada en mediciones reales.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ClientEvent:
    """
    Representa el ciclo de vida de una sesión cliente en el entorno de simulación.
    
    Esta entidad define los límites temporales (épocas de llegada y salida) que rigen 
    la presencia de un agente en el grafo de red, permitiendo modelar el tráfico 
    dinámico y la volatilidad del sistema.

    Attributes
    ----------
    distribution_idx : int
        Identificador único de la trayectoria (Distribution) asociada a este evento.
    arrival_time : int
        Timestep (época discreta) en el cual el cliente inicia su actividad en la red.
    departure_time : int
        Timestep en el cual el cliente finaliza su sesión y se desconecta.
    duration : int
        Extensión total de la sesión expresada en unidades de tiempo discreto.
        Formalmente: departure_time - arrival_time.
    """
    distribution_idx: int
    arrival_time:     int
    departure_time:   int
    duration:         int


@dataclass
class Block:
    """
    Instantánea de estado de las propiedades físicas del canal para un cliente.
    
    El 'Block' encapsula la "visión" espectral de un cliente hacia la infraestructura 
    del edificio en un instante dado. Almacena las ganancias de propagación medidas 
    hacia todos los Access Points definidos en el entorno.

    Attributes
    ----------
    block_idx : int
        Identificador secuencial del bloque dentro de una trayectoria.
    distribution_idx : int
        Identificador de la trayectoria (Distribution) a la que pertenece este bloque.
    datos : pd.DataFrame
        Matriz de estado que contiene la relación [mac_ap, G_2_4, G_5].
        Las ganancias (Gain) se expresan en decibelios (dB) y representan la 
        atenuación del canal (RSSI - P_TX). Los APs no visibles conservan valores NaN.
    """
    block_idx:        int
    distribution_idx: int
    datos:            pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return f"Block(id={self.block_idx}, client_id={self.distribution_idx}, ap_count={len(self.datos)})"


@dataclass
class Distribution:
    """
    Trayectoria histórica y caracterización física de un cliente WiFi.
    
    Esta clase es la unidad fundamental de datos del proyecto. Agrupa la secuencia 
    de mediciones reales (objetos Block) realizadas por un dispositivo físico 
    (MAC Address) a lo largo del tiempo, permitiendo un muestreo (sampling) 
    realista del entorno radioeléctrico.

    Attributes
    ----------
    distribution_idx : int
        Identificador numérico único de la distribución.
    mac_client : str
        Dirección MAC ofuscada del dispositivo cliente original.
    blocks : np.ndarray
        Vector de alta eficiencia conteniendo objetos 'Block' que describen 
        la evolución espacial de la señal para este cliente.
    """
    distribution_idx: int
    mac_client:       str
    blocks:           np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"Distribution(idx={self.distribution_idx}, mac={self.mac_client}, t_len={len(self.blocks)})"
