"""
clases.py – Estructuras de datos del pipeline.

Este módulo define las estructuras base para almacenar la información relacionada
tanto a los eventos en el tiempo (sesiones de clientes) como a los atributos físicos 
de la señal medida a partir de logs reales.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ClientEvent:
    """
    Representa la sesión temporal de conexión (o tráfico) de un cliente WiFi en la simulación.
    
    A través de esta entidad controlamos matemáticamente cuándo un usuario "nace" en el 
    entorno y cuándo "muere", permitiendo simular tráfico dinámico.

    Attributes
    ----------
    distribution_idx : int
        Índice identificador único del cliente temporal al que hace referencia este evento.
    arrival_time : int
        Timestep (época discreta) exacto en el ciclo de simulación donde el cliente aparece.
    departure_time : int
        Timestep exacto donde la sesión de este cliente finaliza y se desconecta.
    duration : int
        Cantidad total de timesteps que el cliente permanece activo. 
        Eq: departure_time - arrival_time.
    """
    distribution_idx: int
    arrival_time:     int
    departure_time:   int
    duration:         int


@dataclass
class Block:
    """
    Bloque estático de mediciones físicas (ganancias/RSSI) de un cliente específico.
    
    Un bloque contiene un DataFrame que detalla la "visión" que tiene este cliente 
    hacia TODOS los Access Points (APs) en un timestamp discreto. Sirve para almacenar
    la atenuación espacial estática sin consumir excesiva memoria.

    Attributes
    ----------
    block_idx : int
        ID único de este bloque de datos.
    distribution_idx : int
        ID del cliente al que lógicamente pertenece esta medición.
    datos : pd.DataFrame
        Matriz de datos asociando a cada AP las ganancias `G_2_4` (Ganancia a 2.4 GHz) 
        y `G_5` (Ganancia a 5 GHz) calculadas en decibelios (dB). Las ganancias 
        se miden restando la potencia de transmisión (P_tx) del RSSI observado.
        Aquellos APs que no fueron captados tendran una ganancia nula ponderada con -infinito.
    """
    block_idx:        int
    distribution_idx: int
    datos:            pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return f"Block(block_idx={self.block_idx}, client_idx={self.distribution_idx})"


@dataclass
class Distribution:
    """
    Historial estadístico y posicional de un cliente Wifi particular en el edificio.
    
    Agrupa secuencias de la clase `Block` para un solo dispositivo físico (MAC Address).
    Representa el trackeo real de las mediciones de campo que experimentó el dispositivo, 
    permitiendo a la simulación tomar muestras (sampling) realistas de los canales.

    Attributes
    ----------
    distribution_idx : int
        ID asignado numéricamente a este cliente durante el proceso global.
    mac_client : str
        Dirección de control de acceso al medio (MAC Address) ocultada por privacidad 
        del dispositivo físico original.
    blocks : np.ndarray
        Vector en memoria de objetos de tipo `Block` que mantienen el paso a paso
        asociado a la medición realizada por este cliente. Cuanto más camina, más bloques.
    """
    distribution_idx: int
    mac_client:       str
    blocks:           np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __repr__(self) -> str:
        return f"Distribution(idx={self.distribution_idx}, blocks_total={len(self.blocks)})"
