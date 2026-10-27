"""Enriquecimiento del dataset de pizzas con un LLM.

El LLM genera el conocimiento del negocio (sucursales con coordenadas, barrios, vías,
nombres, festivos y plan de demanda) y Python construye las filas re-muestreando pedidos
reales, para que precios, cantidades y totales siempre cuadren.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocesamiento import PreprocesadorPizzas


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta sobre la Tierra (km). Acepta escalares o arreglos."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------- configuración por defecto
CIUDADES_DEFECTO = [  # (código, ciudad, país) — la primera es la sede principal, dueña de los datos reales
    ("MED", "Medellín", "Colombia"),
    ("BOG", "Bogotá", "Colombia"),
    ("CAL", "Cali", "Colombia"),
    ("UIO", "Quito", "Ecuador"),
    ("PTY", "Ciudad de Panamá", "Panamá"),
]
# Centros de referencia para validar las coordenadas que devuelva el LLM
CENTROS_DEFECTO = {"MED": (6.2442, -75.5812), "BOG": (4.7110, -74.0721), "CAL": (3.4516, -76.5320),
                   "UIO": (-0.1807, -78.4678), "PTY": (8.9824, -79.5199)}

RESPALDO_SUCURSALES = {
    "factores_dia_semana": {"lunes": 0.85, "martes": 0.90, "miércoles": 0.95, "jueves": 1.05,
                            "viernes": 1.20, "sábado": 1.25, "domingo": 0.80},
    "sucursales": [
        {"codigo": "MED", "nombre_sucursal": "Pizzería El Poblado", "direccion": "Carrera 43A # 7-50",
         "barrio": "El Poblado", "latitud": 6.2090, "longitud": -75.5680, "factor_volumen": 1.00,
         "preferencias_categoria": {"Classic": 1.0, "Supreme": 1.0, "Veggie": 1.0, "Chicken": 1.0},
         "dias_especiales": [{"fecha": "2015-01-01", "factor": 0.8, "motivo": "Año Nuevo"},
                             {"fecha": "2015-01-12", "factor": 1.15, "motivo": "Festivo de Reyes"}]},
        {"codigo": "BOG", "nombre_sucursal": "Pizzería Chapinero", "direccion": "Carrera 13 # 63-20",
         "barrio": "Chapinero", "latitud": 4.6485, "longitud": -74.0630, "factor_volumen": 1.25,
         "preferencias_categoria": {"Classic": 1.1, "Supreme": 1.05, "Veggie": 0.95, "Chicken": 0.9},
         "dias_especiales": [{"fecha": "2015-01-01", "factor": 0.75, "motivo": "Año Nuevo"},
                             {"fecha": "2015-01-12", "factor": 1.15, "motivo": "Festivo de Reyes"}]},
        {"codigo": "CAL", "nombre_sucursal": "Pizzería Granada", "direccion": "Avenida 9N # 15-30",
         "barrio": "Granada", "latitud": 3.4590, "longitud": -76.5330, "factor_volumen": 0.85,
         "preferencias_categoria": {"Classic": 0.95, "Supreme": 1.0, "Veggie": 0.9, "Chicken": 1.2},
         "dias_especiales": [{"fecha": "2015-01-01", "factor": 0.8, "motivo": "Año Nuevo"},
                             {"fecha": "2015-01-12", "factor": 1.1, "motivo": "Festivo de Reyes"}]},
        {"codigo": "UIO", "nombre_sucursal": "Pizzería La Carolina", "direccion": "Av. Amazonas N36-150 y Corea",
         "barrio": "La Carolina", "latitud": -0.1810, "longitud": -78.4850, "factor_volumen": 0.90,
         "preferencias_categoria": {"Classic": 1.0, "Supreme": 0.9, "Veggie": 1.2, "Chicken": 1.0},
         "dias_especiales": [{"fecha": "2015-01-01", "factor": 0.8, "motivo": "Año Nuevo"}]},
        {"codigo": "PTY", "nombre_sucursal": "Pizzería Obarrio", "direccion": "Calle 50 y Calle 53 Este",
         "barrio": "Obarrio", "latitud": 8.9850, "longitud": -79.5190, "factor_volumen": 0.75,
         "preferencias_categoria": {"Classic": 0.9, "Supreme": 1.15, "Veggie": 0.85, "Chicken": 1.1},
         "dias_especiales": [{"fecha": "2015-01-01", "factor": 0.8, "motivo": "Año Nuevo"},
                             {"fecha": "2015-01-09", "factor": 0.9, "motivo": "Día de los Mártires"}]},
    ],
}

_NOMBRES = ["Juan", "María", "Carlos", "Ana", "Luis", "Laura", "Andrés", "Sofía", "Jorge", "Valentina",
            "Diego", "Camila", "Felipe", "Daniela", "Santiago", "Paula", "Alejandro", "Natalia", "Sebastián",
            "Isabella", "Mateo", "Mariana", "Julián", "Gabriela", "David", "Catalina", "Esteban", "Juliana",
            "Ricardo", "Carolina"]
_APELLIDOS = ["García", "Rodríguez", "Martínez", "López", "González", "Pérez", "Sánchez", "Ramírez", "Torres",
              "Flores", "Rivera", "Gómez", "Díaz", "Restrepo", "Vargas", "Castro", "Ortiz", "Morales", "Jiménez",
              "Herrera", "Medina", "Rojas", "Muñoz", "Álvarez", "Romero", "Mendoza", "Silva", "Cárdenas",
              "Mejía", "Ospina"]


def _b(barrio, lat, lon, vias):
    return {"barrio": barrio, "latitud": lat, "longitud": lon, "vias": vias}


RESPALDO_BARRIOS = {
    "MED": [_b("El Poblado", 6.2088, -75.5675, ["Carrera 43A", "Calle 10", "Carrera 37"]),
            _b("Laureles", 6.2450, -75.5960, ["Carrera 76", "Circular 1", "Calle 39"]),
            _b("Envigado", 6.1710, -75.5860, ["Carrera 43", "Calle 38 Sur", "Carrera 27"]),
            _b("Belén", 6.2310, -75.6010, ["Calle 30", "Carrera 76", "Calle 32"]),
            _b("Buenos Aires", 6.2380, -75.5560, ["Calle 49", "Carrera 30", "Calle 45"]),
            _b("La América", 6.2560, -75.6050, ["Calle 44", "Carrera 84", "Calle 48"]),
            _b("Castilla", 6.2950, -75.5700, ["Carrera 68", "Calle 98", "Carrera 65"]),
            _b("Guayabal", 6.2150, -75.5850, ["Carrera 52", "Calle 10 Sur", "Carrera 50"])],
    "BOG": [_b("Chapinero", 4.6486, -74.0628, ["Carrera 13", "Calle 63", "Carrera 7"]),
            _b("Usaquén", 4.6950, -74.0310, ["Carrera 7", "Calle 119", "Carrera 15"]),
            _b("Teusaquillo", 4.6360, -74.0775, ["Calle 39", "Carrera 19", "Calle 45"]),
            _b("Chicó", 4.6760, -74.0480, ["Calle 93", "Carrera 11", "Calle 97"]),
            _b("Galerías", 4.6420, -74.0750, ["Calle 53", "Carrera 24", "Calle 51"]),
            _b("Cedritos", 4.7250, -74.0400, ["Calle 140", "Carrera 9", "Calle 147"]),
            _b("La Candelaria", 4.5970, -74.0730, ["Calle 11", "Carrera 4", "Calle 13"]),
            _b("Rosales", 4.6560, -74.0520, ["Calle 72", "Carrera 4", "Calle 70"])],
    "CAL": [_b("Granada", 3.4590, -76.5330, ["Avenida 9N", "Calle 15N", "Avenida 8N"]),
            _b("San Fernando", 3.4370, -76.5430, ["Carrera 34", "Calle 5", "Carrera 31"]),
            _b("Ciudad Jardín", 3.3600, -76.5300, ["Carrera 105", "Calle 16", "Carrera 100"]),
            _b("El Peñón", 3.4510, -76.5430, ["Carrera 3 Oeste", "Calle 2 Oeste", "Carrera 4"]),
            _b("Santa Mónica", 3.4700, -76.5260, ["Avenida 5N", "Calle 23N", "Avenida 4N"]),
            _b("Tequendama", 3.4280, -76.5420, ["Calle 5", "Carrera 39", "Calle 6"]),
            _b("El Ingenio", 3.3850, -76.5280, ["Carrera 85", "Calle 13", "Carrera 83"]),
            _b("Versalles", 3.4620, -76.5300, ["Avenida 6N", "Calle 21N", "Avenida 6AN"])],
    "UIO": [_b("La Carolina", -0.1820, -78.4840, ["Av. Amazonas", "Av. República", "Av. Shyris"]),
            _b("La Mariscal", -0.2030, -78.4900, ["Av. 6 de Diciembre", "Juan León Mera", "Reina Victoria"]),
            _b("Iñaquito", -0.1760, -78.4800, ["Av. Naciones Unidas", "Av. 10 de Agosto", "Av. Atahualpa"]),
            _b("Bellavista", -0.1880, -78.4730, ["Av. Eloy Alfaro", "Calle Bossano", "Calle Toledo"]),
            _b("El Batán", -0.1700, -78.4700, ["Av. Gaspar de Villarroel", "Av. 6 de Diciembre", "Calle Suecia"]),
            _b("La Floresta", -0.2090, -78.4840, ["Av. Ladrón de Guevara", "Calle Galavis", "Calle Lugo"]),
            _b("Quito Tenis", -0.1690, -78.4970, ["Av. Brasil", "Av. Mariana de Jesús", "Calle Zamora"]),
            _b("Cumbayá", -0.2000, -78.4320, ["Av. Interoceánica", "Calle Francisco de Orellana", "Calle Pampite"])],
    "PTY": [_b("Obarrio", 8.9840, -79.5180, ["Calle 50", "Calle 53 Este", "Av. Samuel Lewis"]),
            _b("San Francisco", 8.9930, -79.5050, ["Vía Porras", "Calle 74", "Calle 69"]),
            _b("Bella Vista", 8.9800, -79.5280, ["Vía España", "Calle 43", "Av. Federico Boyd"]),
            _b("El Cangrejo", 8.9890, -79.5260, ["Vía Argentina", "Calle Eusebio A. Morales", "Calle D"]),
            _b("Costa del Este", 9.0080, -79.4680, ["Av. Paseo del Mar", "Av. Centenario", "Calle Principal"]),
            _b("Punta Pacífica", 8.9790, -79.5070, ["Calle Punta Darién", "Av. Balboa", "Calle Punta Colón"]),
            _b("Marbella", 8.9810, -79.5150, ["Calle 54 Este", "Av. Balboa", "Calle 53 Marbella"]),
            _b("Paitilla", 8.9780, -79.5110, ["Calle 56 Este", "Av. Balboa", "Calle Uruguay"])],
}

COLUMNAS_FINALES = [
    # originales (hoja Datos)
    "pizza_id", "order_id", "pizza_name_id", "quantity", "order_date", "order_time", "unit_price", "total_price",
    "pizza_size", "pizza_category", "pizza_ingredients", "pizza_name",
    # calculadas (hoja Datos)
    "total_calculado", "verificacion", "dia_semana", "hora", "nuevo_pedido", "fecha_hora",
    # enriquecimiento: sucursal
    "sucursal_id", "sucursal", "ciudad", "pais", "sucursal_direccion", "sucursal_latitud", "sucursal_longitud",
    # enriquecimiento: cliente
    "cliente_id", "cliente_nombre", "cliente_direccion", "cliente_barrio", "cliente_latitud", "cliente_longitud",
    "distancia_km", "origen",
]


class EnriquecedorPizzeria:
    """Convierte la pizzería de una sede en una empresa multisede con clientes y un mes completo.

    Uso típico::

        enr = EnriquecedorPizzeria(llm, preprocesador)
        sucursales = enr.generar_sucursales()
        enr.generar_catalogo_ciudades()
        clientes = enr.generar_clientes()
        ventas_mes = enr.completar_mes(datos)
    """

    def __init__(self, llm, preprocesador: PreprocesadorPizzas | None = None,
                 ciudades=None, centros=None, semilla: int = 2015, clientes_base: int = 900,
                 anio: int = 2015, mes: int = 1):
        self.llm = llm
        self.pre = preprocesador or PreprocesadorPizzas()
        self.ciudades = ciudades or CIUDADES_DEFECTO
        self.centros = centros or CENTROS_DEFECTO
        self.semilla = semilla
        self.clientes_base = clientes_base
        self.anio, self.mes = anio, mes
        self.sede_principal = self.ciudades[0][0]
        self.plan = None
        self.sucursales = None
        self.catalogo_ciudad = {}
        self.fuentes = {}
        self.clientes = None

    # ------------------------------------------------------------------ 1. sucursales
    def prompt_sucursales(self) -> str:
        lista = "\n".join(f"- código {c}: {ciu}, {pais}" for c, ciu, pais in self.ciudades)
        return f"""Eres analista de una cadena de pizzerías que en {self._nombre_mes()} de {self.anio} tiene {len(self.ciudades)} sucursales:
{lista}
La sede principal es {self.ciudades[0][1]} (factor_volumen = 1.0). Menú: categorías Classic, Supreme, Veggie y Chicken.

Devuelve SOLO un JSON con esta estructura exacta:
{{
  "factores_dia_semana": {{"lunes": n, "martes": n, "miércoles": n, "jueves": n, "viernes": n, "sábado": n, "domingo": n}},
  "sucursales": [
    {{"codigo": "MED", "nombre_sucursal": "...", "direccion": "dirección real en formato local de una zona comercial",
      "barrio": "...", "latitud": 0.0, "longitud": 0.0, "factor_volumen": 1.0,
      "preferencias_categoria": {{"Classic": n, "Supreme": n, "Veggie": n, "Chicken": n}},
      "dias_especiales": [{{"fecha": "{self.anio}-{self.mes:02d}-DD", "factor": n, "motivo": "..."}}]}}
  ]
}}
Reglas: una sucursal por código y en el mismo orden; latitud y longitud decimales WGS84 reales de esa dirección;
factores_dia_semana entre 0.7 y 1.3; factor_volumen entre 0.6 y 1.4 según el tamaño del mercado;
preferencias_categoria entre 0.8 y 1.25; dias_especiales solo festivos o fechas reales de {self._nombre_mes()} de {self.anio}
de ese país con factor entre 0.6 y 1.4."""

    def _validar_sucursales(self, j):
        dias = {d: float(np.clip(float(j["factores_dia_semana"][d]), 0.6, 1.4)) for d in self.pre.DIAS_SEMANA}
        por_codigo = {s["codigo"]: s for s in j["sucursales"]}
        salida = []
        for cod, ciudad, _ in self.ciudades:
            s = por_codigo[cod]
            lat, lon = float(s["latitud"]), float(s["longitud"])
            if cod in self.centros and haversine_km(lat, lon, *self.centros[cod]) > 25:
                raise ValueError(f"coordenadas de {cod} fuera de {ciudad}")
            pref = {c: float(np.clip(float(s["preferencias_categoria"].get(c, 1)), 0.7, 1.4))
                    for c in self.pre.ORDEN_CATEGORIAS}
            especiales = []
            for e in s.get("dias_especiales", []) or []:
                f = pd.to_datetime(e["fecha"], errors="coerce")
                if pd.notna(f) and f.year == self.anio and f.month == self.mes:
                    especiales.append({"fecha": f.strftime("%Y-%m-%d"),
                                       "factor": float(np.clip(float(e["factor"]), 0.5, 1.5)),
                                       "motivo": str(e.get("motivo", ""))})
            salida.append({"codigo": cod, "nombre_sucursal": str(s["nombre_sucursal"]),
                           "direccion": str(s["direccion"]), "barrio": str(s.get("barrio", "")),
                           "latitud": lat, "longitud": lon,
                           "factor_volumen": 1.0 if cod == self.sede_principal
                           else float(np.clip(float(s["factor_volumen"]), 0.5, 1.5)),
                           "preferencias_categoria": pref, "dias_especiales": especiales})
        return {"factores_dia_semana": dias, "sucursales": salida}

    def generar_sucursales(self) -> pd.DataFrame:
        """LLM → sucursales (nombre, dirección, coordenadas) y plan de demanda."""
        self.plan, self.fuentes["sucursales"] = self.llm.pedir_json(
            "Sucursales", self.prompt_sucursales(), self._validar_sucursales, RESPALDO_SUCURSALES)
        self.sucursales = pd.DataFrame([{
            "sucursal_id": s["codigo"],
            "sucursal": s["nombre_sucursal"],
            "ciudad": ciu, "pais": pais,
            "sucursal_direccion": f'{s["direccion"]}, {s["barrio"]}, {ciu}'.replace(", ,", ","),
            "sucursal_latitud": round(s["latitud"], 5),
            "sucursal_longitud": round(s["longitud"], 5),
            "factor_volumen": s["factor_volumen"],
        } for s, (_, ciu, pais) in zip(self.plan["sucursales"], self.ciudades)])
        print("Fuente de los datos de sucursales:", self.fuentes["sucursales"])
        return self.sucursales

    def tabla_dias_semana(self) -> pd.DataFrame:
        return pd.DataFrame(self.plan["factores_dia_semana"], index=["factor"])

    def tabla_preferencias(self) -> pd.DataFrame:
        return pd.DataFrame([{"sucursal_id": s["codigo"], **s["preferencias_categoria"],
                              "días especiales": "; ".join(f'{e["fecha"][-2:]}/{self.mes:02d} ×{e["factor"]} ({e["motivo"]})'
                                                           for e in s["dias_especiales"])}
                             for s in self.plan["sucursales"]])

    # ------------------------------------------------------------------ 2. barrios y nombres
    @staticmethod
    def prompt_ciudad(ciudad, pais, lat, lon) -> str:
        return f"""Para la ciudad de {ciudad} ({pais}) hay una pizzería con domicilios ubicada en latitud {lat}, longitud {lon}.
Devuelve SOLO un JSON con esta estructura:
{{
  "barrios": [{{"barrio": "...", "latitud": 0.0, "longitud": 0.0, "vias": ["vía real 1", "vía real 2", "vía real 3"]}}],
  "nombres": ["30 nombres de pila comunes en {pais}"],
  "apellidos": ["30 apellidos comunes en {pais}"]
}}
Reglas: 10 barrios residenciales reales de {ciudad} a no más de 8 km de la pizzería, con sus coordenadas
WGS84 aproximadas (centro del barrio) y 3 calles, carreras o avenidas reales de cada barrio. Mezcla nombres masculinos y femeninos."""

    @staticmethod
    def _validador_ciudad(lat_s, lon_s, radio_km=12):
        def validar(j):
            barrios = []
            for b in j["barrios"]:
                la, lo = float(b["latitud"]), float(b["longitud"])
                vias = [str(v) for v in b.get("vias", []) if str(v).strip()]
                if haversine_km(la, lo, lat_s, lon_s) <= radio_km and vias:
                    barrios.append({"barrio": str(b["barrio"]), "latitud": la, "longitud": lo, "vias": vias[:3]})
            nombres = list(dict.fromkeys(str(n).strip() for n in j["nombres"] if str(n).strip()))
            apellidos = list(dict.fromkeys(str(a).strip() for a in j["apellidos"] if str(a).strip()))
            if len(barrios) < 5 or len(nombres) < 15 or len(apellidos) < 15:
                raise ValueError(f"pocos datos válidos: {len(barrios)} barrios, {len(nombres)} nombres, "
                                 f"{len(apellidos)} apellidos")
            return {"barrios": barrios, "nombres": nombres, "apellidos": apellidos}
        return validar

    def generar_catalogo_ciudades(self) -> pd.DataFrame:
        """LLM → barrios (con coordenadas y vías), nombres y apellidos de cada ciudad."""
        if self.sucursales is None:
            self.generar_sucursales()
        for (cod, ciudad, pais), (_, s) in zip(self.ciudades, self.sucursales.iterrows()):
            respaldo = {"barrios": RESPALDO_BARRIOS.get(cod, [_b(ciudad, s["sucursal_latitud"], s["sucursal_longitud"],
                                                                  ["Calle Principal"])]),
                        "nombres": _NOMBRES, "apellidos": _APELLIDOS}
            self.catalogo_ciudad[cod], self.fuentes[cod] = self.llm.pedir_json(
                f"Barrios y nombres de {ciudad}",
                self.prompt_ciudad(ciudad, pais, s["sucursal_latitud"], s["sucursal_longitud"]),
                self._validador_ciudad(s["sucursal_latitud"], s["sucursal_longitud"]),
                respaldo)
        return pd.DataFrame([{"sucursal_id": cod, "fuente": self.fuentes[cod], "barrios": len(c["barrios"]),
                              "nombres": len(c["nombres"]), "apellidos": len(c["apellidos"]),
                              "ejemplo de barrios": ", ".join(b["barrio"] for b in c["barrios"][:4])}
                             for cod, c in self.catalogo_ciudad.items()])

    # ------------------------------------------------------------------ 3. clientes
    @staticmethod
    def formatear_direccion(pais, via, rng) -> str:
        a, b, c = rng.integers(1, 120), rng.integers(1, 99), rng.integers(1, 99)
        if pais == "Colombia":
            return f"{via} # {a}-{b:02d}"
        if pais == "Ecuador":
            return f"{via} N{a}-{rng.integers(10, 250)}"
        return f"{via}, {'Edificio' if rng.random() < 0.6 else 'Casa'} {c}"

    def generar_clientes(self, dispersion_grados: float = 0.0055) -> pd.DataFrame:
        """Clientes con nombre, dirección, barrio, latitud, longitud y distancia a su sucursal."""
        if not self.catalogo_ciudad:
            self.generar_catalogo_ciudades()
        rng = np.random.default_rng(self.semilla)
        filas = []
        for _, s in self.sucursales.iterrows():
            cat = self.catalogo_ciudad[s["sucursal_id"]]
            n = int(round(self.clientes_base * s["factor_volumen"]))
            n = min(n, len(cat["nombres"]) * len(cat["apellidos"]) ** 2)
            usados, i = set(), 0
            while i < n:
                nombre = f'{rng.choice(cat["nombres"])} {rng.choice(cat["apellidos"])} {rng.choice(cat["apellidos"])}'
                if nombre in usados:
                    continue
                usados.add(nombre)
                b = cat["barrios"][rng.integers(len(cat["barrios"]))]
                i += 1
                filas.append({
                    "cliente_id": f'CLI-{s["sucursal_id"]}-{i:04d}',
                    "sucursal_id": s["sucursal_id"],
                    "cliente_nombre": nombre,
                    "cliente_direccion": f'{self.formatear_direccion(s["pais"], rng.choice(b["vias"]), rng)}, '
                                         f'{b["barrio"]}, {s["ciudad"]}',
                    "cliente_barrio": b["barrio"],
                    "cliente_latitud": round(b["latitud"] + rng.normal(0, dispersion_grados), 6),
                    "cliente_longitud": round(b["longitud"] + rng.normal(0, dispersion_grados), 6),
                    # unos pocos clientes piden mucho y la mayoría poco (distribución sesgada)
                    "peso_frecuencia": rng.lognormal(0, 1.0),
                })
        c = pd.DataFrame(filas).merge(self.sucursales[["sucursal_id", "sucursal_latitud", "sucursal_longitud"]],
                                      on="sucursal_id")
        c["distancia_km"] = haversine_km(c["sucursal_latitud"], c["sucursal_longitud"],
                                         c["cliente_latitud"], c["cliente_longitud"]).round(2)
        self.clientes = c.drop(columns=["sucursal_latitud", "sucursal_longitud"])
        print(f"Clientes creados: {len(self.clientes):,}")
        return self.clientes

    def resumen_clientes(self) -> pd.DataFrame:
        return self.clientes.groupby("sucursal_id").agg(
            clientes=("cliente_id", "count"), distancia_media_km=("distancia_km", "mean"),
            distancia_max_km=("distancia_km", "max")).round(2)

    # ------------------------------------------------------------------ 4. mes completo
    def completar_mes(self, datos: pd.DataFrame, jitter_seg: int = 900) -> pd.DataFrame:
        """Genera todos los pedidos del mes para todas las sucursales.

        - La sede principal conserva los pedidos reales de los días completos.
        - El resto se genera copiando pedidos reales (mismas pizzas, cantidades y precios),
          con volumen = promedio real × factor_volumen × factor_día_semana × factor_día_especial (Poisson).
        """
        if self.clientes is None:
            self.generar_clientes()
        rng = np.random.default_rng(self.semilla)
        dias_sem = self.pre.DIAS_SEMANA
        ultimo_dia_real = datos["order_date"].max() - pd.Timedelta(days=1)   # el último día del archivo está incompleto
        inicio = pd.Timestamp(self.anio, self.mes, 1)
        fechas = pd.date_range(inicio, inicio + pd.offsets.MonthEnd(0), freq="D")

        reales = datos[datos["order_date"] <= ultimo_dia_real]
        self.pedidos_dia_base = reales["order_id"].nunique() / reales["order_date"].nunique()
        self.base_real = reales
        print(f"Pedidos reales promedio por día en la sede principal: {self.pedidos_dia_base:.1f}")

        pool = datos.reset_index(drop=True)
        lineas_por_pedido = pool.groupby("order_id").indices
        ids_pool = np.array(list(lineas_por_pedido.keys()))
        mezcla = pd.crosstab(pool["order_id"], pool["pizza_category"], normalize="index").reindex(
            index=ids_pool, columns=self.pre.ORDEN_CATEGORIAS, fill_value=0).to_numpy()
        g = pool.groupby("order_id")
        segundos = (g["fecha_hora"].first() - g["order_date"].first()).dt.total_seconds()
        seg_min, seg_max = segundos.min(), segundos.max()

        bloques = []
        for s in self.plan["sucursales"]:
            cod = s["codigo"]
            pesos = (mezcla @ np.array([s["preferencias_categoria"][c] for c in self.pre.ORDEN_CATEGORIAS])) ** 3
            pesos = pesos / pesos.sum()
            especiales = {e["fecha"]: e["factor"] for e in s["dias_especiales"]}
            cli = self.clientes[self.clientes["sucursal_id"] == cod]
            p_cli = (cli["peso_frecuencia"] / cli["peso_frecuencia"].sum()).to_numpy()

            for fecha in fechas:
                if cod == self.sede_principal and fecha <= ultimo_dia_real:
                    dia = pool[pool["order_date"] == fecha].copy()
                    dia["origen"] = "real"
                    orden = dia["order_id"].to_numpy()
                else:
                    lam = (self.pedidos_dia_base * s["factor_volumen"]
                           * self.plan["factores_dia_semana"][dias_sem[fecha.dayofweek]]
                           * especiales.get(fecha.strftime("%Y-%m-%d"), 1.0))
                    n = rng.poisson(lam)
                    if n == 0:
                        continue
                    elegidos = rng.choice(ids_pool, size=n, p=pesos)
                    idx = np.concatenate([lineas_por_pedido[o] for o in elegidos])
                    copia = np.repeat(np.arange(n), [len(lineas_por_pedido[o]) for o in elegidos])
                    dia = pool.iloc[idx].copy()
                    dia["origen"] = "sintético"
                    seg = np.clip(segundos.loc[elegidos].to_numpy() + rng.uniform(-jitter_seg, jitter_seg, n),
                                  seg_min, seg_max)
                    dia["fecha_hora"] = fecha + pd.to_timedelta(seg[copia].round(), unit="s")
                    dia["order_date"] = fecha
                    orden = copia
                unicos = pd.unique(orden)
                mapa_cli = dict(zip(unicos, rng.choice(cli["cliente_id"].to_numpy(), size=len(unicos), p=p_cli)))
                dia["cliente_id"] = [mapa_cli[o] for o in orden]
                dia["clave_pedido"] = [f"{cod}-{fecha:%d}-{o}" for o in orden]
                dia["sucursal_id"] = cod
                bloques.append(dia)

        mes = pd.concat(bloques, ignore_index=True)
        mes = mes.sort_values(["fecha_hora", "sucursal_id", "clave_pedido"]).reset_index(drop=True)
        mes["order_id"] = pd.factorize(mes["clave_pedido"])[0] + 1
        mes["pizza_id"] = np.arange(1, len(mes) + 1)
        mes["order_time"] = mes["fecha_hora"].dt.time
        mes = (mes.merge(self.sucursales.drop(columns="factor_volumen"), on="sucursal_id", how="left")
                  .merge(self.clientes.drop(columns=["sucursal_id", "peso_frecuencia"]), on="cliente_id", how="left"))
        mes = self.pre.agregar_columnas_calculadas(mes)[COLUMNAS_FINALES].copy()
        for c in ["sucursal_id", "sucursal", "ciudad", "pais", "origen", "cliente_barrio"]:
            mes[c] = mes[c].astype("category")
        mes["sucursal_id"] = mes["sucursal_id"].cat.reorder_categories([c for c, *_ in self.ciudades])
        self.ventas_mes = mes
        print(f"Líneas del mes: {len(mes):,} | pedidos: {mes['order_id'].nunique():,} | "
              f"sucursales: {mes['sucursal_id'].nunique()} | días: {mes['order_date'].nunique()}")
        return mes

    def validar(self, ventas_mes: pd.DataFrame | None = None) -> pd.DataFrame:
        """Validaciones del dataset enriquecido y resumen real vs. sintético por sucursal."""
        v = self.ventas_mes if ventas_mes is None else ventas_mes
        c = self.pre.check
        dias_mes = (pd.Timestamp(self.anio, self.mes, 1) + pd.offsets.MonthEnd(0)).day
        print("Validaciones del dataset enriquecido:")
        c("Sin nulos", v.isna().sum().sum() == 0)
        c("Un único cliente y sucursal por pedido",
          (v.groupby("order_id")[["cliente_id", "sucursal_id"]].nunique() == 1).all().all())
        c("total_price = quantity × unit_price", (v["verificacion"] == "OK").all())
        c("Cubre todo el mes en cada sucursal",
          (v.groupby("sucursal_id", observed=True)["order_date"].nunique() == dias_mes).all())
        c("Datos reales intactos", abs(v.loc[v["origen"] == "real", "total_price"].sum()
                                       - self.base_real["total_price"].sum()) < 0.01)
        return (v.groupby(["sucursal_id", "origen"], observed=True)
                 .agg(líneas=("pizza_id", "count"), pedidos=("order_id", "nunique"), ventas=("total_price", "sum"))
                 .round(2))

    # ------------------------------------------------------------------ utilidades
    def _nombre_mes(self) -> str:
        return ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                "octubre", "noviembre", "diciembre"][self.mes - 1]
