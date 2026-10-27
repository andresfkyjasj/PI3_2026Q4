"""Preprocesamiento y estandarización del dataset de ventas de pizzas.

Replica lo que se hace en las hojas «Tipos» y «Datos» del Excel de clase:
diagnóstico de tipos, conversión (VALOR.NUMERO, FECHA, HORANUMERO), categorías
y columnas calculadas (total_calculado, verificación, día_semana, hora, nuevo_pedido).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PreprocesadorPizzas:
    """Convierte el DataFrame crudo de Hugging Face en un DataFrame tipado y listo para analizar."""

    DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    ORDEN_TAMANOS = ["S", "M", "L", "XL", "XXL"]
    ORDEN_CATEGORIAS = ["Classic", "Supreme", "Veggie", "Chicken"]
    COLUMNAS_TEXTO = ["pizza_name_id", "order_date", "order_time", "pizza_size",
                      "pizza_category", "pizza_ingredients", "pizza_name"]
    # Totales del Excel de clase (primeras 1.700 líneas)
    ESPERADO_EXCEL = {"ventas": 28663.15, "pizzas": 1734, "lineas": 1700, "pedidos": 755}

    def __init__(self, filas_clase: int | None = 1700):
        self.filas_clase = filas_clase

    # ------------------------------------------------------------------ pasos
    def recortar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Toma las mismas filas que el Excel de clase (``None`` = todas)."""
        return df.copy() if self.filas_clase is None else df.head(self.filas_clase).copy()

    @staticmethod
    def diagnosticar(df: pd.DataFrame) -> pd.DataFrame:
        """Equivalente a ESTEXTO / ESNUMERO / CONTARA por columna."""
        filas = []
        for c in df.columns:
            s = df[c]
            filas.append({
                "columna": c,
                "tipo_actual": str(s.dtype),
                "ejemplo": s.iloc[0],
                "no_vacíos (CONTARA)": int(s.notna().sum()),
                "convertibles_a_número": int(pd.to_numeric(s, errors="coerce").notna().sum()),
                "¿es número? (ESNUMERO)": pd.api.types.is_numeric_dtype(s),
            })
        return pd.DataFrame(filas)

    @staticmethod
    def diccionario() -> pd.DataFrame:
        """Diccionario de columnas (hoja Tipos, sección B)."""
        return pd.DataFrame([
            ["pizza_id", "Número de la línea", "Cualitativa (identificador)", "Int64 (entero)"],
            ["order_id", "Número del pedido", "Cualitativa (identificador)", "Int64 (entero)"],
            ["pizza_name_id", "Código de pizza y tamaño", "Cualitativa nominal", "string"],
            ["quantity", "Pizzas en la línea", "Cuantitativa discreta", "int64"],
            ["order_date", "Fecha del pedido (día/mes/año)", "Cuantitativa (tiempo)", "datetime64"],
            ["order_time", "Hora del pedido", "Cuantitativa (tiempo)", "time (hh:mm:ss)"],
            ["unit_price", "Precio de una pizza", "Cuantitativa continua", "float64 (moneda, 2 dec.)"],
            ["total_price", "Cantidad × precio unitario", "Cuantitativa continua", "float64 (moneda, 2 dec.)"],
            ["pizza_size", "Tamaño", "Cualitativa ordinal (S < M < L < XL)", "category ordenada"],
            ["pizza_category", "Categoría", "Cualitativa nominal", "category"],
            ["pizza_ingredients", "Ingredientes", "Texto libre", "string"],
            ["pizza_name", "Nombre de la pizza", "Cualitativa nominal", "string"],
        ], columns=["columna", "qué significa", "tipo de variable", "tipo en pandas"])

    def convertir_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hoja Tipos, secciones D y E: números, moneda, fecha, hora y categorías."""
        d = df.copy()
        for c in self.COLUMNAS_TEXTO:
            d[c] = d[c].astype("string").str.strip()

        # VALOR.NUMERO
        d["pizza_id"] = pd.to_numeric(d["pizza_id"], errors="coerce").round().astype("Int64")
        d["order_id"] = pd.to_numeric(d["order_id"], errors="coerce").round().astype("Int64")
        d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce").round().astype("int64")
        d["unit_price"] = pd.to_numeric(d["unit_price"], errors="coerce").astype("float64").round(2)
        d["total_price"] = pd.to_numeric(d["total_price"], errors="coerce").astype("float64").round(2)

        # SUSTITUIR("-"; "/") + FECHA(año; mes; día)
        d["order_date"] = pd.to_datetime(d["order_date"].str.replace("-", "/", regex=False),
                                         format="%d/%m/%Y", errors="coerce")
        # HORANUMERO
        hora = pd.to_datetime(d["order_time"], format="%H:%M:%S", errors="coerce")
        d["order_time"] = hora.dt.time
        d["fecha_hora"] = d["order_date"] + (hora - hora.dt.normalize())

        d["pizza_size"] = pd.Categorical(d["pizza_size"], categories=self.ORDEN_TAMANOS, ordered=True)
        d["pizza_category"] = pd.Categorical(d["pizza_category"], categories=self.ORDEN_CATEGORIAS)
        return d

    def agregar_columnas_calculadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columnas calculadas de la hoja Datos."""
        d = df.copy()
        d["total_calculado"] = (d["quantity"] * d["unit_price"]).round(2)                      # =D2*G2
        d["verificacion"] = np.where((d["total_calculado"] - d["total_price"]).abs() < 0.005,
                                     "OK", "Revisar")                                           # =SI(REDONDEAR(M2-H2;2)=0;...)
        d["dia_semana"] = pd.Categorical(d["order_date"].dt.dayofweek.map(dict(enumerate(self.DIAS_SEMANA))),
                                         categories=self.DIAS_SEMANA, ordered=True)             # =ELEGIR(DIASEM(E2;2);...)
        d["hora"] = d["fecha_hora"].dt.hour                                                     # =HORA(F2)
        d["nuevo_pedido"] = (~d["order_id"].duplicated()).astype("int64")                      # =SI(CONTAR.SI($B$2:B2;B2)=1;1;0)
        return d

    def procesar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flujo completo: recortar → convertir tipos → columnas calculadas."""
        return self.agregar_columnas_calculadas(self.convertir_tipos(self.recortar(df)))

    # ------------------------------------------------------------------ validaciones
    @staticmethod
    def check(nombre: str, condicion: bool, detalle: str = "") -> bool:
        print(f"{'✔' if condicion else '✘'} {nombre} {detalle}")
        return bool(condicion)

    def validar(self, datos: pd.DataFrame, comparar_excel: bool = True) -> bool:
        """Calidad de la base procesada y comparación con los totales del Excel."""
        c = self.check
        ok = [
            c("Sin valores nulos tras convertir", datos[["pizza_id", "order_id", "quantity", "order_date",
              "order_time", "unit_price", "total_price", "pizza_size", "pizza_category"]].isna().sum().sum() == 0),
            c("Sin filas duplicadas", not datos.duplicated(subset=["pizza_id"]).any()),
            c("total_price = quantity × unit_price", (datos["verificacion"] == "OK").all(),
              f"({(datos['verificacion'] == 'Revisar').sum()} filas a revisar)"),
            c("Fechas en orden cronológico según order_id",
              datos.sort_values("order_id")["fecha_hora"].is_monotonic_increasing),
        ]
        if comparar_excel and self.filas_clase == self.ESPERADO_EXCEL["lineas"]:
            e = self.ESPERADO_EXCEL
            ventas, pizzas, pedidos = datos["total_price"].sum(), datos["quantity"].sum(), datos["nuevo_pedido"].sum()
            print("\nComparación con el Excel de clase:")
            ok += [
                c(f"Ventas totales   = ${e['ventas']:,.2f}", abs(ventas - e["ventas"]) < 0.01, f"→ ${ventas:,.2f}"),
                c(f"Pizzas vendidas  = {e['pizzas']:,}", pizzas == e["pizzas"], f"→ {pizzas:,}"),
                c(f"Líneas de venta  = {e['lineas']:,}", len(datos) == e["lineas"], f"→ {len(datos):,}"),
                c(f"Pedidos          = {e['pedidos']:,}", pedidos == e["pedidos"], f"→ {pedidos:,}"),
            ]
        ultimo = datos["order_date"].max()
        print(f"\nRango de fechas: {datos['order_date'].min():%d/%m/%Y} a {ultimo:%d/%m/%Y} "
              f"({datos['order_date'].nunique()} días; el último tiene {(datos['order_date'] == ultimo).sum()} líneas)")
        return all(ok)
