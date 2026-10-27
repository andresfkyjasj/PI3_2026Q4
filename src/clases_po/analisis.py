"""Analítica descriptiva de la pizzería.

- :class:`AnalisisBase` replica el Excel de clase (Fórmulas, Descriptiva, Relaciones,
  Tabla dinámica, Dashboard y Práctica) sobre CUALQUIER DataFrame de ventas ya
  preprocesado, por ejemplo la base original sin enriquecer (1 sede, 1 al 13 de enero).
- :class:`AnalisisPizzeria` hereda todo lo anterior y agrega el análisis multisede:
  sucursal, ciudad/país, clientes y distancia.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
ORDEN_CATEGORIAS = ["Classic", "Supreme", "Veggie", "Chicken"]


class AnalisisBase:
    """Análisis del Excel de clase sobre un DataFrame de ventas (una fila por línea de pedido)."""

    PALABRAS_DINERO = ("ventas", "ticket", "precio", "valor", "venta", "suma")
    CORTES_PRECIO = [9, 12, 15, 18, 21, 24, 27, 100]

    def __init__(self, ventas: pd.DataFrame, pizza_elegida: str = "The Pepperoni Pizza", dias_incompletos=None):
        """``dias_incompletos``: fechas que se excluyen de los PROMEDIOS por día de la semana
        (por ejemplo el 13/01, que en el archivo solo tiene 50 líneas). Los totales no cambian."""
        self.V = ventas
        self.pizza_elegida = pizza_elegida
        self.dias_incompletos = pd.to_datetime(list(dias_incompletos or []))
        self.pedidos = self._construir_pedidos()

    def _construir_pedidos(self) -> pd.DataFrame:
        """Una fila por pedido: valor del pedido (ticket) y pizzas (más sucursal/cliente si existen)."""
        agg = {"order_date": ("order_date", "first"), "hora": ("hora", "first"),
               "dia_semana": ("dia_semana", "first"), "valor_pedido": ("total_price", "sum"),
               "pizzas": ("quantity", "sum")}
        for c in ["sucursal_id", "ciudad", "cliente_id", "distancia_km", "origen"]:
            if c in self.V.columns:
                agg[c] = (c, "first")
        return self.V.groupby("order_id").agg(**agg).reset_index()

    # ================================================================== utilidades
    @classmethod
    def estilo(cls, df: pd.DataFrame, dinero=None, pct=None, dec=None, enteros=None):
        """Formato automático: columnas de dinero ($), de porcentaje (%) y conteos (miles).

        Detecta por nombre de columna; los parámetros permiten forzar columnas concretas.
        """
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        nombre = {c: str(c).lower() for c in num}
        pct = list(pct) if pct is not None else [c for c in num if nombre[c].startswith("%") or "corr" in nombre[c]]
        dinero = list(dinero) if dinero is not None else [
            c for c in num if c not in pct and any(w in nombre[c] for w in cls.PALABRAS_DINERO)]
        enteros = list(enteros) if enteros is not None else [
            c for c in num if c not in pct + dinero and pd.api.types.is_integer_dtype(df[c])]
        dec = list(dec) if dec is not None else [c for c in num if c not in pct + dinero + enteros]
        fmt = {**{c: "${:,.2f}" for c in dinero}, **{c: "{:.1%}" for c in pct},
               **{c: "{:,.2f}" for c in dec}, **{c: "{:,.0f}" for c in enteros}}
        try:
            return df.style.format(fmt, na_rep="—")
        except Exception:  # noqa: BLE001  (si falta jinja2 se muestra sin formato)
            return df

    def filtrar(self, sucursal="Todas", categoria="Todas", tamano="Todos", dia_semana="Todos") -> pd.DataFrame:
        d = self.V
        if sucursal != "Todas" and "sucursal_id" in d.columns:
            d = d[d["sucursal_id"] == sucursal]
        if categoria != "Todas":
            d = d[d["pizza_category"] == categoria]
        if tamano != "Todos":
            d = d[d["pizza_size"] == tamano]
        if dia_semana != "Todos":
            d = d[d["dia_semana"] == dia_semana]
        return d

    def periodo(self) -> str:
        f = self.V["order_date"]
        return f"{f.min():%d/%m/%Y} a {f.max():%d/%m/%Y} ({f.nunique()} días)"

    # ================================================================== Fórmulas (hoja «Fórmulas»)
    def formulas_basicas(self, fila_excel: int = 185) -> pd.DataFrame:
        """Hoja Fórmulas, secciones B, C, E y F: cada fórmula de Excel con su equivalente en pandas.

        ``fila_excel`` es la fila de la hoja Datos usada en las operaciones con una fila (la 185 = 3 Spicy Italian).
        """
        V = self.V.reset_index(drop=True)
        i = fila_excel - 2                      # la fila 2 de Excel es el índice 0
        cat, pz = V["pizza_category"], V["pizza_name"]
        filas = [
            # B · operaciones con una fila
            (1, "Multiplicar: total de la línea", f"=Datos!D{fila_excel}*Datos!G{fila_excel}",
             "V.quantity[i] * V.unit_price[i]", V.loc[i, "quantity"] * V.loc[i, "unit_price"]),
            (2, "Dividir: precio de cada pizza", f"=Datos!H{fila_excel}/Datos!D{fila_excel}",
             "V.total_price[i] / V.quantity[i]", V.loc[i, "total_price"] / V.loc[i, "quantity"]),
            (3, "Sumar: pizzas en dos líneas", f"=Datos!D{fila_excel}+Datos!D{fila_excel + 1}",
             "V.quantity[i] + V.quantity[i+1]", V.loc[i, "quantity"] + V.loc[i + 1, "quantity"]),
            (4, "Restar: precio más alto − más bajo", "=MAX(G)-MIN(G)",
             "V.unit_price.max() - V.unit_price.min()", V["unit_price"].max() - V["unit_price"].min()),
            (5, "Redondear el precio promedio", "=REDONDEAR(PROMEDIO(G);2)",
             "round(V.unit_price.mean(), 2)", round(V["unit_price"].mean(), 2)),
            # C · funciones de resumen
            (6, "Ventas totales", "=SUMA(H)", "V.total_price.sum()", V["total_price"].sum()),
            (7, "Pizzas vendidas", "=SUMA(D)", "V.quantity.sum()", V["quantity"].sum()),
            (8, "Líneas con número", "=CONTAR(H)", "V.total_price.count()", V["total_price"].count()),
            (9, "Líneas con nombre de pizza", "=CONTARA(L)", "V.pizza_name.notna().sum()", pz.notna().sum()),
            (10, "Celdas vacías en la categoría", "=CONTAR.BLANCO(J)", "V.pizza_category.isna().sum()", cat.isna().sum()),
            (11, "Precio promedio de una línea", "=PROMEDIO(H)", "V.total_price.mean()", V["total_price"].mean()),
            (12, "Precio unitario más bajo", "=MIN(G)", "V.unit_price.min()", V["unit_price"].min()),
            (13, "Precio unitario más alto", "=MAX(G)", "V.unit_price.max()", V["unit_price"].max()),
            (14, "Multiplicar y sumar en un paso", "=SUMAPRODUCTO(D;G)", "(V.quantity * V.unit_price).sum()",
             (V["quantity"] * V["unit_price"]).sum()),
            # E · valores únicos
            (16, "Categorías sin repetir", "=UNICOS(J)", "V.pizza_category.unique()", " · ".join(map(str, cat.unique()))),
            (17, "Categorías en orden alfabético", "=ORDENAR(UNICOS(J))", "sorted(V.pizza_category.unique())",
             " · ".join(sorted(map(str, cat.unique())))),
            (18, "Tamaños sin repetir", "=UNICOS(I)", "V.pizza_size.unique()",
             " · ".join(map(str, V["pizza_size"].unique()))),
            (19, "Categorías distintas", "=CONTARA(UNICOS(J))", "V.pizza_category.nunique()", cat.nunique()),
            (21, "Pizzas distintas en el menú", "=SUMAPRODUCTO(1/CONTAR.SI(L;L))", "V.pizza_name.nunique()", pz.nunique()),
            (22, "Pedidos distintos", "=SUMA(Q)  (nuevo_pedido)", "V.nuevo_pedido.sum()", V["nuevo_pedido"].sum()),
            # F · una condición
            (23, "¿Cuántas líneas son Veggie?", '=CONTAR.SI(J;"Veggie")', '(V.pizza_category == "Veggie").sum()',
             (cat == "Veggie").sum()),
            (24, "¿Cuánto vendió Veggie?", '=SUMAR.SI(J;"Veggie";H)', 'V.loc[cat == "Veggie", "total_price"].sum()',
             V.loc[cat == "Veggie", "total_price"].sum()),
            (25, "Precio unitario promedio de Veggie", '=PROMEDIO.SI(J;"Veggie";G)',
             'V.loc[cat == "Veggie", "unit_price"].mean()', V.loc[cat == "Veggie", "unit_price"].mean()),
            (26, "Líneas con más de una pizza", '=CONTAR.SI(D;">1")', "(V.quantity > 1).sum()", (V["quantity"] > 1).sum()),
            (27, "Ventas de pizzas de $20 o más", '=SUMAR.SI(G;">=20";H)', 'V.loc[V.unit_price >= 20, "total_price"].sum()',
             V.loc[V["unit_price"] >= 20, "total_price"].sum()),
            (28, f"Ventas del {V['order_date'].min():%d/%m/%Y}", "=SUMAR.SI(E;FECHA(...);H)",
             'V.loc[V.order_date == "...", "total_price"].sum()',
             V.loc[V["order_date"] == V["order_date"].min(), "total_price"].sum()),
        ]
        t = pd.DataFrame(filas, columns=["#", "qué quiero saber", "fórmula en Excel", "equivalente en pandas", "resultado"])
        t["resultado"] = [round(float(r), 2) if isinstance(r, (int, float, np.integer, np.floating)) else r
                          for r in t["resultado"]]
        return t.astype({"resultado": object})

    def practica(self) -> pd.DataFrame:
        """Hoja Práctica: las 11 preguntas con su respuesta calculada (como la hoja Soluciones)."""
        V, cat = self.V, self.V["pizza_category"]
        dia5 = V["order_date"].min() + pd.Timedelta(days=4)
        filas = [
            (0, "¿Cuántas líneas de venta hay?", "CONTARA", len(V)),
            (1, "Ventas totales de la categoría Veggie", "SUMAR.SI", V.loc[cat == "Veggie", "total_price"].sum()),
            (2, "Pizzas vendidas de tamaño L", "SUMAR.SI", V.loc[V["pizza_size"] == "L", "quantity"].sum()),
            (3, "Precio unitario promedio de la categoría Chicken", "PROMEDIO.SI",
             V.loc[cat == "Chicken", "unit_price"].mean()),
            (4, f"Ventas del {dia5:%d/%m/%Y}", "SUMAR.SI y FECHA", V.loc[V["order_date"] == dia5, "total_price"].sum()),
            (5, "¿Cuántas líneas tienen más de una pizza?", 'CONTAR.SI con ">1"', (V["quantity"] > 1).sum()),
            (6, "Precio unitario más alto", "MAX", V["unit_price"].max()),
            (7, "Mediana de total_price", "MEDIANA", V["total_price"].median()),
            (8, "Ventas de Supreme en tamaño M", "SUMAR.SI.CONJUNTO",
             V.loc[(cat == "Supreme") & (V["pizza_size"] == "M"), "total_price"].sum()),
            (9, "¿Cuántas pizzas distintas hay en el menú?", "CONTARA(UNICOS(…))", V["pizza_name"].nunique()),
            (10, "Ventas entre las 18:00 y las 20:59", "SUMAR.SI.CONJUNTO con la columna hora",
             V.loc[V["hora"].between(18, 20), "total_price"].sum()),
        ]
        t = pd.DataFrame(filas, columns=["#", "pregunta", "pista: función", "respuesta"])
        t["respuesta"] = t["respuesta"].astype(float).round(2)
        return t

    def pregunta_abierta(self) -> pd.DataFrame:
        """«¿Por qué Pepperoni tiene más líneas que cualquier otra pizza pero no es la que más vende?»"""
        t = (self.V.groupby("pizza_name")
             .agg(líneas=("pizza_id", "count"), pizzas=("quantity", "sum"), ventas=("total_price", "sum"),
                  precio_unit_promedio=("unit_price", "mean")))
        t["puesto por líneas"] = t["líneas"].rank(ascending=False, method="min").astype(int)
        t["puesto por ventas"] = t["ventas"].rank(ascending=False, method="min").astype(int)
        return t.sort_values("líneas", ascending=False).head(6)

    # ================================================================== Descriptiva A · resumen del negocio
    @staticmethod
    def resumen_negocio(d: pd.DataFrame) -> pd.Series:
        ventas, pizzas = d["total_price"].sum(), d["quantity"].sum()
        pedidos, dias = d["nuevo_pedido"].sum(), d["order_date"].nunique()
        r = {
            "Ventas totales": ventas,                       # SUMA(total_price)
            "Pizzas vendidas": pizzas,                      # SUMA(quantity)
            "Líneas de pedido": len(d),                     # CONTARA
            "Pedidos": pedidos,                             # SUMA(nuevo_pedido)
            "Ticket promedio": ventas / pedidos,            # ventas ÷ pedidos
            "Pizzas por pedido": pizzas / pedidos,
            "Precio promedio por pizza": ventas / pizzas,
            "Días con ventas": dias,
            "Venta promedio por día": ventas / dias,
        }
        if "cliente_id" in d.columns:
            r["Clientes únicos"] = d["cliente_id"].nunique()
            r["Pedidos por cliente"] = pedidos / r["Clientes únicos"]
        if "distancia_km" in d.columns:
            r["Distancia media (km)"] = d.drop_duplicates("order_id")["distancia_km"].mean()
        return pd.Series(r)

    def resumen_total(self) -> pd.DataFrame:
        return self.resumen_negocio(self.V).to_frame("Empresa")

    # ================================================================== Descriptiva B · medidas
    @staticmethod
    def medidas(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return pd.Series({
            "n (datos)": s.count(), "Suma": s.sum(), "Media": s.mean(), "Mediana": s.median(),
            "Moda": s.mode().iloc[0], "Mínimo": s.min(), "Máximo": s.max(), "Rango": s.max() - s.min(),
            "Desviación estándar": s.std(ddof=1), "Coef. de variación": s.std(ddof=1) / s.mean(),
            "Q1 (25 %)": s.quantile(0.25), "Q3 (75 %)": s.quantile(0.75),
        })

    def medidas_variables(self) -> pd.DataFrame:
        return pd.DataFrame({"quantity": self.medidas(self.V["quantity"]),
                             "unit_price": self.medidas(self.V["unit_price"]),
                             "total_price": self.medidas(self.V["total_price"]),
                             "valor_pedido": self.medidas(self.pedidos["valor_pedido"])})

    def lectura_media_mediana(self) -> str:
        m = self.medidas(self.V["total_price"])
        rel = "mayor" if m["Media"] > m["Mediana"] else "menor"
        return (f"Lectura: en total_price la media (${m['Media']:.2f}) es {rel} que la mediana (${m['Mediana']:.2f}): "
                "unas pocas líneas con varias pizzas jalan el promedio hacia arriba (sesgo a la derecha).")

    # ================================================================== Descriptiva C-D · frecuencias
    def frecuencias(self, columna: str, d: pd.DataFrame | None = None) -> pd.DataFrame:
        d = self.V if d is None else d
        t = (d.groupby(columna, observed=True)
             .agg(líneas=("pizza_id", "count"), pizzas=("quantity", "sum"), ventas=("total_price", "sum"),
                  precio_unit_promedio=("unit_price", "mean")))
        t.insert(1, "% líneas", t["líneas"] / t["líneas"].sum())
        t.insert(2, "% acumulado", t["% líneas"].cumsum())
        t.insert(5, "% ventas", t["ventas"] / t["ventas"].sum())
        t.index = t.index.astype(str)
        total = pd.DataFrame({"líneas": [t["líneas"].sum()], "% líneas": [1.0], "% acumulado": [np.nan],
                              "pizzas": [t["pizzas"].sum()], "ventas": [t["ventas"].sum()], "% ventas": [1.0],
                              "precio_unit_promedio": [d["unit_price"].mean()]}, index=["Total"])
        return pd.concat([t, total])

    # ================================================================== Descriptiva E · distribución
    def distribucion_precio(self) -> pd.DataFrame:
        c = self.CORTES_PRECIO
        etiquetas = [f"${a} a ${b}" if b < 100 else f"${a} o más" for a, b in zip(c[:-1], c[1:])]
        t = (pd.cut(self.V["total_price"], bins=c, right=False, labels=etiquetas)
             .value_counts(sort=False).to_frame("líneas"))
        t["% líneas"] = t["líneas"] / t["líneas"].sum()
        t.index = t.index.astype(str)
        return t

    # ================================================================== Relaciones A · por día
    def ventas_por_dia(self, d: pd.DataFrame | None = None) -> pd.DataFrame:
        d = self.V if d is None else d
        t = (d.groupby("order_date")
             .agg(día=("dia_semana", "first"), líneas=("pizza_id", "count"), pizzas=("quantity", "sum"),
                  pedidos=("nuevo_pedido", "sum"), ventas=("total_price", "sum")))
        t["día"] = t["día"].astype(str)
        t["ticket promedio"] = t["ventas"] / t["pedidos"]
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        t["ventas pizza elegida"] = d[d["pizza_name"] == self.pizza_elegida].groupby("order_date")["total_price"].sum()
        t["ventas pizza elegida"] = t["ventas pizza elegida"].fillna(0)
        return t

    def lectura_dias(self, excluir_ultimo_incompleto: bool = False) -> str:
        t = self.ventas_por_dia()
        mejor = t["ventas"].idxmax()
        base_min = t.iloc[:-1] if excluir_ultimo_incompleto else t
        peor = base_min["ventas"].idxmin()
        nota = " (sin el último día, que está incompleto)" if excluir_ultimo_incompleto else ""
        pe = t["ventas pizza elegida"]
        return (f"Día con más ventas: {mejor:%d/%m/%Y} ({t.loc[mejor, 'día']}) → ${t.loc[mejor, 'ventas']:,.2f}\n"
                f"Día con menos ventas{nota}: {peor:%d/%m/%Y} ({t.loc[peor, 'día']}) → ${t.loc[peor, 'ventas']:,.2f}\n"
                f"{self.pizza_elegida}: ventas ${pe.sum():,.2f}; su mejor día {pe.idxmax():%d/%m/%Y}")

    def dia_semana_empresa(self) -> pd.Series:
        """Venta PROMEDIO por día de la semana."""
        return self.dia_semana_resumen()["venta promedio por día"]

    def dia_semana_resumen(self) -> pd.DataFrame:
        d = self.V[~self.V["order_date"].isin(self.dias_incompletos)]
        diario = d.groupby(["order_date", "dia_semana"], observed=True)["total_price"].sum().reset_index()
        t = diario.groupby("dia_semana", observed=True)["total_price"].agg(["count", "sum", "mean"])
        t.columns = ["días en el periodo", "ventas", "venta promedio por día"]
        t.index = t.index.astype(str)
        return t

    # ================================================================== Relaciones B-C
    def categoria_x_tamano(self, medida: str = "ventas", d: pd.DataFrame | None = None) -> pd.DataFrame:
        d = self.V if d is None else d
        if medida == "ventas":
            t = pd.crosstab(d["pizza_category"], d["pizza_size"], values=d["total_price"], aggfunc="sum",
                            margins=True, margins_name="Total").fillna(0)
        else:
            t = pd.crosstab(d["pizza_category"], d["pizza_size"], margins=True, margins_name="Total")
        t.index, t.columns = t.index.astype(str), t.columns.astype(str)
        if medida == "ventas":
            t.columns = [f"ventas {c}" for c in t.columns]
        return t

    def combinacion_lider(self) -> str:
        t = pd.crosstab(self.V["pizza_category"], self.V["pizza_size"], values=self.V["total_price"], aggfunc="sum")
        s = t.stack()
        return f"Lectura: la combinación que más vende es {s.idxmax()[0]} {s.idxmax()[1]} (${s.max():,.2f})."

    def consulta(self, fecha="2015-01-08", categoria="Veggie", tamano="L", sucursal="Todas") -> pd.Series:
        """SUMAR.SI.CONJUNTO con varias condiciones ("Todas"/"Todos" ignora la condición)."""
        d = self.filtrar(sucursal, categoria, tamano)
        d = d[d["order_date"] == pd.Timestamp(fecha)]
        return pd.Series({"Ventas": round(d["total_price"].sum(), 2), "Pizzas": int(d["quantity"].sum()),
                          "Líneas": len(d),
                          "Precio promedio": round(d["unit_price"].mean(), 2) if len(d) else "No hubo ventas"})

    # ================================================================== Relaciones D · hora
    def ventas_por_hora(self) -> pd.DataFrame:
        t = self.V.groupby("hora").agg(líneas=("pizza_id", "count"), ventas=("total_price", "sum"))
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        return t

    # ================================================================== Relaciones E · ranking
    def ranking_pizzas(self, d: pd.DataFrame | None = None) -> pd.DataFrame:
        d = self.V if d is None else d
        t = (d.groupby("pizza_name")
             .agg(categoría=("pizza_category", "first"), pizzas=("quantity", "sum"), ventas=("total_price", "sum"))
             .sort_values("ventas", ascending=False))
        t["categoría"] = t["categoría"].astype(str)
        t.insert(0, "posición", t["ventas"].rank(ascending=False, method="min").astype(int))
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        t["% acumulado"] = t["% ventas"].cumsum()
        return t

    def pizzas_para_80(self) -> int:
        return int((self.ranking_pizzas()["% acumulado"] < 0.8).sum() + 1)

    # ================================================================== Tabla dinámica
    def tabla_dinamica(self, filas="pizza_category", columnas="pizza_size", valores="total_price",
                       sucursal="Todas", dia_semana="Todos") -> pd.DataFrame:
        d = self.filtrar(sucursal=sucursal, dia_semana=dia_semana)
        t = d.pivot_table(index=filas, columns=columnas, values=valores, aggfunc="sum",
                          margins=True, margins_name="Total", observed=True).fillna(0).round(2)
        t.columns = [f"suma {c}" for c in t.columns]
        return t

    # ================================================================== Dashboard (KPIs)
    def kpis(self, sucursal="Todas", categoria="Todas", tamano="Todos") -> dict:
        d = self.filtrar(sucursal, categoria, tamano)
        if d.empty:
            return {}
        dia = d.groupby("order_date")["total_price"].sum()
        hora = d.groupby("hora")["total_price"].sum()
        k = {"VENTAS": f"${d['total_price'].sum():,.2f}", "PIZZAS": f"{d['quantity'].sum():,}",
             "PEDIDOS": f"{d['order_id'].nunique():,}"}
        if "cliente_id" in d.columns:
            k["CLIENTES"] = f"{d['cliente_id'].nunique():,}"
        else:
            k["LÍNEAS"] = f"{len(d):,}"
        k.update({"PRECIO PROM./PIZZA": f"${d['total_price'].sum() / d['quantity'].sum():,.2f}",
                  "% VENTAS TOTALES": f"{d['total_price'].sum() / self.V['total_price'].sum():.1%}",
                  "MEJOR DÍA": f"{dia.idxmax():%d/%m}", "HORA PICO": f"{hora.idxmax()}:00",
                  "PIZZA LÍDER": d.groupby("pizza_name")["total_price"].sum().idxmax()})
        return k

    # ================================================================== conclusiones y exportación
    def hallazgos(self) -> list[str]:
        V, dias, dsem, hora = self.V, self.ventas_por_dia(), self.dia_semana_empresa(), self.ventas_por_hora()
        rk, cat, tam = self.ranking_pizzas(), self.frecuencias("pizza_category"), self.frecuencias("pizza_size")
        cat, tam = cat.drop("Total"), tam.drop("Total")
        mejor = dias["ventas"].idxmax()
        m = self.medidas(V["total_price"])
        return [
            f"En el periodo {self.periodo()} se vendieron ${V['total_price'].sum():,.2f}: {V['quantity'].sum():,} pizzas "
            f"en {V['nuevo_pedido'].sum():,} pedidos (ticket promedio ${V['total_price'].sum() / V['nuevo_pedido'].sum():.2f}; "
            f"{V['quantity'].sum() / V['nuevo_pedido'].sum():.2f} pizzas por pedido).",
            f"El día de mayor venta fue el {mejor:%d/%m} ({dias.loc[mejor, 'día']}); en promedio el día de la semana más "
            f"fuerte es el {dsem.idxmax()} y el más flojo el {dsem.idxmin()}.",
            f"La hora pico es las {hora['ventas'].idxmax()}:00; hay dos picos (almuerzo y comida).",
            f"Tamaño: {tam['líneas'].idxmax()} es el más vendido ({tam.loc[tam['líneas'].idxmax(), '% ventas']:.1%} de las ventas). "
            f"Categoría: {cat['ventas'].idxmax()} lidera en ventas y {cat['líneas'].idxmax()} en líneas.",
            f"{rk.index[0]} es la pizza que más vende; {self.pizzas_para_80()} de {len(rk)} pizzas explican el 80 % de las ventas.",
            f"total_price: media ${m['Media']:.2f} vs. mediana ${m['Mediana']:.2f} → distribución sesgada a la derecha.",
        ]

    def tablas_exportables(self) -> dict[str, pd.DataFrame]:
        return {
            "Datos": self.V.assign(order_time=self.V["order_time"].astype(str)),
            "Formulas": self.formulas_basicas(),
            "Resumen_negocio": self.resumen_total().reset_index(names="indicador"),
            "Medidas": self.medidas_variables().reset_index(names="medida"),
            "Frec_tamano": self.frecuencias("pizza_size").reset_index(names="tamaño"),
            "Frec_categoria": self.frecuencias("pizza_category").reset_index(names="categoría"),
            "Distribucion_precio": self.distribucion_precio().reset_index(names="intervalo"),
            "Ventas_dia": self.ventas_por_dia().reset_index(),
            "Dia_semana": self.dia_semana_resumen().reset_index(names="día"),
            "Categoria_x_tamano": self.categoria_x_tamano().reset_index(),
            "Ventas_hora": self.ventas_por_hora().reset_index(),
            "Ranking_pizzas": self.ranking_pizzas().reset_index(),
            "Tabla_dinamica": self.tabla_dinamica().reset_index(),
            "Practica": self.practica(),
        }

    def exportar_excel(self, ruta="reportes/analisis_pizzeria.xlsx") -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tablas = self.tablas_exportables()
        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            for nombre, t in tablas.items():
                t = t.copy()
                for c in t.columns:
                    if isinstance(t[c].dtype, pd.CategoricalDtype):
                        t[c] = t[c].astype(str)
                t.columns = [str(c) for c in t.columns]
                t.to_excel(writer, sheet_name=nombre[:31], index=False)
        print(f"Excel con {len(tablas)} hojas guardado en {ruta}")
        return ruta


class AnalisisPizzeria(AnalisisBase):
    """Análisis del dataset enriquecido: todo lo del Excel + sucursal, ciudad/país, clientes y distancia."""

    CORTES_KM = [0, 2, 4, 6, 8, np.inf]
    ETIQUETAS_KM = ["0-2 km", "2-4 km", "4-6 km", "6-8 km", "8+ km"]

    def __init__(self, ventas: pd.DataFrame, sucursales: pd.DataFrame, clientes: pd.DataFrame,
                 pizza_elegida: str = "The Pepperoni Pizza"):
        self.sucursales = sucursales
        self.clientes = clientes
        self.SUC = list(sucursales["sucursal_id"])
        self.nombre_suc = dict(zip(sucursales["sucursal_id"], sucursales["ciudad"]))
        super().__init__(ventas, pizza_elegida)

    def _construir_pedidos(self) -> pd.DataFrame:
        p = super()._construir_pedidos()
        p["rango_distancia"] = pd.cut(p["distancia_km"], bins=self.CORTES_KM, labels=self.ETIQUETAS_KM, right=False)
        return p

    def _con_nombres(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [f"{c} · {self.nombre_suc.get(c, c)}" for c in df.columns]
        return df

    # ================================================================== resumen por sucursal
    def resumen_total(self) -> pd.DataFrame:
        por_suc = pd.concat({c: self.resumen_negocio(g) for c, g in self.V.groupby("sucursal_id", observed=True)}, axis=1)
        return pd.concat([super().resumen_total(), self._con_nombres(por_suc)], axis=1)

    def ventas_por_sucursal(self) -> pd.DataFrame:
        t = (self.V.groupby(["sucursal_id", "sucursal", "ciudad", "pais"], observed=True)
             .agg(ventas=("total_price", "sum"), pizzas=("quantity", "sum"), pedidos=("nuevo_pedido", "sum"),
                  clientes=("cliente_id", "nunique"))
             .reset_index())
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        t["ticket promedio"] = t["ventas"] / t["pedidos"]
        t["venta por cliente"] = t["ventas"] / t["clientes"]
        t = t.sort_values("ventas", ascending=False).reset_index(drop=True)
        t.insert(0, "posición", range(1, len(t) + 1))
        return t

    def ventas_por_pais(self) -> pd.DataFrame:
        t = (self.V.groupby("pais", observed=True)
             .agg(sucursales=("sucursal_id", "nunique"), ventas=("total_price", "sum"), pedidos=("nuevo_pedido", "sum"))
             .sort_values("ventas", ascending=False))
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        return t

    def medidas_ticket_sucursal(self) -> pd.DataFrame:
        t = pd.concat({c: self.medidas(g["valor_pedido"])
                       for c, g in self.pedidos.groupby("sucursal_id", observed=True)}, axis=1)
        return self._con_nombres(t)

    def mix_sucursal(self, columna: str) -> pd.DataFrame:
        """% de las ventas de cada sucursal por categoría o tamaño."""
        t = pd.crosstab(self.V["sucursal_id"], self.V[columna], values=self.V["total_price"], aggfunc="sum",
                        normalize="index")
        t = t.loc[:, t.sum() > 0]
        t.columns = [f"% {c}" for c in t.columns]
        return t

    # ================================================================== tiempo por sucursal
    def ventas_diarias_sucursal(self) -> pd.DataFrame:
        return self.V.pivot_table(index="order_date", columns="sucursal_id", values="total_price",
                                  aggfunc="sum", observed=True)

    def mejores_dias_sucursal(self) -> pd.DataFrame:
        d = self.ventas_diarias_sucursal()
        return pd.DataFrame({"mejor día": d.idxmax().dt.strftime("%d/%m (%a)"), "ventas mejor día": d.max(),
                             "peor día": d.idxmin().dt.strftime("%d/%m (%a)"), "ventas peor día": d.min(),
                             "venta promedio diaria": d.mean(), "desv. estándar diaria (ventas)": d.std()})

    def dia_semana_sucursal(self) -> pd.DataFrame:
        """Venta PROMEDIO por día de la semana (no la suma: el mes tiene distinto número de cada día)."""
        diario = self.V.groupby(["sucursal_id", "order_date", "dia_semana"], observed=True)["total_price"].sum().reset_index()
        t = diario.pivot_table(index="sucursal_id", columns="dia_semana", values="total_price", aggfunc="mean",
                               observed=True)
        t.columns = t.columns.astype(str)
        return t

    def dia_semana_tabla(self) -> pd.DataFrame:
        t = pd.concat([self.dia_semana_sucursal(), self.dia_semana_empresa().to_frame("EMPRESA").T])
        t.columns = [f"ventas {c}" for c in t.columns]
        return t

    def hora_sucursal(self) -> pd.DataFrame:
        return self.V.pivot_table(index="sucursal_id", columns="hora", values="total_price", aggfunc="sum",
                                  observed=True).fillna(0)

    def hora_pico_sucursal(self) -> pd.DataFrame:
        h = self.hora_sucursal()
        col = lambda hs: [x for x in hs if x in h.columns]  # noqa: E731
        return pd.DataFrame({"hora pico": h.idxmax(axis=1),
                             "% ventas en hora pico": h.max(axis=1) / h.sum(axis=1),
                             "% ventas almuerzo (12-13 h)": h[col([12, 13])].sum(axis=1) / h.sum(axis=1),
                             "% ventas comida (17-19 h)": h[col([17, 18, 19])].sum(axis=1) / h.sum(axis=1)})

    def top_pizzas_sucursal(self, n: int = 3) -> pd.DataFrame:
        t = (self.V.groupby(["sucursal_id", "pizza_name"], observed=True)["total_price"].sum().reset_index()
             .sort_values("total_price", ascending=False).groupby("sucursal_id", observed=True).head(n))
        t["puesto"] = t.groupby("sucursal_id", observed=True).cumcount() + 1
        t = t.pivot(index="sucursal_id", columns="puesto", values="pizza_name")
        t.columns = [f"top {c}" for c in t.columns]
        return t

    # ================================================================== clientes
    def clientes_detalle(self) -> pd.DataFrame:
        t = (self.pedidos.groupby(["sucursal_id", "cliente_id"], observed=True)
             .agg(pedidos=("order_id", "count"), ventas=("valor_pedido", "sum"), pizzas=("pizzas", "sum"),
                  distancia_km=("distancia_km", "first"), primer_pedido=("order_date", "min"),
                  ultimo_pedido=("order_date", "max"))
             .reset_index()
             .merge(self.clientes[["cliente_id", "cliente_nombre", "cliente_barrio"]], on="cliente_id"))
        t["ticket promedio"] = t["ventas"] / t["pedidos"]
        t["segmento"] = pd.cut(t["pedidos"], bins=[0, 1, 3, np.inf],
                               labels=["Ocasional (1)", "Recurrente (2-3)", "Frecuente (4+)"])
        return t

    @staticmethod
    def pareto_20(d: pd.DataFrame) -> float:
        """% de ventas que aporta el 20 % de los mejores clientes."""
        v = d["ventas"].sort_values(ascending=False)
        return v.head(max(1, int(round(len(v) * 0.2)))).sum() / v.sum()

    def clientes_por_sucursal(self) -> pd.DataFrame:
        pc = self.clientes_detalle()
        t = pc.groupby("sucursal_id", observed=True).agg(
            clientes_activos=("cliente_id", "count"), pedidos=("pedidos", "sum"), ventas=("ventas", "sum"),
            pedidos_por_cliente=("pedidos", "mean"), venta_por_cliente=("ventas", "mean"),
            max_pedidos_un_cliente=("pedidos", "max"))
        t["clientes en catálogo"] = self.clientes.groupby("sucursal_id").size()
        t["% catálogo activo"] = t["clientes_activos"] / t["clientes en catálogo"]
        t["% ventas top 20 % clientes"] = pd.Series({c: self.pareto_20(g)
                                                     for c, g in pc.groupby("sucursal_id", observed=True)})
        t.index = t.index.astype(str)
        t.loc["EMPRESA"] = [pc["cliente_id"].nunique(), pc["pedidos"].sum(), pc["ventas"].sum(),
                            pc["pedidos"].mean(), pc["ventas"].mean(), pc["pedidos"].max(),
                            len(self.clientes), pc["cliente_id"].nunique() / len(self.clientes), self.pareto_20(pc)]
        for c in ["clientes_activos", "pedidos", "max_pedidos_un_cliente", "clientes en catálogo"]:
            t[c] = t[c].astype(int)
        return t

    def segmentos_clientes(self, medida: str = "clientes") -> pd.DataFrame:
        pc = self.clientes_detalle()
        if medida == "clientes":
            t = pc.pivot_table(index="sucursal_id", columns="segmento", values="cliente_id", aggfunc="count", observed=True)
        else:
            t = pc.pivot_table(index="sucursal_id", columns="segmento", values="ventas", aggfunc="sum", observed=True)
            t = t.div(t.sum(axis=1), axis=0)
            t.columns = [f"% {c}" for c in t.columns]
        return t

    def top_clientes(self, n: int = 10) -> pd.DataFrame:
        return self.clientes_detalle().sort_values("ventas", ascending=False).head(n)[
            ["sucursal_id", "cliente_id", "cliente_nombre", "cliente_barrio", "pedidos", "ventas",
             "ticket promedio", "distancia_km"]]

    # ================================================================== distancia
    def distancia_rangos(self, d: pd.DataFrame | None = None) -> pd.DataFrame:
        d = self.pedidos if d is None else d
        t = d.groupby("rango_distancia", observed=False).agg(
            pedidos=("order_id", "count"), clientes=("cliente_id", "nunique"), ventas=("valor_pedido", "sum"),
            ticket_promedio=("valor_pedido", "mean"), pizzas_por_pedido=("pizzas", "mean"))
        t["% pedidos"] = t["pedidos"] / t["pedidos"].sum()
        t["% ventas"] = t["ventas"] / t["ventas"].sum()
        t["% acumulado ventas"] = t["% ventas"].cumsum()
        t.index = t.index.astype(str)
        return t

    def distancia_sucursal(self) -> pd.DataFrame:
        t = pd.concat({c: self.medidas(g["distancia_km"]) for c, g in self.pedidos.groupby("sucursal_id", observed=True)},
                      axis=1).T[["Media", "Mediana", "Mínimo", "Máximo", "Desviación estándar", "Q1 (25 %)", "Q3 (75 %)"]]
        t["corr. distancia vs ticket"] = pd.Series({c: g["distancia_km"].corr(g["valor_pedido"])
                                                    for c, g in self.pedidos.groupby("sucursal_id", observed=True)})
        return t

    def distancia_rango_sucursal(self) -> pd.DataFrame:
        t = self.pedidos.pivot_table(index="sucursal_id", columns="rango_distancia", values="valor_pedido",
                                     aggfunc="sum", observed=False).fillna(0)
        t = t.div(t.sum(axis=1), axis=0)
        t.columns = [f"% {c}" for c in t.columns]
        return t

    def barrios(self) -> pd.DataFrame:
        return (self.pedidos.merge(self.clientes[["cliente_id", "cliente_barrio"]], on="cliente_id")
                .groupby(["sucursal_id", "cliente_barrio"], observed=True)
                .agg(pedidos=("order_id", "count"), ventas=("valor_pedido", "sum"),
                     distancia_media=("distancia_km", "mean"))
                .reset_index().sort_values(["sucursal_id", "ventas"], ascending=[True, False]))

    def barrio_lider_sucursal(self) -> pd.DataFrame:
        return self.barrios().groupby("sucursal_id", observed=True).head(1).set_index("sucursal_id")

    # ================================================================== conclusiones y exportación
    def hallazgos(self) -> list[str]:
        V, ps, pp = self.V, self.ventas_por_sucursal(), self.ventas_por_pais()
        dias, dsem, hora = self.ventas_por_dia(), self.dia_semana_empresa(), self.ventas_por_hora()
        rk, pc, dr = self.ranking_pizzas(), self.clientes_detalle(), self.distancia_rangos()
        seg = pc["segmento"].value_counts(normalize=True)
        top, ult = ps.iloc[0], ps.iloc[-1]
        mejor = dias["ventas"].idxmax()
        rango_top = dr["ventas"].idxmax()
        corr = self.pedidos["distancia_km"].corr(self.pedidos["valor_pedido"])
        fuerza = "prácticamente nula" if abs(corr) < 0.1 else "débil" if abs(corr) < 0.3 else "moderada"
        return [
            f"La empresa vendió ${V['total_price'].sum():,.2f} en el mes: {V['quantity'].sum():,} pizzas en "
            f"{V['order_id'].nunique():,} pedidos de {V['cliente_id'].nunique():,} clientes "
            f"(ticket promedio ${V['total_price'].sum() / V['nuevo_pedido'].sum():.2f}).",
            f"{top['ciudad']} lidera con ${top['ventas']:,.2f} ({top['% ventas']:.1%} del total); "
            f"{ult['ciudad']} es la de menor venta ({ult['% ventas']:.1%}).",
            "Por país: " + "; ".join(f"{p} {r['% ventas']:.1%}" for p, r in pp.iterrows()) + ".",
            f"El día de mayor venta fue el {mejor:%d/%m} ({dias.loc[mejor, 'día']}); el día de la semana más fuerte "
            f"en promedio es el {dsem.idxmax()} y el más flojo el {dsem.idxmin()}.",
            f"La hora pico de la empresa es las {hora['ventas'].idxmax()}:00; el negocio tiene dos picos (almuerzo y comida).",
            f"{rk.index[0]} es la pizza que más vende; {self.pizzas_para_80()} de {len(rk)} pizzas explican el 80 % de las ventas.",
            f"Clientes: {seg.get('Ocasional (1)', 0):.0%} pidió una sola vez, {seg.get('Recurrente (2-3)', 0):.0%} entre 2 y 3 "
            f"veces y {seg.get('Frecuente (4+)', 0):.0%} cuatro o más; el 20 % de mejores clientes aporta el "
            f"{self.pareto_20(pc):.0%} de las ventas.",
            f"Distancia: el pedido promedio viaja {self.pedidos['distancia_km'].mean():.1f} km; el rango {rango_top} "
            f"concentra más ventas ({dr.loc[rango_top, '% ventas']:.0%}) y la correlación distancia-ticket es "
            f"{corr:+.2f} ({fuerza}).",
        ]

    def prompt_resumen_ejecutivo(self) -> str:
        ps = self.ventas_por_sucursal()
        contexto = "\n\n".join([
            "Resumen por sucursal:\n" + ps[["sucursal_id", "ciudad", "pais", "ventas", "% ventas", "pedidos",
                                            "clientes", "ticket promedio"]].round(3).to_string(index=False),
            "Venta promedio por día de la semana:\n" + self.dia_semana_sucursal().round(0).to_string(),
            "Hora pico por sucursal:\n" + self.hora_pico_sucursal().round(3).to_string(),
            "Clientes por sucursal:\n" + self.clientes_por_sucursal().round(3).to_string(),
            "Ventas por rango de distancia (empresa):\n" + self.distancia_rangos().round(3).to_string(),
            "Hallazgos:\n" + "\n".join(self.hallazgos()),
        ])
        return ("Eres consultor de analítica para una cadena de pizzerías. Con estas tablas escribe en español: "
                "(1) un resumen ejecutivo de 5 viñetas con cifras, (2) 5 decisiones concretas para la gerencia "
                "(por sucursal, horario, clientes y zona de domicilio). No inventes datos que no estén en las tablas.\n\n"
                + contexto)

    def tablas_exportables(self) -> dict[str, pd.DataFrame]:
        base = super().tablas_exportables()
        base.pop("Datos")
        return {
            "Datos_enriquecidos": self.V.assign(order_time=self.V["order_time"].astype(str)),
            "Sucursales": self.sucursales,
            "Clientes": self.clientes.drop(columns="peso_frecuencia", errors="ignore"),
            **base,
            "Ventas_sucursal": self.ventas_por_sucursal(),
            "Ventas_pais": self.ventas_por_pais().reset_index(),
            "Ticket_sucursal": self.medidas_ticket_sucursal().reset_index(names="medida"),
            "Mix_categoria_suc": self.mix_sucursal("pizza_category").reset_index(),
            "Mix_tamano_suc": self.mix_sucursal("pizza_size").reset_index(),
            "Ventas_dia_suc": self.ventas_diarias_sucursal().reset_index(),
            "Mejores_dias_suc": self.mejores_dias_sucursal().reset_index(),
            "Dia_semana_suc": self.dia_semana_tabla().reset_index(names="sucursal_id"),
            "Hora_pico_suc": self.hora_pico_sucursal().reset_index(),
            "Top3_suc": self.top_pizzas_sucursal().reset_index(),
            "Clientes_suc": self.clientes_por_sucursal().reset_index(names="sucursal_id"),
            "Clientes_detalle": self.clientes_detalle(),
            "Distancia_rangos": self.distancia_rangos().reset_index(),
            "Distancia_suc": self.distancia_sucursal().reset_index(names="sucursal_id"),
            "Barrios": self.barrios(),
        }
