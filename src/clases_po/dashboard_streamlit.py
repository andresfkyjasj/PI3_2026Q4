"""Dashboard web de la pizzería multisede con Streamlit + Plotly.

La clase :class:`DashboardStreamlit` toma el dataset enriquecido de la Semana 5 y:

1. calcula los indicadores reutilizando :class:`AnalisisPizzeria` (mismas cifras que el notebook);
2. arma cada visual como una figura de Plotly (se puede ver en Jupyter o en la web);
3. redacta la respuesta a la pregunta de negocio con cifras calculadas, no escritas a mano;
4. compone todo en una app de Streamlit con filtros (``render()``), lista para publicarse
   en Streamlit Community Cloud desde GitHub.

Uso en Jupyter::

    dash = DashboardStreamlit.desde_csv("datos/pizzeria_enero_2015_enriquecido.csv")
    dash.kpis()
    dash.figura_ventas_sucursal()

Uso en la app (``src/dashboard/app.py``)::

    DashboardStreamlit.desde_csv(RUTA_DATOS).render()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:  # dentro del paquete clases_po
    from .analisis import AnalisisPizzeria
except ImportError:  # cuando la app agrega src/clases_po al path y lo importa como módulo suelto
    from analisis import AnalisisPizzeria


DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
ORDEN_TAMANOS = ["S", "M", "L", "XL", "XXL"]
ORDEN_CATEGORIAS = ["Classic", "Supreme", "Veggie", "Chicken"]


def dinero(x: float, decimales: int = 0) -> str:
    """$340.436 · formato colombiano (punto de miles, coma decimal)."""
    s = f"{x:,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return f"${s}"


def numero(x: float, decimales: int = 0) -> str:
    return f"{x:,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float, decimales: int = 1) -> str:
    return f"{x * 100:.{decimales}f} %".replace(".", ",")


class DashboardStreamlit:
    """Dashboard de ventas de la pizzería: figuras Plotly + app Streamlit.

    Parámetros
    ----------
    ventas : DataFrame enriquecido (una fila por línea de pedido) con las columnas de la Semana 5.
    titulo : título de la app.
    """

    # Un color fijo por sucursal (identidad), siempre en el mismo orden; nunca por posición en el ranking
    PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    COLOR = "#2a78d6"
    ACENTO = "#eb6834"
    TEXTO, TEXTO_2, REJILLA = "#1f1f1e", "#5f5e5a", "#e6e5e0"
    ESCALA = "Blues"

    PREGUNTA_NEGOCIO = (
        "¿Dónde, cuándo, qué y a quién le vende la empresa en enero de 2015, y en qué debe "
        "enfocar sus recursos: sucursales, horarios, menú, clientes y zona de reparto?"
    )
    PREGUNTAS = {
        "P1": "¿Qué sucursal y qué país aportan más a las ventas?",
        "P2": "¿Cuándo se concentra la demanda (día de la semana y hora)?",
        "P3": "¿Qué se vende y cambia el menú de una sede a otra?",
        "P4": "¿Quiénes son los clientes y qué tan fieles son?",
        "P5": "¿Hasta dónde llegan los pedidos y la distancia cambia lo que se compra?",
    }

    COLUMNAS_REQUERIDAS = ["order_id", "pizza_id", "order_date", "total_price", "quantity", "unit_price",
                           "pizza_size", "pizza_category", "pizza_name", "dia_semana", "hora", "nuevo_pedido",
                           "sucursal_id", "sucursal", "ciudad", "pais", "sucursal_latitud", "sucursal_longitud",
                           "cliente_id", "cliente_nombre", "cliente_barrio", "cliente_latitud", "cliente_longitud",
                           "distancia_km"]

    def __init__(self, ventas: pd.DataFrame, titulo: str = "Pizzería multisede · Ventas de enero 2015",
                 pizza_elegida: str = "The Pepperoni Pizza"):
        faltan = [c for c in self.COLUMNAS_REQUERIDAS if c not in ventas.columns]
        if faltan:
            raise ValueError(f"Al dataset le faltan columnas del enriquecimiento: {faltan}")
        self.titulo = titulo
        self.pizza_elegida = pizza_elegida
        self.V_total = self.tipar(ventas)
        self.orden_suc = list(pd.unique(self.V_total.sort_values("sucursal_id")["sucursal_id"].astype(str)))
        self.color_suc = {s: self.PALETA[i % len(self.PALETA)] for i, s in enumerate(self.orden_suc)}
        self._preparar(self.V_total)

    # ================================================================== datos
    @classmethod
    def desde_csv(cls, ruta: str | Path, **kwargs) -> "DashboardStreamlit":
        """Carga el CSV enriquecido de la Semana 5."""
        return cls(pd.read_csv(ruta), **kwargs)

    @staticmethod
    def exportar_datos(origen: str | Path, destino: str | Path,
                       excluir=("pizza_ingredients", "total_calculado", "verificacion", "pizza_name_id")) -> Path:
        """Copia el CSV enriquecido a la carpeta de la app sin las columnas que el dashboard no usa.

        La app lee su propia copia (``src/dashboard/datos``) porque la GitHub Action de la Semana 1
        limpia ``src/sincronicas/datos`` en cada push.
        """
        d = pd.read_csv(origen)
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        d.drop(columns=[c for c in excluir if c in d.columns]).to_csv(destino, index=False)
        print(f"{len(d):,} filas → {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
        return destino

    @staticmethod
    def tipar(df: pd.DataFrame) -> pd.DataFrame:
        """El CSV pierde los tipos de pandas: se vuelven a poner (mismo criterio de PreprocesadorPizzas)."""
        d = df.copy()
        d["order_date"] = pd.to_datetime(d["order_date"])
        if "fecha_hora" in d.columns:
            d["fecha_hora"] = pd.to_datetime(d["fecha_hora"])
        d["pizza_size"] = pd.Categorical(d["pizza_size"], categories=ORDEN_TAMANOS, ordered=True)
        d["pizza_category"] = pd.Categorical(d["pizza_category"], categories=ORDEN_CATEGORIAS)
        d["dia_semana"] = pd.Categorical(d["order_date"].dt.dayofweek.map(dict(enumerate(DIAS_SEMANA))),
                                         categories=DIAS_SEMANA, ordered=True)
        orden = [s for s in ["MED", "BOG", "CAL", "UIO", "PTY"] if s in set(d["sucursal_id"])]
        orden += sorted(set(d["sucursal_id"]) - set(orden))
        d["sucursal_id"] = pd.Categorical(d["sucursal_id"], categories=orden)
        for c in ["quantity", "hora", "nuevo_pedido"]:
            d[c] = d[c].astype("int64")
        return d

    def _preparar(self, V: pd.DataFrame) -> None:
        """Crea el objeto de análisis (AnalisisPizzeria) para los datos filtrados."""
        self.V = V
        suc_cols = ["sucursal_id", "sucursal", "ciudad", "pais", "sucursal_direccion",
                    "sucursal_latitud", "sucursal_longitud"]
        suc_cols = [c for c in suc_cols if c in V.columns]
        self.sucursales = (V[suc_cols].drop_duplicates("sucursal_id").assign(
            sucursal_id=lambda x: x["sucursal_id"].astype(str))
            .set_index("sucursal_id").reindex([s for s in self.orden_suc if s in set(V["sucursal_id"].astype(str))])
            .reset_index())
        cli_cols = ["cliente_id", "sucursal_id", "cliente_nombre", "cliente_barrio",
                    "cliente_latitud", "cliente_longitud", "distancia_km"]
        if "cliente_direccion" in V.columns:
            cli_cols.insert(3, "cliente_direccion")
        self.clientes = V[cli_cols].drop_duplicates("cliente_id").assign(
            sucursal_id=lambda x: x["sucursal_id"].astype(str))
        self.A = AnalisisPizzeria(V, self.sucursales, self.clientes, pizza_elegida=self.pizza_elegida) if len(V) else None

    def filtrar(self, sucursales=None, categorias=None, tamanos=None, fechas=None) -> "DashboardStreamlit":
        """Devuelve un dashboard nuevo con los datos filtrados (los colores por sucursal se conservan)."""
        V = self.V_total
        if sucursales:
            V = V[V["sucursal_id"].astype(str).isin(sucursales)]
        if categorias:
            V = V[V["pizza_category"].astype(str).isin(categorias)]
        if tamanos:
            V = V[V["pizza_size"].astype(str).isin(tamanos)]
        if fechas and len(fechas) == 2:
            ini, fin = pd.Timestamp(fechas[0]), pd.Timestamp(fechas[1])
            V = V[(V["order_date"] >= ini) & (V["order_date"] <= fin)]
        nuevo = object.__new__(DashboardStreamlit)
        nuevo.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("V", "A", "sucursales", "clientes")})
        nuevo.V_total = self.V_total
        nuevo._preparar(V.copy())
        return nuevo

    @property
    def vacio(self) -> bool:
        return self.A is None or self.V.empty

    def periodo(self) -> str:
        f = self.V["order_date"]
        return f"{f.min():%d/%m/%Y} a {f.max():%d/%m/%Y}"

    # ================================================================== indicadores
    def kpis(self) -> dict:
        """Indicadores clave (los mismos de la Semana 5, ahora para la vista filtrada)."""
        V, P = self.V, self.A.pedidos
        ventas, pedidos = V["total_price"].sum(), V["order_id"].nunique()
        return {
            "Ventas": ventas,
            "Pedidos": pedidos,
            "Ticket promedio": ventas / pedidos,
            "Clientes": V["cliente_id"].nunique(),
            "Pizzas por pedido": V["quantity"].sum() / pedidos,
            "Distancia media (km)": P["distancia_km"].mean(),
            "% de las ventas totales": ventas / self.V_total["total_price"].sum(),
        }

    def kpis_tabla(self) -> pd.DataFrame:
        k = self.kpis()
        fmt = {"Ventas": dinero(k["Ventas"]), "Pedidos": numero(k["Pedidos"]),
               "Ticket promedio": dinero(k["Ticket promedio"], 2), "Clientes": numero(k["Clientes"]),
               "Pizzas por pedido": numero(k["Pizzas por pedido"], 2),
               "Distancia media (km)": numero(k["Distancia media (km)"], 1),
               "% de las ventas totales": pct(k["% de las ventas totales"])}
        return pd.DataFrame({"indicador": fmt.keys(), "valor": fmt.values()})

    # ================================================================== respuesta de negocio
    def respuestas(self) -> list[dict]:
        """Pregunta → respuesta (con cifras) → visual que la muestra → decisión sugerida."""
        A = self.A
        ps = A.ventas_por_sucursal()
        pp = A.ventas_por_pais()
        top, ult = ps.iloc[0], ps.iloc[-1]
        dsem = A.dia_semana_empresa()
        hora = A.ventas_por_hora()
        h_pico = int(hora["ventas"].idxmax())
        pc = A.clientes_detalle()
        seg_cli = pc["segmento"].value_counts(normalize=True)
        seg_ven = pc.groupby("segmento", observed=False)["ventas"].sum() / pc["ventas"].sum()
        dr = A.distancia_rangos()
        bajo6 = dr.loc[[r for r in ["0-2 km", "2-4 km", "4-6 km"] if r in dr.index], "% ventas"].sum()
        corr = A.pedidos["distancia_km"].corr(A.pedidos["valor_pedido"])
        rk = A.ranking_pizzas()
        mix = A.mix_sucursal("pizza_category") if len(self.sucursales) > 1 else None

        r = []
        texto_pais = "; ".join(f"{p} {pct(v)}" for p, v in pp["% ventas"].items())
        r.append({
            "id": "P1", "pregunta": self.PREGUNTAS["P1"],
            "respuesta": (f"{top['ciudad']} lidera con {dinero(top['ventas'])} ({pct(top['% ventas'])}); "
                          f"{ult['ciudad']} es la de menor venta ({pct(ult['% ventas'])}). Por país: {texto_pais}."),
            "visual": "Barras horizontales ordenadas de mayor a menor, un color por sucursal.",
            "decision": (f"Priorizar inventario y pauta en {top['ciudad']}; revisar el plan comercial de "
                         f"{ult['ciudad']}, cuyo ticket ({dinero(ult['ticket promedio'], 2)}) es similar: "
                         "su brecha es de volumen de pedidos, no de gasto por pedido."),
        })
        r.append({
            "id": "P2", "pregunta": self.PREGUNTAS["P2"],
            "respuesta": (f"El {dsem.idxmax()} es el día más fuerte ({dinero(dsem.max())} promedio) y el "
                          f"{dsem.idxmin()} el más flojo ({dinero(dsem.min())}). La hora pico es las {h_pico}:00; "
                          "hay dos picos: almuerzo (12-13 h) y comida (17-19 h)."),
            "visual": "Mapa de calor día × sucursal y columnas por hora.",
            "decision": (f"Reforzar personal y repartidores el {dsem.idxmax()} y en las franjas 12-13 h y 17-19 h; "
                         f"usar promociones el {dsem.idxmin()} para mover la demanda."),
        })
        r.append({
            "id": "P3", "pregunta": self.PREGUNTAS["P3"],
            "respuesta": (f"{rk.index[0]} es la pizza que más vende; {A.pizzas_para_80()} de {len(rk)} pizzas "
                          "explican el 80 % de las ventas. "
                          + ("Perfil por sede: " + "; ".join(
                              f"{self._ciudad(s)} → {fila.idxmax().replace('% ', '')} ({pct(fila.max())})"
                              for s, fila in mix.iterrows()) + "." if mix is not None else "")),
            "visual": "Mapa de calor % de ventas por categoría en cada sede y top 10 pizzas.",
            "decision": "Inventario y promociones diferenciados por sede según su categoría dominante; "
                        "revisar las pizzas fuera del 80 % antes de recortar el menú.",
        })
        r.append({
            "id": "P4", "pregunta": self.PREGUNTAS["P4"],
            "respuesta": (f"{numero(len(pc))} clientes con {numero(pc['pedidos'].mean(), 2)} pedidos en promedio. "
                          f"Los frecuentes (4+ pedidos) son el {pct(seg_cli.get('Frecuente (4+)', 0), 0)} de los "
                          f"clientes y generan el {pct(seg_ven.get('Frecuente (4+)', 0), 0)} de las ventas; "
                          f"el 20 % de mejores clientes aporta el {pct(A.pareto_20(pc), 0)}."),
            "visual": "Columnas de clientes por segmento y curva de Pareto por sucursal.",
            "decision": "Programa de fidelización para los frecuentes y campaña de segunda compra para "
                        f"el {pct(seg_cli.get('Ocasional (1)', 0), 0)} que pidió una sola vez.",
        })
        r.append({
            "id": "P5", "pregunta": self.PREGUNTAS["P5"],
            "respuesta": (f"El {pct(bajo6, 0)} de las ventas llega de menos de 6 km y el pedido promedio viaja "
                          f"{numero(A.pedidos['distancia_km'].mean(), 1)} km. La correlación distancia-ticket es "
                          f"{numero(corr, 2)}: el cliente lejano gasta casi lo mismo, pero pide menos."),
            "visual": "Columnas por rango de distancia y mapa de clientes alrededor de cada sede.",
            "decision": "Definir el radio de reparto en 6 km y cobrar domicilio por distancia fuera de él; "
                        "la distancia no justifica precios distintos del menú.",
        })
        return r

    def respuestas_tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.respuestas()).set_index("id")

    def _ciudad(self, suc_id: str) -> str:
        s = self.sucursales.set_index("sucursal_id")
        return s.loc[suc_id, "ciudad"] if suc_id in s.index else suc_id

    # ================================================================== estilo de las figuras
    def _estilo(self, fig: go.Figure, titulo: str = "", alto: int = 380, leyenda: bool = False) -> go.Figure:
        fig.update_layout(
            title={"text": titulo, "x": 0, "xanchor": "left", "font": {"size": 15, "color": self.TEXTO}},
            height=alto, margin={"l": 10, "r": 10, "t": 50 if titulo else 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": self.TEXTO_2},
            separators=",.", showlegend=leyenda,
            legend={"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0, "title": None},
            hoverlabel={"bgcolor": "white", "font_size": 12},
        )
        fig.update_xaxes(showgrid=False, linecolor=self.REJILLA, ticks="", title_text=None)
        fig.update_yaxes(gridcolor=self.REJILLA, zeroline=False, linecolor=self.REJILLA, title_text=None,
                         tickformat=",.0f")
        if leyenda:
            fig.update_layout(margin={"b": 60})
        return fig

    # ================================================================== P1 · sucursales
    def figura_ventas_sucursal(self) -> go.Figure:
        d = self.A.ventas_por_sucursal().sort_values("ventas")
        d["etiqueta"] = [f"{dinero(v)} · {pct(p)}" for v, p in zip(d["ventas"], d["% ventas"])]
        fig = go.Figure(go.Bar(
            x=d["ventas"], y=d["ciudad"], orientation="h", text=d["etiqueta"], textposition="outside",
            marker={"color": [self.color_suc[s] for s in d["sucursal_id"]], "cornerradius": 4},
            customdata=np.stack([d["pedidos"], d["ticket promedio"], d["clientes"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<br>Pedidos: %{customdata[0]:,}"
                          "<br>Ticket: $%{customdata[1]:,.2f}<br>Clientes: %{customdata[2]:,}<extra></extra>",
            cliponaxis=False, width=0.6))
        fig.update_xaxes(range=[0, d["ventas"].max() * 1.3], tickprefix="$", showticklabels=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        return self._estilo(fig, "P1 · Ventas del mes por sucursal", alto=60 * len(d) + 90)

    def figura_ventas_pais(self) -> go.Figure:
        d = self.A.ventas_por_pais().reset_index().sort_values("ventas")
        fig = go.Figure(go.Bar(x=d["ventas"], y=d["pais"], orientation="h", width=0.55,
                               text=[pct(p) for p in d["% ventas"]], textposition="outside", cliponaxis=False,
                               marker={"color": self.COLOR, "cornerradius": 4},
                               hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, d["ventas"].max() * 1.2], showticklabels=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        return self._estilo(fig, "Participación por país", alto=60 * len(d) + 90)

    # ================================================================== P2 · cuándo
    def figura_ventas_diarias(self) -> go.Figure:
        diario = self.A.ventas_diarias_sucursal()
        fig = go.Figure()
        for s in diario.columns.astype(str):
            fig.add_trace(go.Scatter(x=diario.index, y=diario[s], mode="lines", name=self._ciudad(s),
                                     line={"color": self.color_suc[s], "width": 2},
                                     hovertemplate=f"<b>{self._ciudad(s)}</b> · %{{x|%d/%m}}<br>$%{{y:,.0f}}<extra></extra>"))
        fig.update_yaxes(tickprefix="$", rangemode="tozero")
        fig.update_xaxes(tickformat="%d/%m")
        return self._estilo(fig, "Ventas diarias por sucursal", alto=360, leyenda=True)

    def figura_dia_semana(self) -> go.Figure:
        t = self.A.dia_semana_sucursal()
        y = [self._ciudad(s) for s in t.index.astype(str)]
        fig = go.Figure(go.Heatmap(
            z=t.values, x=list(t.columns), y=y, colorscale=self.ESCALA, xgap=2, ygap=2,
            text=[[dinero(v) for v in fila] for fila in t.values], texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> · %{x}<br>Venta promedio: $%{z:,.0f}<extra></extra>",
            colorbar={"tickprefix": "$", "tickformat": ",.0f", "thickness": 10, "len": 0.8}))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        return self._estilo(fig, "P2 · Venta promedio por día de la semana", alto=60 * len(y) + 110)

    def figura_hora(self) -> go.Figure:
        h = self.A.ventas_por_hora().reset_index()
        pico = h["ventas"].idxmax()
        colores = [self.ACENTO if i == pico else self.COLOR for i in range(len(h))]
        fig = go.Figure(go.Bar(x=h["hora"], y=h["ventas"], marker={"color": colores, "cornerradius": 4},
                               customdata=h["% ventas"],
                               hovertemplate="%{x}:00<br>Ventas: $%{y:,.0f}<br>%{customdata:.1%} del total<extra></extra>"))
        for x0, x1, txt in [(11.5, 13.5, "almuerzo"), (16.5, 19.5, "comida")]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=self.REJILLA, opacity=0.45, line_width=0, layer="below",
                          annotation_text=txt, annotation_position="top left",
                          annotation_font={"size": 11, "color": self.TEXTO_2})
        fig.update_xaxes(tickmode="linear", ticksuffix="h")
        fig.update_yaxes(tickprefix="$")
        return self._estilo(fig, f"Ventas por hora · pico a las {int(h.loc[pico, 'hora'])}:00", alto=360)

    # ================================================================== P3 · qué
    def figura_mix_categoria(self) -> go.Figure:
        t = self.A.mix_sucursal("pizza_category")
        y = [self._ciudad(s) for s in t.index.astype(str)]
        x = [c.replace("% ", "") for c in t.columns]
        fig = go.Figure(go.Heatmap(
            z=t.values, x=x, y=y, colorscale=self.ESCALA, xgap=2, ygap=2,
            text=[[pct(v) for v in fila] for fila in t.values], texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.1%} de sus ventas<extra></extra>",
            colorbar={"tickformat": ".0%", "thickness": 10, "len": 0.8}))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        return self._estilo(fig, "P3 · % de las ventas de cada sede por categoría", alto=60 * len(y) + 110)

    def figura_top_pizzas(self, n: int = 10) -> go.Figure:
        rk = self.A.ranking_pizzas().head(n).iloc[::-1]
        nombres = [s.replace("The ", "").replace(" Pizza", "") for s in rk.index]
        fig = go.Figure(go.Bar(x=rk["ventas"], y=nombres, orientation="h", width=0.6,
                               marker={"color": self.COLOR, "cornerradius": 4},
                               customdata=np.stack([rk["pizzas"], rk["% ventas"]], axis=-1),
                               hovertemplate="<b>%{y}</b><br>Ventas: $%{x:,.0f}<br>Pizzas: %{customdata[0]:,}"
                                             "<br>%{customdata[1]:.1%} del total<extra></extra>"))
        fig.update_xaxes(tickprefix="$", showgrid=True, gridcolor=self.REJILLA)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
        return self._estilo(fig, f"Top {n} pizzas por ventas", alto=32 * n + 90)

    def figura_tamanos(self) -> go.Figure:
        t = self.A.frecuencias("pizza_size").drop("Total")
        t = t[t["pizzas"] > 0]
        fig = go.Figure(go.Bar(x=t.index, y=t["pizzas"], width=0.6, marker={"color": self.COLOR, "cornerradius": 4},
                               text=[pct(p) for p in t["% ventas"]], textposition="outside", cliponaxis=False,
                               hovertemplate="Tamaño %{x}<br>%{y:,} pizzas<extra></extra>"))
        fig.update_yaxes(range=[0, t["pizzas"].max() * 1.18])
        return self._estilo(fig, "Pizzas por tamaño (texto: % de ventas)", alto=340)

    # ================================================================== P4 · clientes
    def figura_segmentos(self) -> go.Figure:
        pc = self.A.clientes_detalle()
        cli = pc["segmento"].value_counts(normalize=True).reindex(pc["segmento"].cat.categories)
        ven = (pc.groupby("segmento", observed=False)["ventas"].sum() / pc["ventas"].sum()).reindex(cli.index)
        fig = go.Figure([
            go.Bar(name="% de clientes", x=cli.index.astype(str), y=cli.values, marker_color="#9ec5f0",
                   marker_cornerradius=4, text=[pct(v, 0) for v in cli.values], textposition="outside",
                   hovertemplate="%{x}<br>%{y:.1%} de los clientes<extra></extra>"),
            go.Bar(name="% de ventas", x=ven.index.astype(str), y=ven.values, marker_color=self.COLOR,
                   marker_cornerradius=4, text=[pct(v, 0) for v in ven.values], textposition="outside",
                   hovertemplate="%{x}<br>%{y:.1%} de las ventas<extra></extra>"),
        ])
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig = self._estilo(fig, "P4 · Clientes vs. ventas por segmento de frecuencia", alto=380, leyenda=True)
        fig.update_yaxes(tickformat=".0%", range=[0, max(cli.max(), ven.max()) * 1.2])
        return fig

    def figura_pareto_clientes(self) -> go.Figure:
        pc = self.A.clientes_detalle()
        fig = go.Figure()
        for s in self.sucursales["sucursal_id"]:
            v = pc.loc[pc["sucursal_id"].astype(str) == s, "ventas"].sort_values(ascending=False).to_numpy()
            if not len(v):
                continue
            fig.add_trace(go.Scatter(x=np.arange(1, len(v) + 1) / len(v), y=v.cumsum() / v.sum(), mode="lines",
                                     name=self._ciudad(s), line={"color": self.color_suc[s], "width": 2},
                                     hovertemplate=f"<b>{self._ciudad(s)}</b><br>%{{x:.0%}} de clientes → "
                                                   "%{y:.0%} de ventas<extra></extra>"))
        fig.add_vline(x=0.2, line_dash="dot", line_color=self.TEXTO_2, line_width=1,
                      annotation_text="20 % de clientes", annotation_position="bottom right",
                      annotation_font={"size": 11, "color": self.TEXTO_2})
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=self.REJILLA)
        fig = self._estilo(fig, "Curva de Pareto de clientes", alto=380, leyenda=True)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        return fig

    # ================================================================== P5 · distancia
    def figura_distancia(self) -> go.Figure:
        dr = self.A.distancia_rangos()
        fig = go.Figure(go.Bar(
            x=dr.index, y=dr["ventas"], width=0.6, marker={"color": self.COLOR, "cornerradius": 4},
            text=[pct(v, 0) for v in dr["% ventas"]], textposition="outside", cliponaxis=False,
            customdata=np.stack([dr["pedidos"], dr["ticket_promedio"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<br>Pedidos: %{customdata[0]:,}"
                          "<br>Ticket: $%{customdata[1]:,.2f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$", range=[0, dr["ventas"].max() * 1.18])
        return self._estilo(fig, "P5 · Ventas por distancia sucursal → cliente", alto=360)

    def figura_ticket_distancia(self) -> go.Figure:
        dr = self.A.distancia_rangos()
        fig = go.Figure(go.Bar(x=dr.index, y=dr["ticket_promedio"], width=0.6,
                               marker={"color": "#9ec5f0", "cornerradius": 4},
                               text=[dinero(v, 2) for v in dr["ticket_promedio"]], textposition="outside",
                               cliponaxis=False, hovertemplate="<b>%{x}</b><br>Ticket: $%{y:,.2f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$", range=[0, dr["ticket_promedio"].max() * 1.2])
        return self._estilo(fig, "Ticket promedio por distancia (casi plano)", alto=360)

    def figura_mapa(self, sucursal: str | None = None) -> go.Figure:
        """Clientes alrededor de una sede: color = distancia, tamaño = pedidos; la sede en naranja."""
        pc = self.A.clientes_detalle().merge(self.clientes[["cliente_id", "cliente_latitud", "cliente_longitud"]],
                                             on="cliente_id")
        suc = self.sucursales
        if sucursal:
            pc = pc[pc["sucursal_id"].astype(str) == sucursal]
            suc = suc[suc["sucursal_id"] == sucursal]
        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat=pc["cliente_latitud"], lon=pc["cliente_longitud"], mode="markers", name="Clientes",
            marker={"size": np.clip(5 + pc["pedidos"] * 1.6, 5, 26), "color": pc["distancia_km"],
                    "colorscale": self.ESCALA, "cmin": 0, "cmax": 10, "opacity": 0.8,
                    "colorbar": {"title": {"text": "km"}, "thickness": 10, "len": 0.7}},
            customdata=np.stack([pc["cliente_nombre"], pc["cliente_barrio"], pc["pedidos"],
                                 pc["ventas"], pc["distancia_km"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Pedidos: %{customdata[2]}"
                          "<br>Ventas: $%{customdata[3]:,.0f}<br>Distancia: %{customdata[4]:.1f} km<extra></extra>"))
        fig.add_trace(go.Scattermap(
            lat=suc["sucursal_latitud"], lon=suc["sucursal_longitud"], mode="markers+text", name="Sucursal",
            marker={"size": 18, "color": self.ACENTO}, text=suc["sucursal"], textposition="top right",
            textfont={"size": 12, "color": self.TEXTO},
            hovertemplate="<b>%{text}</b><extra></extra>"))
        lat0, lon0 = suc["sucursal_latitud"].mean(), suc["sucursal_longitud"].mean()
        zoom = 11 if len(suc) == 1 else 3.6
        fig.update_layout(map={"style": "carto-positron", "center": {"lat": lat0, "lon": lon0}, "zoom": zoom})
        fig = self._estilo(fig, "Clientes alrededor de la sede (tamaño = pedidos, color = distancia)", alto=470)
        fig.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0})
        return fig

    # ================================================================== Streamlit
    def render(self) -> None:
        """Arma la app completa. Se llama desde ``src/dashboard/app.py`` con ``streamlit run``."""
        import streamlit as st

        st.set_page_config(page_title="Pizzería · Dashboard", page_icon="🍕", layout="wide")
        st.markdown(_CSS, unsafe_allow_html=True)

        # ---------------- filtros (barra lateral)
        base = self.V_total
        with st.sidebar:
            st.header("Filtros")
            nombres = {s: f"{s} · {self._ciudad_total(s)}" for s in self.orden_suc}
            sel_suc = st.multiselect("Sucursales", self.orden_suc, default=self.orden_suc,
                                     format_func=lambda s: nombres[s])
            sel_cat = st.multiselect("Categoría", ORDEN_CATEGORIAS, default=ORDEN_CATEGORIAS)
            tams = [t for t in ORDEN_TAMANOS if t in set(base["pizza_size"].astype(str))]
            sel_tam = st.multiselect("Tamaño", tams, default=tams)
            fmin, fmax = base["order_date"].min().date(), base["order_date"].max().date()
            sel_fec = st.date_input("Rango de fechas", (fmin, fmax), min_value=fmin, max_value=fmax,
                                    format="DD/MM/YYYY")
            st.divider()
            st.caption("Datos: Bizwhiznetwork/pizza_sales (Hugging Face), enriquecidos en la Semana 5. "
                       "Medellín = pedidos reales de enero 2015; las otras sedes son simuladas a partir "
                       "de pedidos reales (columna `origen`).")

        d = self.filtrar(sel_suc, sel_cat, sel_tam, sel_fec if isinstance(sel_fec, tuple) else None)

        # ---------------- encabezado y pregunta de negocio
        st.title(self.titulo)
        st.markdown(f'<div class="pregunta"><span>Pregunta de negocio</span>{self.PREGUNTA_NEGOCIO}</div>',
                    unsafe_allow_html=True)
        if d.vacio:
            st.warning("No hay ventas con esos filtros. Amplía la selección en la barra lateral.")
            return
        st.caption(f"Periodo: {d.periodo()} · {numero(len(d.V))} líneas de pedido · "
                   f"{len(d.sucursales)} sucursal(es)")

        # ---------------- KPIs
        k = d.kpis()
        cols = st.columns(6)
        extra = {"delta_arrow": "off"} if _acepta_delta_arrow(st) else {}
        cols[0].metric("Ventas", dinero(k["Ventas"]), f"{pct(k['% de las ventas totales'], 0)} del total",
                       delta_color="off", **extra)
        cols[1].metric("Pedidos", numero(k["Pedidos"]))
        cols[2].metric("Ticket promedio", dinero(k["Ticket promedio"], 2))
        cols[3].metric("Clientes", numero(k["Clientes"]))
        cols[4].metric("Pizzas por pedido", numero(k["Pizzas por pedido"], 2))
        cols[5].metric("Distancia media", f"{numero(k['Distancia media (km)'], 1)} km")

        tabs = st.tabs(["🎯 Respuesta de negocio", "🏪 Dónde · sucursales", "📅 Cuándo", "🍕 Qué",
                        "👥 A quién · clientes", "📍 Distancia y mapa", "📄 Datos"])
        cfg = {"displayModeBar": False}

        with tabs[0]:
            st.subheader("Cinco preguntas, cinco respuestas")
            for r in d.respuestas():
                with st.container(border=True):
                    st.markdown(f"**{r['id']} · {r['pregunta']}**")
                    st.markdown(_esc(r["respuesta"]))
                    c1, c2 = st.columns(2)
                    c1.markdown(f"📊 **Cómo se ve:** {_esc(r['visual'])}")
                    c2.markdown(f"✅ **Decisión:** {_esc(r['decision'])}")

        with tabs[1]:
            c1, c2 = st.columns([1.6, 1])
            c1.plotly_chart(d.figura_ventas_sucursal(), width="stretch", config=cfg)
            c2.plotly_chart(d.figura_ventas_pais(), width="stretch", config=cfg)
            t = d.A.ventas_por_sucursal()[["posición", "sucursal_id", "ciudad", "pais", "ventas", "% ventas",
                                           "pedidos", "clientes", "ticket promedio", "venta por cliente"]]
            st.dataframe(t, hide_index=True, width="stretch", column_config={
                "ventas": st.column_config.NumberColumn("ventas", format="$%,.0f"),
                "% ventas": st.column_config.ProgressColumn("% ventas", format="percent", min_value=0, max_value=1),
                "ticket promedio": st.column_config.NumberColumn(format="$%.2f"),
                "venta por cliente": st.column_config.NumberColumn(format="$%.2f")})

        with tabs[2]:
            st.plotly_chart(d.figura_ventas_diarias(), width="stretch", config=cfg)
            c1, c2 = st.columns(2)
            c1.plotly_chart(d.figura_dia_semana(), width="stretch", config=cfg)
            c2.plotly_chart(d.figura_hora(), width="stretch", config=cfg)

        with tabs[3]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(d.figura_mix_categoria(), width="stretch", config=cfg)
            c2.plotly_chart(d.figura_tamanos(), width="stretch", config=cfg)
            st.plotly_chart(d.figura_top_pizzas(10), width="stretch", config=cfg)

        with tabs[4]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(d.figura_segmentos(), width="stretch", config=cfg)
            c2.plotly_chart(d.figura_pareto_clientes(), width="stretch", config=cfg)
            st.markdown("**Top 10 clientes**")
            st.dataframe(d.A.top_clientes(10), hide_index=True, width="stretch", column_config={
                "ventas": st.column_config.NumberColumn(format="$%.2f"),
                "ticket promedio": st.column_config.NumberColumn(format="$%.2f"),
                "distancia_km": st.column_config.NumberColumn("distancia (km)", format="%.1f")})

        with tabs[5]:
            c1, c2 = st.columns(2)
            c1.plotly_chart(d.figura_distancia(), width="stretch", config=cfg)
            c2.plotly_chart(d.figura_ticket_distancia(), width="stretch", config=cfg)
            opciones = list(d.sucursales["sucursal_id"])
            sede = st.selectbox("Sucursal para el mapa", opciones, format_func=lambda s: f"{s} · {d._ciudad(s)}")
            st.plotly_chart(d.figura_mapa(sede), width="stretch")

        with tabs[6]:
            cols_ver = ["order_id", "order_date", "hora", "sucursal_id", "ciudad", "cliente_nombre",
                        "cliente_barrio", "pizza_name", "pizza_size", "pizza_category", "quantity",
                        "total_price", "distancia_km", "origen"]
            cols_ver = [c for c in cols_ver if c in d.V.columns]
            st.dataframe(d.V[cols_ver], hide_index=True, width="stretch", height=420)
            st.download_button("Descargar CSV filtrado", d.V.to_csv(index=False).encode("utf-8"),
                               "pizzeria_filtrado.csv", "text/csv")

    def _ciudad_total(self, suc_id: str) -> str:
        fila = self.V_total.loc[self.V_total["sucursal_id"].astype(str) == suc_id, "ciudad"]
        return fila.iloc[0] if len(fila) else suc_id


def _acepta_delta_arrow(st) -> bool:
    import inspect
    try:
        return "delta_arrow" in inspect.signature(st.metric).parameters
    except (TypeError, ValueError):
        return False


def _esc(texto: str) -> str:
    """Streamlit interpreta $...$ como fórmula LaTeX: se escapa el símbolo de pesos."""
    return texto.replace("$", r"\$")


_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.pregunta {border-left: 4px solid #eb6834; background: rgba(235,104,52,.07); padding: .8rem 1rem;
           border-radius: 6px; margin: .2rem 0 1rem; font-size: 1.02rem; line-height: 1.45;}
.pregunta span {display:block; font-size:.72rem; font-weight:700; letter-spacing:.06em;
                text-transform:uppercase; color:#b4491f; margin-bottom:.2rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: .55rem .8rem;}
</style>
"""
