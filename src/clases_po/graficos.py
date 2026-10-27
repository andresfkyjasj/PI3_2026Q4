"""Gráficos de la pizzería (hojas «Gráficos» y «Dashboard» del Excel, en matplotlib).

- :class:`GraficosBase` dibuja los 9 gráficos del Excel y el dashboard sobre un
  :class:`AnalisisBase` (por ejemplo, la base sin enriquecer).
- :class:`GraficosPizzeria` hereda todo y agrega los gráficos multisede
  (sucursales, clientes, distancia y mapa) sobre un :class:`AnalisisPizzeria`.

Cada método dibuja a partir de las tablas de la clase de análisis, así el gráfico
y la tabla siempre muestran el mismo número.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class GraficosBase:
    """Gráficos del Excel con un estilo común: un solo eje por gráfico, rejilla tenue y ejes discretos."""

    PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    COLOR = "#2a78d6"         # color único cuando no se distinguen series
    ACENTO = "#eb6834"        # resaltado (pizza elegida, sucursal en el mapa)
    TEXTO, TEXTO_2 = "#0b0b0b", "#52514e"

    def __init__(self, analisis, aplicar_estilo: bool = True):
        self.A = analisis
        self.dinero = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        self.pct = mticker.PercentFormatter(1.0)
        if aplicar_estilo:
            self.aplicar_estilo()

    @staticmethod
    def aplicar_estilo() -> None:
        plt.rcParams.update({
            "figure.figsize": (11, 4.5), "figure.dpi": 100,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
            "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
            "text.parse_math": False,   # para que "$9 a $12" no se interprete como fórmula
        })

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _sin_rejilla(ax, eje: str) -> None:
        ax.grid(axis=eje, visible=False)

    def _barras_h(self, ax, etiquetas, valores, colores=None, titulo="", dinero=True):
        ax.barh(etiquetas, valores, color=colores or self.COLOR, height=0.6)
        if dinero:
            ax.xaxis.set_major_formatter(self.dinero)
        ax.set_title(titulo)
        self._sin_rejilla(ax, "y")

    def _heatmap(self, ax, matriz, etiquetas_x, etiquetas_y, titulo, formato=None, anotar=False, cbar_fmt=None):
        im = ax.imshow(matriz, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(etiquetas_x)), etiquetas_x)
        ax.set_yticks(range(len(etiquetas_y)), etiquetas_y)
        if anotar:
            umbral = np.nanpercentile(matriz, 65)
            for i in range(matriz.shape[0]):
                for j in range(matriz.shape[1]):
                    ax.text(j, i, formato(matriz[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if matriz[i, j] > umbral else self.TEXTO)
        ax.grid(False)
        ax.set_title(titulo)
        plt.colorbar(im, ax=ax, format=cbar_fmt, fraction=0.03)

    # ------------------------------------------------------------------ Gráficos 1 y 2 · tiempo
    def ventas_por_dia(self):
        """1. Columnas: ventas por día · 2. Línea: la pizza elegida en el tiempo."""
        t = self.A.ventas_por_dia()
        etiquetas = t.index.strftime("%d/%m")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(etiquetas, t["ventas"], color=self.COLOR, width=0.7)
        axes[0].yaxis.set_major_formatter(self.dinero)
        axes[0].set_title("1. Ventas por día"); axes[0].tick_params(axis="x", rotation=45)
        self._sin_rejilla(axes[0], "x")
        axes[1].plot(etiquetas, t["ventas pizza elegida"], color=self.ACENTO, lw=2, marker="o", ms=5)
        axes[1].yaxis.set_major_formatter(self.dinero); axes[1].set_ylim(bottom=0)
        axes[1].set_title(f"2. {self.A.pizza_elegida} en el tiempo"); axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout(); plt.show()

    def dia_semana(self):
        t = self.A.dia_semana_resumen()
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(t.index, t["venta promedio por día"], color=self.COLOR, width=0.6)
        for x, (v, n) in enumerate(zip(t["venta promedio por día"], t["días en el periodo"])):
            ax.text(x, v, f"{n} día{'s' if n > 1 else ''}", ha="center", va="bottom", fontsize=8, color=self.TEXTO_2)
        ax.yaxis.set_major_formatter(self.dinero)
        ax.set_title("Venta promedio por día de la semana"); self._sin_rejilla(ax, "x")
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ Gráficos 3, 4 y 5 · categorías y tamaños
    def categorias(self):
        """3. Barras: ventas por categoría · 4. Circular: participación de cada categoría."""
        t = self.A.frecuencias("pizza_category").drop("Total").sort_values("ventas")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self._barras_h(axes[0], t.index, t["ventas"], titulo="3. Ventas por categoría")
        axes[1].pie(t["ventas"], labels=t.index, autopct="%1.1f%%", startangle=90, counterclock=False,
                    colors=self.PALETA[:len(t)], wedgeprops={"edgecolor": "white", "linewidth": 2},
                    textprops={"fontsize": 9})
        axes[1].set_title("4. Participación de cada categoría en las ventas")
        plt.tight_layout(); plt.show()

    def tamanos(self):
        """5. Columnas: pizzas por tamaño, respetando el orden natural S → XL (no ordenar por valor)."""
        t = self.A.frecuencias("pizza_size").drop("Total")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(t.index, t["pizzas"], color=self.COLOR, width=0.6)
        for x, (v, p) in enumerate(zip(t["pizzas"], t["% ventas"])):
            ax.text(x, v, f"{v:,} pizzas\n{p:.1%} de ventas", ha="center", va="bottom", fontsize=8, color=self.TEXTO_2)
        ax.set_ylim(0, t["pizzas"].max() * 1.2)
        ax.set_title("5. Pizzas vendidas por tamaño (orden S → XL)"); self._sin_rejilla(ax, "x")
        plt.tight_layout(); plt.show()

    def tamano_y_categoria(self):
        """5. Columnas: pizzas por tamaño (orden S → XL) y ventas por categoría."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        t = self.A.frecuencias("pizza_size").drop("Total")
        axes[0].bar(t.index, t["pizzas"], color=self.COLOR, width=0.6)
        axes[0].set_title("5. Pizzas vendidas por tamaño (orden S → XL)")
        self._sin_rejilla(axes[0], "x")
        t = self.A.frecuencias("pizza_category").drop("Total").sort_values("ventas")
        self._barras_h(axes[1], t.index, t["ventas"], titulo="Ventas por categoría")
        plt.tight_layout(); plt.show()

    def categoria_tamano(self):
        """6. Columnas agrupadas: categoría × tamaño."""
        ct = self.A.categoria_x_tamano("ventas").drop(index="Total", columns="ventas Total")
        ct.columns = [c.replace("ventas ", "") for c in ct.columns]
        ct = ct.loc[:, ct.sum() > 0]
        fig, ax = plt.subplots(figsize=(10, 4))
        ancho = 0.8 / ct.shape[1]
        for k, tam in enumerate(ct.columns):
            ax.bar(np.arange(len(ct)) + k * ancho, ct[tam], width=ancho * 0.92, color=self.PALETA[k], label=tam)
        ax.set_xticks(np.arange(len(ct)) + ancho * (ct.shape[1] - 1) / 2, ct.index)
        ax.yaxis.set_major_formatter(self.dinero)
        ax.legend(title="Tamaño", frameon=False, ncol=ct.shape[1])
        ax.set_title("6. Ventas por categoría y tamaño"); self._sin_rejilla(ax, "x")
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ Gráfico 7 · hora
    def ventas_hora(self):
        h = self.A.ventas_por_hora()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(h.index, h["ventas"], color=self.COLOR, width=0.75)
        ax.yaxis.set_major_formatter(self.dinero); ax.set_xticks(h.index)
        ax.set_title("7. Ventas por hora (dos picos: almuerzo y comida)"); self._sin_rejilla(ax, "x")
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ Gráfico 8 · ranking
    def top_pizzas(self, n: int = 10):
        top = self.A.ranking_pizzas().head(n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 0.45 * n + 0.8))
        self._barras_h(ax, top.index, top["ventas"], titulo=f"8. Top {n} pizzas por ventas")
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ Gráfico 9 · distribución
    def histogramas(self):
        dist = self.A.distribucion_precio()
        valor = self.A.pedidos["valor_pedido"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(dist.index, dist["líneas"], width=0.98, color=self.COLOR)
        axes[0].set_title("9. Histograma de total_price (líneas)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].hist(valor, bins=range(0, int(valor.quantile(0.99)) + 10, 10), color=self.COLOR, edgecolor="white")
        axes[1].xaxis.set_major_formatter(self.dinero)
        axes[1].set_title("Histograma del valor del pedido (ticket)")
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ Dashboard
    def dashboard(self, sucursal="Todas", categoria="Todas", tamano="Todos", titulo="Dashboard de ventas"):
        d = self.A.filtrar(sucursal, categoria, tamano)
        if d.empty:
            print("No hay ventas con esos filtros.")
            return
        kpis = self.A.kpis(sucursal, categoria, tamano)
        dia = d.groupby("order_date")["total_price"].sum()
        hora = d.groupby("hora")["total_price"].sum()
        top = d.groupby("pizza_name")["total_price"].sum().nlargest(5).iloc[::-1]
        cat = d.groupby("pizza_category", observed=True)["total_price"].sum()
        filtros = f"Categoría: {categoria} · Tamaño: {tamano}"
        if "sucursal_id" in d.columns:
            filtros = f"Sucursal: {sucursal} · " + filtros

        fig = plt.figure(figsize=(14, 8.2))
        fig.suptitle(f"{titulo} · {self.A.periodo()} · {filtros}", fontweight="bold", fontsize=12)
        paso = 0.94 / max(len(kpis), 1)
        for i, (k, v) in enumerate(kpis.items()):
            fig.text(0.03 + i * paso, 0.88, k, fontsize=7.5, color=self.TEXTO_2)
            fig.text(0.03 + i * paso, 0.845, v, fontsize=11 if len(v) < 14 else 7.5, fontweight="bold",
                     color=self.TEXTO)
        gs = fig.add_gridspec(2, 2, top=0.8, hspace=0.38, wspace=0.28)
        a1, a2, a3, a4 = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
        a1.bar(dia.index.day, dia.values, color=self.COLOR); a1.set_title("Ventas por día")
        a2.bar(hora.index, hora.values, color=self.COLOR); a2.set_title("Ventas por hora")
        self._barras_h(a3, top.index, top.values, titulo="Top 5 pizzas"); a3.tick_params(axis="y", labelsize=8)
        a4.bar(cat.index.astype(str), cat.values, color=self.COLOR, width=0.6); a4.set_title("Ventas por categoría")
        for a in (a1, a2, a4):
            a.yaxis.set_major_formatter(self.dinero)
            self._sin_rejilla(a, "x")
        plt.show()


class GraficosPizzeria(GraficosBase):
    """Gráficos multisede: un color fijo por sucursal (nunca por posición en el ranking)."""

    def __init__(self, analisis, aplicar_estilo: bool = True):
        super().__init__(analisis, aplicar_estilo)
        self.color_suc = dict(zip(self.A.SUC, self.PALETA))

    # ------------------------------------------------------------------ sucursales
    def ventas_sucursal(self):
        d = self.A.ventas_por_sucursal().sort_values("ventas")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        barras = ax.barh(d["ciudad"], d["ventas"], color=[self.color_suc[c] for c in d["sucursal_id"]], height=0.6)
        for b, v, p in zip(barras, d["ventas"], d["% ventas"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f"  ${v:,.0f} ({p:.0%})", va="center", fontsize=9)
        ax.xaxis.set_major_formatter(self.dinero)
        ax.set_xlim(0, d["ventas"].max() * 1.25)
        ax.set_title("Ventas del mes por sucursal")
        self._sin_rejilla(ax, "y")
        plt.tight_layout(); plt.show()

    def ventas_diarias(self, ultimo_dia_real: int | None = 12):
        dias = self.A.ventas_por_dia()
        diario = self.A.ventas_diarias_sucursal()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
        axes[0].bar(dias.index.day, dias["ventas"], color=self.COLOR, width=0.75)
        axes[0].yaxis.set_major_formatter(self.dinero); axes[0].set_xlabel("día del mes")
        axes[0].set_title("Empresa: ventas por día"); self._sin_rejilla(axes[0], "x")
        for c in self.A.SUC:
            axes[1].plot(diario.index.day, diario[c], color=self.color_suc[c], lw=2, label=self.A.nombre_suc[c])
        axes[1].yaxis.set_major_formatter(self.dinero); axes[1].set_xlabel("día del mes")
        axes[1].set_title("Ventas diarias por sucursal")
        axes[1].legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        if ultimo_dia_real:
            axes[1].axvline(ultimo_dia_real + 0.5, color="gray", ls=":", lw=1)
            axes[1].text(ultimo_dia_real + 0.7, axes[1].get_ylim()[0], f" ← {self.A.SUC[0]} real | simulado →",
                         fontsize=8, color="gray", va="bottom")
        plt.tight_layout(); plt.show()

    def dia_semana(self):
        t = self.A.dia_semana_sucursal()
        fig, ax = plt.subplots(figsize=(10, 3.2))
        self._heatmap(ax, t.values, list(t.columns), [f"{c} · {self.A.nombre_suc[c]}" for c in t.index],
                      "Venta promedio por día de la semana y sucursal", formato=lambda v: f"${v:,.0f}",
                      anotar=True, cbar_fmt=self.dinero)
        plt.tight_layout(); plt.show()

    def ventas_hora(self):
        h = self.A.ventas_por_hora()
        hs = self.A.hora_sucursal()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(h.index, h["ventas"], color=self.COLOR, width=0.75)
        axes[0].yaxis.set_major_formatter(self.dinero); axes[0].set_xticks(h.index)
        axes[0].set_title("Empresa: ventas por hora (almuerzo y comida)"); self._sin_rejilla(axes[0], "x")
        self._heatmap(axes[1], hs.div(hs.sum(axis=1), axis=0).values, list(hs.columns),
                      [self.A.nombre_suc[c] for c in hs.index], "% de ventas de cada sucursal por hora",
                      cbar_fmt=self.pct)
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ clientes
    def clientes(self):
        pc = self.A.clientes_detalle()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        frec = pc["pedidos"].value_counts().sort_index()
        axes[0].bar(frec.index, frec.values, color=self.COLOR, width=0.75)
        axes[0].set_xlabel("pedidos en el mes"); axes[0].set_ylabel("clientes")
        axes[0].set_title("¿Cuántas veces pide un cliente en el mes?"); self._sin_rejilla(axes[0], "x")
        for c in self.A.SUC:
            v = pc.loc[pc["sucursal_id"] == c, "ventas"].sort_values(ascending=False).to_numpy()
            axes[1].plot(np.arange(1, len(v) + 1) / len(v), v.cumsum() / v.sum(), color=self.color_suc[c],
                         lw=2, label=self.A.nombre_suc[c])
        axes[1].axvline(0.2, color="gray", ls=":", lw=1)
        axes[1].xaxis.set_major_formatter(self.pct); axes[1].yaxis.set_major_formatter(self.pct)
        axes[1].set_xlabel("% de clientes (de mejor a peor)"); axes[1].set_ylabel("% acumulado de ventas")
        axes[1].set_title("Curva de Pareto de clientes por sucursal"); axes[1].legend(frameon=False, fontsize=8)
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ distancia
    def distancia(self):
        dr = self.A.distancia_rangos()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        axes[0].bar(dr.index, dr["ventas"], color=self.COLOR, width=0.6)
        axes[0].yaxis.set_major_formatter(self.dinero); axes[0].set_title("Ventas por rango de distancia (empresa)")
        axes[1].bar(dr.index, dr["ticket_promedio"], color=self.COLOR, width=0.6)
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.1f}"))
        axes[1].set_title("Ticket promedio por rango de distancia")
        for ax in axes:
            self._sin_rejilla(ax, "x")
        plt.tight_layout(); plt.show()

    def mapa_clientes(self, km_max: float = 10):
        pc = self.A.clientes_detalle().merge(
            self.A.clientes[["cliente_id", "cliente_latitud", "cliente_longitud"]], on="cliente_id")
        suc = self.A.sucursales
        fig, axes = plt.subplots(1, len(suc), figsize=(3.6 * len(suc), 4.2))
        axes = np.atleast_1d(axes)
        sc = None
        for ax, (_, s) in zip(axes, suc.iterrows()):
            cl = pc[pc["sucursal_id"] == s["sucursal_id"]]
            sc = ax.scatter(cl["cliente_longitud"], cl["cliente_latitud"], c=cl["distancia_km"], cmap="Blues",
                            vmin=0, vmax=km_max, s=10 + cl["pedidos"] * 6, alpha=0.85,
                            edgecolor="#fcfcfb", linewidth=0.4)
            ax.scatter(s["sucursal_longitud"], s["sucursal_latitud"], marker="*", s=320, color=self.ACENTO,
                       edgecolor=self.TEXTO, linewidth=0.6, zorder=5)
            ax.set_title(f'{s["ciudad"]} ({s["pais"]})', fontsize=10)
            ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=axes, label="distancia (km)", fraction=0.012)
        fig.suptitle("Ubicación de clientes (tamaño = pedidos) y sucursal (★)", fontweight="bold")
        plt.show()
