"""App web del dashboard de la pizzería (Semana 6).

Ejecutar en local (desde la raíz del repositorio):
    streamlit run src/clases_po/app.py

En Streamlit Community Cloud: Main file path = src/clases_po/app.py
"""
import sys
from pathlib import Path

import streamlit as st

AQUI = Path(__file__).resolve().parent                 # src/clases_po
sys.path.insert(0, str(AQUI))                          # importa el módulo sin cargar todo el paquete
                                                       # (así la app no necesita datasets ni scikit-learn)
from dashboard_streamlit import DashboardStreamlit     # noqa: E402

RUTA_DATOS = AQUI.parent / "sincronicas" / "datos" / "ventas_enriquecidas.csv"


@st.cache_resource(show_spinner="Cargando ventas de enero 2015…")
def cargar() -> DashboardStreamlit:
    return DashboardStreamlit.desde_csv(RUTA_DATOS)


cargar().render()
