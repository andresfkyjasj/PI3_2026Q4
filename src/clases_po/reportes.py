"""Generación de reportes de DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class Reportes:
    """Imprime y guarda un resumen legible de un DataFrame."""

    def generar_reporte(self, df: pd.DataFrame) -> str:
        """Genera el contenido textual del reporte."""
        self._validar_dataframe(df)
        numericas = df.select_dtypes(include="number")
        lineas = [
            "REPORTE DEL DATAFRAME",
            "=" * 60,
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Columnas y tipos:",
            *[f"- {columna}: {tipo}" for columna, tipo in df.dtypes.items()],
            "",
            "Valores nulos por columna:",
            *[f"- {columna}: {cantidad}" for columna, cantidad in df.isna().sum().items()],
        ]

        if not numericas.empty:
            lineas += [
                "",
                "Estadísticas numéricas:",
                numericas.describe().round(3).to_string(),
            ]

        return "\n".join(lineas) + "\n"

    def imprimir_reporte(self, df: pd.DataFrame) -> None:
        """Imprime en la consola el reporte del DataFrame."""
        print(self.generar_reporte(df), end="")

    def guardar_reporte(self, df: pd.DataFrame, ruta: str | Path) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generar_reporte(df), encoding="utf-8")
        return path
        
    def reporte_regresion_lineal(
        self,
        df: pd.DataFrame,
        regresion: object,
        ruta: str | Path = "entrenamiento/reporte_regresion_lineal.txt",
    ) -> Path:
        """Genera un TXT con el DataFrame y los resultados de una regresión.

        La regresión debe haber sido ejecutada previamente con
        ``RegresionLineal.ejecutar(df)``. El reporte no vuelve a entrenar,
        predecir ni evaluar el modelo: utiliza los resultados que la instancia
        conserva en ``modelo``, ``parametros``, ``metricas`` y
        ``ruta_modelo``.

        El archivo incluye la configuración de columnas, las dimensiones de
        los conjuntos de datos, los parámetros ``w`` y ``b``, las métricas
        MAE/RMSE/R², la ruta del modelo persistido, el resumen del DataFrame
        y sus primeras diez filas.

        Args:
            df: DataFrame utilizado para preparar la regresión.
            regresion: Instancia ejecutada de ``RegresionLineal``.
            ruta: Ruta del archivo TXT que se creará.

        Returns:
            Ruta del archivo TXT generado.

        Raises:
            TypeError: Si ``df`` no es un DataFrame, ``regresion`` no es una
                instancia de ``RegresionLineal`` o ``ruta`` no es una ruta.
            ValueError: Si la instancia todavía no tiene resultados de una
                ejecución completa.
            OSError: Si no se puede crear o escribir el archivo.
        """
        self._validar_dataframe(df)
        if not isinstance(ruta, (str, Path)):
            raise TypeError("ruta debe ser una cadena o pathlib.Path.")

        from .regresion_lineal import RegresionLineal

        if not isinstance(regresion, RegresionLineal):
            raise TypeError("regresion debe ser una instancia de RegresionLineal.")
        resultados_requeridos = {
            "modelo": regresion.modelo,
            "parametros": regresion.parametros,
            "metricas": regresion.metricas,
            "ruta_modelo": regresion.ruta_modelo,
        }
        faltantes = [
            nombre for nombre, valor in resultados_requeridos.items()
            if valor is None
        ]
        if faltantes:
            raise ValueError(
                "La regresión debe ejecutarse antes de generar el reporte. "
                f"Faltan resultados: {', '.join(faltantes)}."
            )

        lineas = [
            "REPORTE DE REGRESIÓN LINEAL",
            "=" * 70,
            f"Columnas predictoras (X): {regresion.feature_column}",
            f"Columna objetivo (y): {regresion.target_column}",
            f"Test size: {regresion.test_size}",
            f"Random state: {regresion.random_state}",
            "",
            "MODELO",
            "-" * 70,
            f"Ruta del modelo: {regresion.ruta_modelo}",
            f"Parámetros: {regresion.parametros}",
            "",
            "MÉTRICAS DE EVALUACIÓN",
            "-" * 70,
            *[
                f"{nombre.upper()}: {valor:.6f}"
                for nombre, valor in regresion.metricas.items()
            ],
            "",
            "DIMENSIONES DE LOS DATOS",
            "-" * 70,
            f"DataFrame original: {df.shape}",
            f"X: {regresion.X.shape if regresion.X is not None else 'no disponible'}",
            f"y: {regresion.y.shape if regresion.y is not None else 'no disponible'}",
            (
                "X_train: "
                f"{regresion.X_train.shape if regresion.X_train is not None else 'no disponible'}"
            ),
            (
                "X_test: "
                f"{regresion.X_test.shape if regresion.X_test is not None else 'no disponible'}"
            ),
            (
                "y_train: "
                f"{regresion.y_train.shape if regresion.y_train is not None else 'no disponible'}"
            ),
            (
                "y_test: "
                f"{regresion.y_test.shape if regresion.y_test is not None else 'no disponible'}"
            ),
            "",
            "REPORTE DEL DATAFRAME",
            "-" * 70,
            self.generar_reporte(df).rstrip(),
            "",
            "PRIMERAS 10 FILAS",
            "-" * 70,
            df.head(10).to_string(index=False),
            "",
        ]
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lineas), encoding="utf-8")
        print(f"[reporte_regresion_lineal] reporte guardado en: {path}")
        return path


    @staticmethod
    def _validar_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
