"""Cliente para el LLM de NVIDIA (API compatible con OpenAI).

Encapsula la llamada en streaming del ejemplo oficial de NVIDIA, agrega caché en disco,
extracción de JSON y reintentos con respaldo para que un notebook nunca se rompa.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


class LLMNvidia:
    """Conexión con el API de NVIDIA.

    Parámetros
    ----------
    api_key : clave ``nvapi-...``. Si no se pasa, se lee de la variable de entorno ``NVIDIA_API_KEY``.
    modelo : identificador del modelo en NVIDIA.
    pensar : valor de ``enable_thinking``. ``True`` razona más pero es mucho más lento.
    mostrar_razonamiento : imprime el ``reasoning_content`` mientras llega.
    ruta_cache : archivo JSON donde se guardan las respuestas (re-ejecución reproducible).
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        modelo: str = "nvidia/nemotron-3-ultra-550b-a55b",
        pensar: bool = False,
        mostrar_razonamiento: bool = False,
        ruta_cache: str | Path = "datos/cache_llm.json",
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.modelo = modelo
        self.pensar = pensar
        self.mostrar_razonamiento = mostrar_razonamiento
        self.ruta_cache = Path(ruta_cache)
        self.cache = self._leer_cache()
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(base_url=self.BASE_URL, api_key=self.api_key)
            except ImportError:
                print("Instala la librería con: pip install openai")

    # ------------------------------------------------------------------ estado
    @property
    def activo(self) -> bool:
        """True si hay cliente configurado y la conexión no ha fallado."""
        return self.client is not None

    def desactivar(self, motivo: str = "") -> None:
        self.client = None
        if motivo:
            print(f"LLM desactivado: {motivo}")

    def __repr__(self) -> str:
        estado = "activo" if self.activo else "inactivo (se usará el respaldo)"
        return f"LLMNvidia(modelo='{self.modelo}', pensar={self.pensar}, estado={estado})"

    # ------------------------------------------------------------------ llamadas
    def _crear_stream(self, mensajes: list[dict], temperature: float, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.modelo,
            messages=mensajes,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.pensar}},
            stream=True,
        )

    def _recorrer_stream(self, completion, imprimir: bool) -> str:
        """Mismo recorrido del ejemplo de NVIDIA: razonamiento y contenido a medida que llegan."""
        partes = []
        for chunk in completion:
            if not chunk.choices:
                continue
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning and (imprimir or self.mostrar_razonamiento):
                print(reasoning, end="")
            if chunk.choices[0].delta.content is not None:
                partes.append(chunk.choices[0].delta.content)
                if imprimir:
                    print(chunk.choices[0].delta.content, end="")
        return "".join(partes)

    def probar_conexion(
        self,
        prompt: str = "En una sola frase en español: ¿para qué sirve la analítica descriptiva en una pizzería?",
    ) -> bool:
        """Llamada de prueba en streaming (código de referencia de NVIDIA)."""
        if not self.activo:
            print("Prueba omitida: no hay NVIDIA_API_KEY configurada.")
            return False
        try:
            completion = self._crear_stream([{"role": "user", "content": prompt}], temperature=1, max_tokens=16384)
            self._recorrer_stream(completion, imprimir=True)
            print()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo conectar con NVIDIA ({type(e).__name__}: {e}).")
            self.desactivar("se continuará con el catálogo de respaldo")
            return False

    def consultar(self, prompt: str, temperature: float = 0.6, max_tokens: int = 8192,
                  usar_cache: bool = True, solo_json: bool = True) -> str:
        """Envía un prompt y devuelve el texto completo. Usa y alimenta la caché en disco."""
        clave = hashlib.sha256(f"{self.modelo}|{self.pensar}|{prompt}".encode("utf-8")).hexdigest()[:20]
        if usar_cache and clave in self.cache:
            return self.cache[clave]
        mensajes = [{"role": "user", "content": prompt}]
        if solo_json:
            mensajes.insert(0, {"role": "system", "content": "Respondes únicamente con JSON válido, sin texto adicional."})
        texto = self._recorrer_stream(self._crear_stream(mensajes, temperature, max_tokens), imprimir=False)
        self.cache[clave] = texto
        self._guardar_cache()
        return texto

    def generar_texto(self, prompt: str, temperature: float = 0.4, max_tokens: int = 4096) -> str | None:
        """Texto libre impreso en streaming (por ejemplo, un resumen ejecutivo)."""
        if not self.activo:
            print("LLM no configurado: se omite la generación de texto.")
            return None
        try:
            texto = self._recorrer_stream(
                self._crear_stream([{"role": "user", "content": prompt}], temperature, max_tokens), imprimir=True)
            print()
            return texto
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo generar el texto con el LLM ({type(e).__name__}).")
            return None

    def pedir_json(self, nombre: str, prompt: str, validar: Callable[[Any], Any], respaldo: Any,
                   intentos: int = 3) -> tuple[Any, str]:
        """Pide un JSON, lo valida y reintenta. Devuelve ``(datos, fuente)`` con fuente 'LLM' o 'respaldo'.

        ``validar(datos)`` debe devolver los datos limpios o lanzar una excepción.
        """
        if not self.activo:
            return respaldo, "respaldo"
        for i in range(1, intentos + 1):
            try:
                datos = validar(self.extraer_json(self.consultar(prompt, usar_cache=(i == 1))))
                print(f"✔ {nombre}: respuesta del LLM válida (intento {i})")
                return datos, "LLM"
            except Exception as e:  # noqa: BLE001
                print(f"… {nombre}: intento {i} no válido ({type(e).__name__}: {str(e)[:120]})")
        print(f"✘ {nombre}: se usa el catálogo de respaldo")
        return respaldo, "respaldo"

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def extraer_json(texto: str) -> Any:
        """Quita bloques de razonamiento y ```json``` y devuelve el objeto JSON."""
        texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
        texto = re.sub(r"```(?:json)?", "", texto).strip()
        inicio = min([i for i in (texto.find("{"), texto.find("[")) if i >= 0], default=-1)
        fin = max(texto.rfind("}"), texto.rfind("]"))
        if inicio < 0 or fin < 0:
            raise ValueError("La respuesta no contiene JSON")
        return json.loads(texto[inicio:fin + 1])

    def _leer_cache(self) -> dict:
        if self.ruta_cache.exists():
            try:
                return json.loads(self.ruta_cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _guardar_cache(self) -> None:
        self.ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_cache.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
