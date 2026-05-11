from datetime import datetime
import os

from firebase_admin import messaging

from prediccion_individual import predecir_semana_municipio
from umbrales import cargar_umbrales

# =========================
# CONFIG
# =========================
UMBRALES = cargar_umbrales()

# evitar enviar múltiples veces el mismo día
ULTIMO_ENVIO = None


# =========================
# CALCULAR ALERTA
# =========================
def calcular_nivel_alerta(municipio, predicciones):

    if not predicciones:
        return "nulo", "Sin datos suficientes"

    umbral = UMBRALES.get(municipio, {}).get("alerta")

    if not umbral:
        return "nulo", "Sin umbral definido"

    # 🔥 coger máximo nivel previsto HOY
    max_nivel = max([
        p.get("nivel", 0)
        for p in predicciones[:1]
    ])

    # =========================
    # CLASIFICACIÓN
    # =========================
    if max_nivel >= umbral:
        return (
            "rojo",
            f"🔴 ALERTA ROJA en {municipio.capitalize()}. "
            f"Nivel previsto: {max_nivel:.2f} m"
        )

    elif max_nivel >= umbral * 0.8:
        return (
            "naranja",
            f"🟠 Vigilancia en {municipio.capitalize()}. "
            f"Nivel previsto elevado: {max_nivel:.2f} m"
        )

    else:
        return (
            "verde",
            f"🟢 Situación estable en {municipio.capitalize()}. "
            f"Nivel previsto: {max_nivel:.2f} m"
        )


# =========================
# ENVIAR PUSH
# =========================
def enviar_notificacion(tokens, titulo, cuerpo):

    for token in tokens:

        try:

            message = messaging.Message(

                notification=messaging.Notification(
                    title=titulo,
                    body=cuerpo,
                ),

                token=token,
            )

            messaging.send(message)

            print(f"✅ Notificación enviada a {token[:15]}")

        except Exception as e:

            print(f"❌ Error enviando push: {e}")


# =========================
# ALERTAS DIARIAS
# =========================
def enviar_alertas_diarias(tokens, sites):

    global ULTIMO_ENVIO

    ahora = datetime.now()

    # =========================
    # SOLO UNA VEZ AL DÍA
    # =========================
    fecha_hoy = ahora.strftime("%Y-%m-%d")

    if ULTIMO_ENVIO == fecha_hoy:
        return

    # =========================
    # SOLO A LAS 08:00
    # =========================
    # DESCOMENTA PARA PRODUCCIÓN
    # if ahora.hour != 8:
    #     return

    print("🚨 ENVIANDO ALERTAS DIARIAS")

    # =========================
    # RECORRER MUNICIPIOS
    # =========================
    for site in sites:

        site_id = site["id"]
        nombre = site["name"]

        try:

            # =========================
            # TOKENS SUSCRITOS
            # =========================
            tokens_municipio = tokens.get(site_id, set())

            # nadie suscrito
            if not tokens_municipio:
                continue

            # =========================
            # PREDICCIÓN IA
            # =========================
            pred = predecir_semana_municipio(site_id)

            # =========================
            # CALCULAR ALERTA
            # =========================
            nivel_alerta, mensaje = calcular_nivel_alerta(
                site_id,
                pred
            )

            # =========================
            # TÍTULO
            # =========================
            if nivel_alerta == "rojo":

                titulo = "🔴 Alerta roja por desbordamiento"

            elif nivel_alerta == "naranja":

                titulo = "🟠 Riesgo moderado de crecida"

            else:

                titulo = "🟢 Situación hidrológica estable"

            # =========================
            # ENVIAR PUSH
            # =========================
            enviar_notificacion(
                tokens_municipio,
                titulo,
                mensaje
            )

            print(f"📩 {nombre}: enviada a {len(tokens_municipio)} usuarios")

        except Exception as e:

            print(f"❌ Error en alertas de {nombre}: {e}")

    # =========================
    # MARCAR COMO ENVIADO
    # =========================
    ULTIMO_ENVIO = fecha_hoy