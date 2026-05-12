from datetime import datetime
from firebase_admin import messaging

from app.prediccion_individual import predecir_semana_municipio
from app.umbrales import cargar_umbrales

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

    municipio = municipio.strip().lower()

    if not predicciones:
        return "nulo", "Sin datos suficientes"

    # =========================
    # OBTENER UMBRALES
    # =========================
    umbrales = UMBRALES.get(municipio)

    if not umbrales:
        return "nulo", f"Sin umbral definido para {municipio}"

    amarillo = umbrales.get("amarillo")
    naranja = umbrales.get("naranja")
    rojo = umbrales.get("rojo")

    # =========================
    # MÁXIMO NIVEL PREVISTO HOY
    # =========================
    max_nivel = max([
        p.get("nivel", 0)
        for p in predicciones[:1]
    ])

    nombre = municipio.replace("_", " ").title()

    # =========================
    # ALERTA ROJA
    # =========================
    if rojo is not None and max_nivel >= rojo:

        return (
            "rojo",
            f"🔴 ALERTA ROJA en {nombre}. "
            f"Nivel previsto: {max_nivel:.2f} m"
        )

    # =========================
    # ALERTA NARANJA
    # =========================
    elif naranja is not None and max_nivel >= naranja:

        return (
            "naranja",
            f"🟠 Riesgo importante en {nombre}. "
            f"Nivel previsto: {max_nivel:.2f} m"
        )

    # =========================
    # ALERTA AMARILLA
    # =========================
    elif amarillo is not None and max_nivel >= amarillo:

        return (
            "amarillo",
            f"🟡 Vigilancia activa en {nombre}. "
            f"Nivel previsto: {max_nivel:.2f} m"
        )

    # =========================
    # NORMAL
    # =========================
    else:

        return (
            "verde",
            f"🟢 Situación estable en {nombre}. "
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
            # TÍTULO PUSH
            # =========================
            if nivel_alerta == "rojo":

                titulo = "🔴 Alerta roja por desbordamiento"

            elif nivel_alerta == "naranja":

                titulo = "🟠 Riesgo importante de crecida"

            elif nivel_alerta == "amarillo":

                titulo = "🟡 Vigilancia por crecida"

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