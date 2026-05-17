import os
from datetime import datetime
from firebase_admin import messaging

from app.prediccion_individual import predecir_semana_municipio
from app.umbrales import cargar_umbrales

# =========================
# CONFIG
# =========================
UMBRALES = cargar_umbrales()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://dashboard-ebro.fly.dev").rstrip("/")
PUSH_ICON_URL = f"{PUBLIC_BASE_URL}/static/icon.png"

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
def _is_invalid_token_error(exc):
    text = str(exc).lower()
    code = str(getattr(exc, "code", "") or getattr(exc, "error_code", "")).lower()
    combined = f"{code} {text}"

    return any(
        marker in combined
        for marker in (
            "device unregistered",
            "registration-token-not-registered",
            "requested entity was not found",
            "unregistered",
            "invalid registration token",
            "invalid-registration-token",
        )
    )


def enviar_notificacion(tokens, titulo, cuerpo):

    sent = 0
    invalid_tokens = set()

    for token in list(tokens):

        try:

            message = messaging.Message(
                data={
                    "title": str(titulo),
                    "body": str(cuerpo),
                    "url": "/",
                    "tag": "rio-ebro-alert",
                    "icon": PUSH_ICON_URL,
                },
                webpush=messaging.WebpushConfig(
                    headers={
                        "TTL": "86400",
                        "Urgency": "high",
                    },
                    notification=messaging.WebpushNotification(
                        title=str(titulo),
                        body=str(cuerpo),
                        icon=PUSH_ICON_URL,
                        badge=PUSH_ICON_URL,
                        tag="rio-ebro-alert",
                        renotify=True,
                        require_interaction=True,
                    ),
                    fcm_options=messaging.WebpushFCMOptions(
                        link=f"{PUBLIC_BASE_URL}/"
                    ),
                ),
                token=token,
            )

            messaging.send(message)

            sent += 1

            print(f"✅ Notificación enviada a {token[:15]}")

        except Exception as e:

            if _is_invalid_token_error(e):
                invalid_tokens.add(token)
                print(f"[PUSH CLEANUP] Token invalido detectado: {token[:15]}")

            print(f"❌ Error enviando push: {e}")


    return {
        "sent": sent,
        "invalid_tokens": invalid_tokens,
    }


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

    result = {
        "sent": 0,
        "invalid_tokens": set(),
        "processed_sites": 0,
    }

    if ULTIMO_ENVIO == fecha_hoy:
        return result

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
            tokens_municipio = set(tokens.get(site_id, set()))

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
            push_result = enviar_notificacion(
                tokens_municipio,
                titulo,
                mensaje
            )

            result["sent"] += push_result["sent"]
            result["invalid_tokens"].update(push_result["invalid_tokens"])
            result["processed_sites"] += 1

            print(
                f"📩 {nombre}: enviada a {push_result['sent']} usuarios "
                f"({len(push_result['invalid_tokens'])} invalidos)"
            )

        except Exception as e:

            print(f"❌ Error en alertas de {nombre}: {e}")

    # =========================
    # MARCAR COMO ENVIADO
    # =========================
    ULTIMO_ENVIO = fecha_hoy

    return result
