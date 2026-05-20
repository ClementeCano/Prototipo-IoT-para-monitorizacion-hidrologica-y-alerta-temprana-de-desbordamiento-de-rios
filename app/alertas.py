import os
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from typing import Any
from zoneinfo import ZoneInfo

import firebase_admin
from firebase_admin import messaging

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from app.prediccion_individual import predecir_semana_municipio
    from app.umbrales import cargar_umbrales
except ImportError:
    from app.prediccion_individual import predecir_semana_municipio
    from app.umbrales import cargar_umbrales

# =========================
# CONFIG
# =========================
UMBRALES = cargar_umbrales()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://dashboard-ebro.fly.dev").rstrip("/")
PUSH_ICON_URL = f"{PUBLIC_BASE_URL}/static/icon.png"
ALERT_HOUR = int(os.getenv("ALERT_HOUR", "8"))
ALERT_MINUTE = int(os.getenv("ALERT_MINUTE", "0"))
ALERT_TIMEZONE = os.getenv("ALERT_TIMEZONE", "Europe/Madrid")
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "alertas@rio-ebro.local").strip()
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1").lower() in {"1", "true", "yes"}
SMTP_SSL = os.getenv("SMTP_SSL", "0").lower() in {"1", "true", "yes"}

# evitar enviar múltiples veces el mismo día
ULTIMO_ENVIO = None


def _now_alert_tz():
    try:
        return datetime.now(ZoneInfo(ALERT_TIMEZONE))
    except Exception:
        return datetime.now()


def _titulo_por_nivel(nivel_alerta: str) -> str:
    if nivel_alerta == "rojo":
        return "🔴 Alerta roja por desbordamiento"

    if nivel_alerta == "naranja":
        return "🟠 Riesgo importante de crecida"

    if nivel_alerta == "amarillo":
        return "🟡 Vigilancia por crecida"

    return "🟢 Situación hidrológica estable"


def smtp_config_status() -> dict[str, Any]:
    missing = []

    if not SMTP_HOST:
        missing.append("SMTP_HOST")

    if SMTP_USER and not SMTP_PASSWORD:
        missing.append("SMTP_PASSWORD")

    return {
        "configured": not missing,
        "missing": missing,
        "host": SMTP_HOST or None,
        "port": SMTP_PORT,
        "from": SMTP_FROM,
        "starttls": SMTP_STARTTLS,
        "ssl": SMTP_SSL,
        "auth_enabled": bool(SMTP_USER),
    }


def smtp_config_message() -> str:
    status = smtp_config_status()

    if status["configured"]:
        return "SMTP configurado"

    missing = ", ".join(status["missing"])

    return (
        "El envío por correo no está configurado en el servidor. "
        f"Faltan estas variables de entorno: {missing}. "
        "Para Gmail usa SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, "
        "SMTP_USER=tu_correo@gmail.com, SMTP_PASSWORD=contraseña_de_aplicación, "
        "SMTP_FROM=tu_correo@gmail.com y SMTP_STARTTLS=1."
    )


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


def enviar_notificacion(tokens, titulo, cuerpo, tag="rio-ebro-alert", url="/"):

    sent = 0
    invalid_tokens = set()
    errors = []

    if not firebase_admin._apps:
        error = "firebase_not_initialized"
        print(f"❌ Error enviando push: {error}")
        return {
            "sent": sent,
            "invalid_tokens": invalid_tokens,
            "errors": [error],
        }

    for token in list(tokens):

        try:

            message = messaging.Message(
                data={
                    "title": str(titulo),
                    "body": str(cuerpo),
                    "url": str(url or "/"),
                    "tag": str(tag or "rio-ebro-alert"),
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
                        tag=str(tag or "rio-ebro-alert"),
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

            errors.append(str(e))
            print(f"❌ Error enviando push: {e}")


    return {
        "sent": sent,
        "invalid_tokens": invalid_tokens,
        "errors": errors,
    }


def enviar_email(destinatario: str, asunto: str, cuerpo: str) -> dict[str, Any]:
    destinatario = (destinatario or "").strip()

    if not destinatario:
        raise RuntimeError("email_empty")

    smtp_status = smtp_config_status()

    if not smtp_status["configured"]:
        raise RuntimeError(smtp_config_message())

    message = EmailMessage()
    message["From"] = SMTP_FROM
    message["To"] = destinatario
    message["Subject"] = asunto
    message.set_content(cuerpo)

    smtp_class = smtplib.SMTP_SSL if SMTP_SSL else smtplib.SMTP

    with smtp_class(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        if not SMTP_SSL and SMTP_STARTTLS:
            smtp.starttls()

        if SMTP_USER:
            smtp.login(SMTP_USER, SMTP_PASSWORD)

        smtp.send_message(message)

    print(f"✅ Email enviado a {destinatario}")
    return {"sent": 1}


def construir_alerta_site(site: dict[str, Any]) -> dict[str, Any]:
    site_id = site["id"]
    pred = predecir_semana_municipio(site_id)
    nivel_alerta, mensaje = calcular_nivel_alerta(site_id, pred)

    return {
        "site_id": site_id,
        "site_name": site.get("name", site_id),
        "nivel_alerta": nivel_alerta,
        "titulo": _titulo_por_nivel(nivel_alerta),
        "mensaje": mensaje,
    }


def _selected_sites_for_user(user: dict[str, Any], sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferences = user.get("preferences") or {}
    selected = set(preferences.get("sites") or [])

    return [
        site
        for site in sites
        if site.get("id") in selected
    ]


def _push_tokens_for_user(user: dict[str, Any]) -> set[str]:
    return {
        (device.get("token") or "").strip()
        for device in user.get("devices", [])
        if (device.get("token") or "").strip()
    }


def _build_email_body(user: dict[str, Any], alerts: list[dict[str, Any]]) -> str:
    lines = [
        f"Hola {user.get('name') or user.get('email')},",
        "",
        "Este es tu resumen diario de alertas hidrológicas del Río Ebro.",
        "",
    ]

    for alert in alerts:
        lines.extend([
            f"{alert['site_name']}",
            f"{alert['titulo']}",
            f"{alert['mensaje']}",
            "",
        ])

    lines.extend([
        f"Dashboard: {PUBLIC_BASE_URL}/",
        "",
        "Puedes cambiar la hora, el canal y los municipios desde tu perfil.",
    ])

    return "\n".join(lines)


def _site_alert_from_cache(
    site: dict[str, Any],
    alert_cache: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    site_id = site["id"]

    if alert_cache is not None and site_id in alert_cache:
        return alert_cache[site_id]

    alert = construir_alerta_site(site)

    if alert_cache is not None:
        alert_cache[site_id] = alert

    return alert


def enviar_alerta_usuario(
    user: dict[str, Any],
    sites: list[dict[str, Any]],
    alert_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preferences = user.get("preferences") or {}
    channel = preferences.get("notification_channel", "push")
    selected_sites = _selected_sites_for_user(user, sites)

    result = {
        "user_id": user.get("id"),
        "sent": 0,
        "push_sent": 0,
        "email_sent": 0,
        "invalid_tokens": set(),
        "processed_sites": 0,
        "errors": [],
        "skipped": False,
        "reason": None,
    }

    if not selected_sites:
        result["skipped"] = True
        result["reason"] = "no_sites"
        return result

    site_alerts = []

    for site in selected_sites:
        try:
            site_alerts.append(_site_alert_from_cache(site, alert_cache))
            result["processed_sites"] += 1
        except Exception as exc:
            error = f"{site.get('id')}: {exc}"
            result["errors"].append(error)
            print(f"❌ Error preparando alerta de {site.get('name')}: {exc}")

    if not site_alerts:
        result["skipped"] = True
        result["reason"] = "no_alerts_built"
        return result

    if channel == "email":
        try:
            email_result = enviar_email(
                user.get("email", ""),
                "Resumen diario de alertas Río Ebro",
                _build_email_body(user, site_alerts),
            )
            result["email_sent"] += email_result["sent"]
            result["sent"] += email_result["sent"]
        except Exception as exc:
            result["errors"].append(str(exc))
            print(f"❌ Error enviando email a {user.get('email')}: {exc}")

        return result

    push_tokens = _push_tokens_for_user(user)

    if not push_tokens:
        result["skipped"] = True
        result["reason"] = "no_push_tokens"
        return result

    for alert in site_alerts:
        push_result = enviar_notificacion(
            push_tokens,
            alert["titulo"],
            alert["mensaje"],
            tag=f"rio-ebro-alert-{alert['site_id']}",
        )
        result["push_sent"] += push_result["sent"]
        result["sent"] += push_result["sent"]
        result["invalid_tokens"].update(push_result["invalid_tokens"])
        result["errors"].extend(push_result.get("errors", []))

    return result


def enviar_alertas_usuarios(users: list[dict[str, Any]], sites: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "sent": 0,
        "push_sent": 0,
        "email_sent": 0,
        "invalid_tokens": set(),
        "processed_sites": 0,
        "processed_users": 0,
        "per_user": {},
        "errors": [],
    }

    alert_cache: dict[str, dict[str, Any]] = {}

    for user in users:
        user_result = enviar_alerta_usuario(user, sites, alert_cache=alert_cache)
        user_id = user.get("id")

        result["sent"] += user_result["sent"]
        result["push_sent"] += user_result["push_sent"]
        result["email_sent"] += user_result["email_sent"]
        result["processed_sites"] += user_result["processed_sites"]
        result["invalid_tokens"].update(user_result["invalid_tokens"])
        result["errors"].extend(user_result["errors"])

        if user_id:
            result["per_user"][user_id] = user_result

        if not user_result.get("skipped"):
            result["processed_users"] += 1

    return result


# =========================
# ALERTAS DIARIAS
# =========================
def enviar_alertas_diarias(tokens, sites, force=False):

    global ULTIMO_ENVIO

    ahora = _now_alert_tz()

    # =========================
    # SOLO UNA VEZ AL DÍA
    # =========================
    fecha_hoy = ahora.strftime("%Y-%m-%d")

    result = {
        "sent": 0,
        "invalid_tokens": set(),
        "processed_sites": 0,
        "skipped": False,
        "reason": None,
        "errors": [],
    }

    if ULTIMO_ENVIO == fecha_hoy:
        result["skipped"] = True
        result["reason"] = "already_sent_today"
        return result

    # =========================
    # SOLO A LA HORA CONFIGURADA
    # =========================
    if not force and (ahora.hour != ALERT_HOUR or ahora.minute != ALERT_MINUTE):
        result["skipped"] = True
        result["reason"] = f"outside_alert_time_{ALERT_HOUR:02d}_{ALERT_MINUTE:02d}"
        return result

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

            titulo = _titulo_por_nivel(nivel_alerta)

            # =========================
            # ENVIAR PUSH
            # =========================
            push_result = enviar_notificacion(
                tokens_municipio,
                titulo,
                mensaje,
                tag=f"rio-ebro-alert-{site_id}",
            )

            result["sent"] += push_result["sent"]
            result["invalid_tokens"].update(push_result["invalid_tokens"])
            result["errors"].extend(push_result.get("errors", []))
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
