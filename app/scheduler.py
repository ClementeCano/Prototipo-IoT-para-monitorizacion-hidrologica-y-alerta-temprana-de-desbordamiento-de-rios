from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime

from db_ingest import insertar_saih, insertar_aemet
from api.saih_opendata import fetch_saih_signals
from api.aemet_opendata import (
    fetch_aemet_municipio_horaria,
    extract_prob_precip_summary,
    extract_rain_forecast_mm
)
from core.config import collect_all_tags, SITES


def job():
    print("📡 Actualizando datos...")

    # 🔥 MISMA FECHA PARA TODO EL JOB
    fecha_actual = datetime.now().replace(minute=0, second=0, microsecond=0)

    # =========================
    # 🔹 SAIH
    # =========================
    try:
        tags = collect_all_tags()
        datos_saih = fetch_saih_signals(tags)

        for site in SITES:
            municipio = site["name"]

            nivel_tag = (site.get("saih") or {}).get("nivel")
            caudal_tag = (site.get("saih") or {}).get("caudal")

            nivel = datos_saih.get(nivel_tag, {}).get("valor") if nivel_tag else None
            caudal = datos_saih.get(caudal_tag, {}).get("valor") if caudal_tag else None

            if nivel is None and caudal is None:
                continue

            insertar_saih(
                fecha_actual,
                municipio,
                nivel,
                caudal
            )

            time.sleep(2)  # Pequeña pausa para no saturar la API

    except Exception as e:
        print("❌ Error en SAIH:", e)

    # =========================
    # 🔹 AEMET
    # =========================
    for site in SITES:
        municipio = site["name"]
        muni = site.get("aemet_muni")

        if not muni:
            continue

        try:
            data = fetch_aemet_municipio_horaria(muni)

            mm = extract_rain_forecast_mm(data)
            prob = extract_prob_precip_summary(data)

            insertar_aemet(
                fecha_actual,
                municipio,
                mm.get("aemet_mm_24h_sum", 0.0),
                prob.get("aemet_prob_24h_max", 0.0)
            )

        except Exception as e:
            print(f"❌ Error AEMET en {municipio}:", e)


# scheduler = BackgroundScheduler()
# scheduler.add_job(job, 'interval', hours=1)
# scheduler.start()


def main():
    job()
    print("✅ Job finalizado correctamente")


if __name__ == "__main__":
    main()