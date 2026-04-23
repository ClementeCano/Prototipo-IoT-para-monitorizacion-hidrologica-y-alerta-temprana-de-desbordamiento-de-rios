import psycopg2
import os
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


def insertar_saih(fecha, municipio, nivel, caudal):

    query = """
    INSERT INTO mediciones_saih (fecha, municipio, nivel_m, caudal_m3s)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (fecha, municipio) DO NOTHING;
    """
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (fecha, municipio, nivel, caudal))

        conn.commit()
        cur.close()


def insertar_aemet(fecha, municipio, lluvia, prob):
    
    query = """
    INSERT INTO meteorologia_aemet (fecha, municipio, lluvia_mm, prob_precipitacion)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (fecha, municipio) DO NOTHING;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (fecha, municipio, lluvia, prob))

        conn.commit()
        cur.close()