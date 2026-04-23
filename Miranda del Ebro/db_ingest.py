import psycopg2

# =========================
# CONFIG
# =========================
conn = psycopg2.connect(
    dbname="tfg_db",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)


def insertar_saih(fecha, municipio, nivel, caudal):
    cur = conn.cursor()

    query = """
    INSERT INTO mediciones_saih (fecha, municipio, nivel_m, caudal_m3s)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (fecha, municipio) DO NOTHING;
    """

    cur.execute(query, (fecha, municipio, nivel, caudal))

    conn.commit()
    cur.close()


def insertar_aemet(fecha, municipio, lluvia, prob):
    cur = conn.cursor()

    query = """
    INSERT INTO meteorologia_aemet (fecha, municipio, lluvia_mm, prob_precipitacion)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (fecha, municipio) DO NOTHING;
    """

    cur.execute(query, (fecha, municipio, lluvia, prob))

    conn.commit()
    cur.close()