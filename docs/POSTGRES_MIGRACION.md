# Migracion a PostgreSQL

## Como funciona

La aplicacion usa `USER_STORE_BACKEND=auto`.

- Si existe `DATABASE_URL` o `DB_NAME/DB_USER/DB_PASSWORD/DB_HOST/DB_PORT`, guarda usuarios, sesiones, dispositivos push y descargas en PostgreSQL.
- Si no existe `DATABASE_URL`, usa `app/users.json` como fallback local.
- En el primer arranque con PostgreSQL, `MIGRATE_USERS_JSON_ON_START=1` importa `USERS_FILE` si la base esta vacia.

## Tablas creadas automaticamente

- `app_users`: usuarios, preferencias y estado de alertas.
- `user_sessions`: sesiones persistentes.
- `push_devices`: tokens/dispositivos push.
- `download_records`: historial de descargas.

## Fly.io

1. Crear Postgres si aun no existe:
   `fly postgres create`
2. Adjuntarlo a la app:
   `fly postgres attach --app dashboard-ebro <nombre-postgres>`
3. Desplegar:
   `fly deploy`
4. Revisar logs:
   `fly logs`

Fly inyectara `DATABASE_URL` como secret. El volumen `/data` sigue siendo util como respaldo y para importar `users.json` en el primer arranque.

## SAIH Ebro

`SAIH_SSL_MODE=auto` prueba certificados normales y, si saihebro.com falla por cadena de certificados, usa fallback sin verificacion SSL solo para esa integracion. Si algun dia SAIH corrige la cadena, se puede endurecer con `SAIH_SSL_MODE=strict`.
