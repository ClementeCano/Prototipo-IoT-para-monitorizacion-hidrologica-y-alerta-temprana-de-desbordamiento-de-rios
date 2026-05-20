# Informe de mejoras del sistema

## Resumen ejecutivo

El proyecto ya cubre el flujo principal: dashboard en tiempo real, descarga de historicos, usuarios, preferencias y alertas por email o push. Las mejoras prioritarias para convertirlo en un sistema mas fiable son la persistencia real de datos de usuario, el endurecimiento de integraciones externas y la reduccion de duplicidad en flujos criticos.

## Hallazgos principales

1. La persistencia estaba incompleta en despliegue: `tokens.json` apuntaba al volumen de Fly, pero `users.json` podia quedarse en el filesystem efimero del contenedor si `USERS_FILE` no estaba configurado.
2. El guardado de suscripciones push tenia logica duplicada entre el endpoint principal y el endpoint de prueba.
3. La API SAIH se consultaba con verificacion SSL desactivada por defecto.
4. Las alertas programadas dependen de procesos en memoria; para un TFG es aceptable, pero en produccion convendria evolucionar a una cola o scheduler persistente.
5. `users.json` y `tokens.json` son suficientes para prototipo/despliegue pequeno, pero el siguiente salto natural es SQLite o PostgreSQL.

## Cambios aplicados

- `DATA_DIR` se usa como base real para ficheros persistentes.
- Fly queda configurado para guardar usuarios y tokens en `/data`.
- `SESSION_COOKIE_SECURE=1` queda configurado en Fly para cookies solo por HTTPS.
- `app/tokens.json` queda ignorado para evitar versionar nuevos tokens de dispositivos. Como el fichero ya estaba versionado, conviene retirarlo del indice de Git con `git rm --cached app/tokens.json` antes del proximo commit si no quieres conservarlo en el repositorio.
- El endpoint de token push reutiliza el mismo helper de persistencia que el flujo de prueba.
- SAIH verifica SSL por defecto; se puede desactivar solo con `SAIH_VERIFY_SSL=0` si un entorno local lo necesita.

## Recomendaciones siguientes

1. Migrar usuarios, sesiones, dispositivos push y descargas a SQLite/PostgreSQL con migraciones.
2. Separar `app.py` en routers: `auth`, `alerts`, `history`, `realtime`, `diagnostics`.
3. Sustituir `print` por logging estructurado con niveles y trazas por job.
4. Anadir tests de integracion para registro, login, preferencias, descarga historica y alertas programadas.
5. Crear un endpoint de diagnostico privado para comprobar almacenamiento, Firebase, SMTP, AEMET y SAIH.
6. Externalizar los jobs de alerta a una cola/scheduler persistente si el sistema crece.

## Checklist de despliegue

- `fly deploy`
- Confirmar volumen montado en `/data`.
- Confirmar variables `DATA_DIR=/data`, `USERS_FILE=/data/users.json`, `TOKENS_FILE=/data/tokens.json`.
- Desde un movil, iniciar sesion y guardar alertas push una vez para registrar el token.
- Probar una alerta y revisar logs si `sent` no sube.

## Verificacion realizada

- Compilacion Python de `app.py`, `alertas.py`, `user_store.py` y `saih_opendata.py`.
- Revision de diff con `git diff --check`.
- Arranque local con Uvicorn en `http://127.0.0.1:8001`.
- Comprobacion HTTP de `/` y `/api/sites` con respuesta `200`.
