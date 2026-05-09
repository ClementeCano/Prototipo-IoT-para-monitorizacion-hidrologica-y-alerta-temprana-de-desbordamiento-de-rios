def generar_alerta(nivel_previsto, umbrales):
    alerta, pre, emergencia = umbrales

    if nivel_previsto < alerta:
        return {
            "nivel": "bajo",
            "mensaje": "🟢 Riesgo bajo de desbordamiento"
        }
    elif nivel_previsto < pre:
        return {
            "nivel": "moderado",
            "mensaje": "🟡 Riesgo moderado (vigilancia)"
        }
    elif nivel_previsto < emergencia:
        return {
            "nivel": "alto",
            "mensaje": "🟠 Riesgo alto (pre-alerta)"
        }
    else:
        return {
            "nivel": "critico",
            "mensaje": "🔴 ALERTA ROJA: posible desbordamiento"
        }