def cargar_umbrales(path="umbrales.txt"):
    umbrales = {}

    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            if "->" not in linea:
                continue

            nombre, valores = linea.split("->")

            nombre = nombre.strip().lower()
            nombre = nombre.replace(" ", "_").replace("-", "_")

            partes = [v.strip() for v in valores.split(",")]

            # limpiar "-"
            nums = []
            for p in partes:
                try:
                    nums.append(float(p))
                except:
                    nums.append(None)

            umbrales[nombre] = {
                "prealerta": nums[0],
                "alerta": nums[1],
                "alarma": nums[2]
            }

    return umbrales