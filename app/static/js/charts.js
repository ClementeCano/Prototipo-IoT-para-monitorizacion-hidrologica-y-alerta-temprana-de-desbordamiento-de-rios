function setConn(isUp) {
      connDot.className = "dot " + (isUp ? "ok" : "");
      connText.textContent = isUp ? "Conectado" : "Desconectado";
    }

    function fmt(x, decimals = 2) {
      if (x === null || x === undefined) return "-";
      const n = Number(x);
      if (Number.isNaN(n)) return "-";
      return n.toFixed(decimals);
    }

    function toNum(value) {
      if (value === null || value === undefined || value === "") {
        return null;
      }
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function setEstadoRio(nivel) {
      if (typeof nivel !== "number" || !Number.isFinite(nivel)) {
        estadoDot.className = "dot";
        estadoTxt.textContent = "-";
        return;
      }

      if (nivel >= NIVEL_BAD) {
        estadoDot.className = "dot bad";
        estadoTxt.textContent = "ALTO (visual)";
      } else if (nivel >= NIVEL_WARN) {
        estadoDot.className = "dot warn";
        estadoTxt.textContent = "MEDIO (visual)";
      } else {
        estadoDot.className = "dot ok";
        estadoTxt.textContent = "Normal";
      }
    }

    function clampProb(v) {
      if (v === null || v === undefined) return null;
      const n = Number(v);
      if (Number.isNaN(n)) return null;
      return Math.max(0, Math.min(100, n));
    }

    function setProbBars(v6, v24) {
      const c6 = clampProb(v6);
      const c24 = clampProb(v24);

      barP6.style.width = (c6 === null ? "0%" : `${c6}%`);
      barP24.style.width = (c24 === null ? "0%" : `${c24}%`);
    }

    function setRainState(mm24, p24) {
      const mm = Number(mm24);
      const prob = clampProb(p24);

      if (Number.isNaN(mm) && prob === null) {
        rainDot.className = "dot";
        rainState.textContent = "-";
        return;
      }

      if ((prob !== null && prob >= 70) || (!Number.isNaN(mm) && mm >= 10)) {
        rainDot.className = "dot bad";
        rainState.textContent = "Riesgo lluvia alto (visual)";
      } else if ((prob !== null && prob >= 40) || (!Number.isNaN(mm) && mm >= 3)) {
        rainDot.className = "dot warn";
        rainState.textContent = "Riesgo lluvia medio (visual)";
      } else {
        rainDot.className = "dot ok";
        rainState.textContent = "Riesgo lluvia bajo (visual)";
      }
    }

    function limpiarPred(pred) {
      if (!Array.isArray(pred)) return [];

      return pred
        .map((p) => {
          // 🔥 ahora es objeto, no array
          if (!p || typeof p !== "object") return null;

          const nivel = toNum(p.nivel);
          const caudal = toNum(p.caudal);

          if (nivel === null && caudal === null) return null;

          return [nivel, caudal];
        })
        .filter(Boolean);
    }

    function predChartLabels(pred, limpia) {
      if (!Array.isArray(pred)) {
        return limpia.map((_, i) => `Dia ${i + 1}`);
      }

      const labels = [];

      pred.forEach((p) => {
        if (!p || typeof p !== "object") return;

        const nivel = toNum(p.nivel);
        const caudal = toNum(p.caudal);

        if (nivel === null && caudal === null) return;

        labels.push(p.target_date || p.date || p.fecha || `Dia ${labels.length + 1}`);
      });

      return labels.length === limpia.length
        ? labels
        : limpia.map((_, i) => `Dia ${i + 1}`);
    }

    function evaluarRiesgoIA(pred) {
      const limpia = limpiarPred(pred);

      if (limpia.length === 0) {
        return { texto: "Sin datos", clase: "" };
      }

      let maxNivel = -Infinity;
      let diaMax = -1;

      limpia.forEach((p, i) => {
        const nivel = p[0];
        if (typeof nivel === "number" && nivel > maxNivel) {
          maxNivel = nivel;
          diaMax = i;
        }
      });

      if (!Number.isFinite(maxNivel)) {
        return { texto: "Sin datos", clase: "" };
      }

      if (maxNivel >= IA_BAD) {
        return {
          texto: `Riesgo alto en ${diaMax + 1} días`,
          clase: "bad"
        };
      }

      if (maxNivel >= IA_WARN) {
        return {
          texto: `Posible crecida en ${diaMax + 1} días`,
          clase: "warn"
        };
      }

      return {
        texto: "Sin riesgo relevante",
        clase: "ok"
      };
    }

    function colorNivel(v) {
      if (v === null || v === undefined || !Number.isFinite(v)) {
        return "rgba(255,255,255,.35)";
      }
      if (v >= IA_BAD) return "#ef4444";
      if (v >= IA_WARN) return "#f59e0b";
      return "#22c55e";
    }

    function updatePredChart(pred, pending = false) {
      lastPredData = pred;
      lastPredPending = pending;
      const canvas = document.getElementById("predChart");
      if (!canvas) return;

      const limpia = limpiarPred(pred);

      if (limpia.length === 0) {
        // El primer payload del WebSocket puede llegar antes de que termine la IA.
        // Si ya hay una gráfica válida, la mantenemos hasta recibir datos nuevos.
        if (!predChart) {
          if (pending) {
            setChartLoading(predChartLoading, true, [
              "Generando prediccion",
              "Cargando modelo",
              "Preparando grafica"
            ]);
          } else {
            setChartMessage(predChartLoading, "Prediccion aun no disponible");
          }
        }
        return;
      }

      setChartLoading(predChartLoading, false);
      const labels = limpia.map((_, i) => `Día ${i+1}`);

      const nivel = limpia.map(p => p[0]);
      const caudal = limpia.map(p => p[1]);

      // 🔥 FILTRO DE VALORES EXTREMOS (MUY IMPORTANTE)
      function clampArray(arr, min, max) {
        return arr.map(v => {
          if (v === null) return null;
          return Math.max(min, Math.min(max, v));
        });
      }

      const nivelClean = clampArray(nivel, 0, 10);     // niveles reales
      const caudalClean = clampArray(caudal, 0, 2000); // ajusta a tu río

      // 🔥 ESCALAS DINÁMICAS
      const finiteNivel = nivelClean.filter(v => Number.isFinite(v));
      const finiteCaudal = caudalClean.filter(v => Number.isFinite(v));
      const maxNivel = finiteNivel.length ? Math.max(...finiteNivel) : 1;
      const maxCaudal = finiteCaudal.length ? Math.max(...finiteCaudal) : 1;

      if (predChart) predChart.destroy();

      const chartTextColor = getChartTextColor();

      predChart = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Nivel (m)",
              data: nivelClean,
              borderWidth: 3,
              tension: 0.35,
              pointRadius: 5,
              borderColor: "#22c55e",
              yAxisID: "yNivel"
            },
            {
              label: "Caudal (m³/s)",
              data: caudalClean,
              borderWidth: 2,
              tension: 0.35,
              borderDash: [6,6],
              borderColor: "#7c3aed",
              yAxisID: "yCaudal"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,

          plugins: {
            legend: {
              labels: { color: chartTextColor }
            }
          },

          scales: {
            yNivel: {
              type: "linear",
              position: "left",
              min: 0,
              max: Math.ceil(maxNivel + 0.5),
              ticks: {
                color: chartTextColor,
                callback: v => v.toFixed(1)
              },
              title: {
                display: true,
                text: "Nivel (m)",
                color: chartTextColor
              }
            },

            yCaudal: {
              type: "linear",
              position: "right",
              min: 0,
              max: Math.ceil(maxCaudal * 1.2),
              ticks: {
                color: chartTextColor,
                callback: v => Math.round(v)
              },
              grid: { drawOnChartArea: false },
              title: {
                display: true,
                text: "Caudal (m³/s)",
                color: chartTextColor
              }
            }
          }
        }
      });
    }

    updatePredChart = function(pred, pending = false, interval = null) {
      lastPredData = pred;
      lastPredPending = pending;
      lastPredInterval = interval;
      const canvas = document.getElementById("predChart");
      if (!canvas) return;

      const limpia = limpiarPred(pred);

      if (limpia.length === 0) {
        if (predIntervalInfo) {
          predIntervalInfo.classList.remove("active");
          predIntervalInfo.textContent = "";
        }
        if (!predChart) {
          if (pending) {
            setChartLoading(predChartLoading, true, [
              "Generando prediccion",
              "Cargando modelo",
              "Preparando grafica"
            ]);
          } else {
            setChartMessage(predChartLoading, "Prediccion aun no disponible");
          }
        }
        return;
      }

      setChartLoading(predChartLoading, false);
      const labels = predChartLabels(pred, limpia);
      const nivel = limpia.map(p => p[0]);
      const caudal = limpia.map(p => p[1]);
      const hasPersistedD1Dates = Array.isArray(pred)
        && pred.some(point => point && typeof point === "object" && (point.target_date || point.date || point.fecha));
      const nivelMae = toNum(interval?.nivel_mae);
      const caudalMae = toNum(interval?.caudal_mae);
      const intervalSamples = Number(interval?.samples || 0);
      const hasNivelBand = Number.isFinite(nivelMae) && nivelMae > 0;
      const hasCaudalBand = Number.isFinite(caudalMae) && caudalMae > 0;

      function clampArray(arr, min, max) {
        return arr.map(v => {
          if (v === null) return null;
          return Math.max(min, Math.min(max, v));
        });
      }

      const nivelClean = clampArray(nivel, 0, 10);
      const caudalClean = clampArray(caudal, 0, 2000);
      const nivelLower = hasNivelBand ? clampArray(nivel.map(v => v === null ? null : v - nivelMae), 0, 10) : [];
      const nivelUpper = hasNivelBand ? clampArray(nivel.map(v => v === null ? null : v + nivelMae), 0, 10) : [];
      const caudalLower = hasCaudalBand ? clampArray(caudal.map(v => v === null ? null : v - caudalMae), 0, 2000) : [];
      const caudalUpper = hasCaudalBand ? clampArray(caudal.map(v => v === null ? null : v + caudalMae), 0, 2000) : [];
      const finiteNivel = [...nivelClean, ...nivelUpper].filter(v => Number.isFinite(v));
      const finiteCaudal = [...caudalClean, ...caudalUpper].filter(v => Number.isFinite(v));
      const maxNivel = finiteNivel.length ? Math.max(...finiteNivel) : 1;
      const maxCaudal = finiteCaudal.length ? Math.max(...finiteCaudal) : 1;

      if (predIntervalInfo) {
        predIntervalInfo.classList.add("active");
        const sourceText = hasPersistedD1Dates
          ? "Mostrando ultimas predicciones D+1 guardadas porque no se ha podido regenerar una prediccion actual. "
          : "";
        if (hasNivelBand || hasCaudalBand) {
          const nivelTxt = hasNivelBand ? `nivel +/- ${fmt(nivelMae, 3)} m` : "nivel sin banda";
          const caudalTxt = hasCaudalBand ? `caudal +/- ${fmt(caudalMae, 2)} m3/s` : "caudal sin banda";
          predIntervalInfo.textContent = `${sourceText}Bandas de error historico (${intervalSamples || "-"} muestras): ${nivelTxt}; ${caudalTxt}.`;
        } else {
          predIntervalInfo.textContent = `${sourceText}Aun no hay suficientes puntos reales comparados para calcular bandas de error.`;
        }
      }

      if (predChart) predChart.destroy();

      const chartTextColor = getChartTextColor();
      const datasets = [];

      if (hasNivelBand) {
        datasets.push(
          {
            label: "Banda nivel inferior",
            data: nivelLower,
            borderWidth: 0,
            pointRadius: 0,
            borderColor: "rgba(34,197,94,0)",
            backgroundColor: "rgba(34,197,94,.12)",
            yAxisID: "yNivel",
            isBand: true
          },
          {
            label: "Banda nivel",
            data: nivelUpper,
            borderWidth: 0,
            pointRadius: 0,
            borderColor: "rgba(34,197,94,0)",
            backgroundColor: "rgba(34,197,94,.12)",
            fill: "-1",
            yAxisID: "yNivel",
            isBand: true
          }
        );
      }

      datasets.push({
        label: "Nivel (m)",
        data: nivelClean,
        borderWidth: 3,
        tension: 0.35,
        pointRadius: 5,
        borderColor: "#22c55e",
        yAxisID: "yNivel"
      });

      if (hasCaudalBand) {
        datasets.push(
          {
            label: "Banda caudal inferior",
            data: caudalLower,
            borderWidth: 0,
            pointRadius: 0,
            borderColor: "rgba(124,58,237,0)",
            backgroundColor: "rgba(124,58,237,.10)",
            yAxisID: "yCaudal",
            isBand: true
          },
          {
            label: "Banda caudal",
            data: caudalUpper,
            borderWidth: 0,
            pointRadius: 0,
            borderColor: "rgba(124,58,237,0)",
            backgroundColor: "rgba(124,58,237,.10)",
            fill: "-1",
            yAxisID: "yCaudal",
            isBand: true
          }
        );
      }

      datasets.push({
        label: "Caudal (m3/s)",
        data: caudalClean,
        borderWidth: 2,
        tension: 0.35,
        borderDash: [6, 6],
        borderColor: "#7c3aed",
        yAxisID: "yCaudal"
      });

      predChart = new Chart(canvas, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: {
                color: chartTextColor,
                filter: item => !datasets[item.datasetIndex]?.isBand
              }
            },
            tooltip: {
              mode: "index",
              intersect: false,
              filter: item => !item.dataset.isBand
            }
          },
          interaction: {
            mode: "index",
            intersect: false
          },
          scales: {
            yNivel: {
              type: "linear",
              position: "left",
              min: 0,
              max: Math.ceil(maxNivel + 0.5),
              ticks: {
                color: chartTextColor,
                callback: v => v.toFixed(1)
              },
              title: {
                display: true,
                text: "Nivel (m)",
                color: chartTextColor
              }
            },
            yCaudal: {
              type: "linear",
              position: "right",
              min: 0,
              max: Math.ceil(maxCaudal * 1.2),
              ticks: {
                color: chartTextColor,
                callback: v => Math.round(v)
              },
              grid: { drawOnChartArea: false },
              title: {
                display: true,
                text: "Caudal (m3/s)",
                color: chartTextColor
              }
            }
          }
        }
      });
    };

    function clearReliabilityChart(message = "") {
      setChartLoading(reliabilityChartLoading, false);

      if (reliabilityChart) {
        reliabilityChart.destroy();
        reliabilityChart = null;
      }

      reliabilityNivelMae.textContent = "-";
      reliabilityCaudalMae.textContent = "-";
      reliabilityNivelRmse.textContent = "-";
      reliabilityCaudalRmse.textContent = "-";
      reliabilitySamples.textContent = "-";
      reliabilityLastValidation.textContent = "-";
      reliabilityErr.classList.remove("info");

      if (message) {
        reliabilityErr.style.display = "block";
        reliabilityErr.textContent = message;
      } else {
        reliabilityErr.style.display = "none";
        reliabilityErr.textContent = "";
      }
      reliabilityErr.title = "";
    }

    function updateReliabilityChart(data) {
      const canvas = document.getElementById("reliabilityChart");
      if (!canvas) return;

      const points = Array.isArray(data?.points) ? data.points : [];

      if (!points.length) {
        const update = data?.update || {};
        let message = "Aun no hay predicciones D+1 guardadas para comparar. Se guardaran automaticamente cada dia.";

        if (data?.error) {
          message = `No hay validacion disponible: ${data.error}`;
        } else if (update.error) {
          message = "No se han podido leer las medias diarias guardadas para completar la comparativa. Se reintentara automaticamente.";
        } else if (update.skipped === "saih_rate_limited") {
          message = `SAIH Ebro ha limitado temporalmente las peticiones. Se reintentara automaticamente en ${update.next_check_seconds || "unos"} segundos.`;
        } else if (update.skipped === "recently_checked") {
          message = `Ya hay predicciones D+1 guardadas y las medias diarias se han revisado hace poco. Se reintentara en ${update.next_check_seconds || "unos"} segundos.`;
        } else if (data?.total > 0) {
          message = "Hay predicciones D+1 guardadas, pero aun no existe una media diaria real para ese dia.";
        }

        clearReliabilityChart(message);
        reliabilityErr.classList.add("info");
        reliabilityErr.title = update.error || update.skipped || "";
        return;
      }

      setChartLoading(reliabilityChartLoading, false);
      if (data.update?.error) {
        reliabilityErr.classList.add("info");
        reliabilityErr.style.display = "block";
        reliabilityErr.textContent = "Mostrando la comparativa guardada. Las medias diarias reales no se han podido actualizar ahora mismo.";
        reliabilityErr.title = data.update.error;
      } else if ((data.pending || 0) > 0) {
        reliabilityErr.classList.add("info");
        reliabilityErr.style.display = "block";
        reliabilityErr.textContent = "Mostrando predicciones guardadas. Los valores reales apareceran cuando exista la media diaria persistida.";
        reliabilityErr.title = data.update?.skipped || "";
      } else {
        reliabilityErr.classList.remove("info");
        reliabilityErr.style.display = "none";
        reliabilityErr.textContent = "";
        reliabilityErr.title = "";
      }
      reliabilityNivelMae.textContent = data.metrics?.nivel_mae ?? "-";
      reliabilityCaudalMae.textContent = data.metrics?.caudal_mae ?? "-";
      reliabilityNivelRmse.textContent = data.metrics?.nivel_rmse ?? "-";
      reliabilityCaudalRmse.textContent = data.metrics?.caudal_rmse ?? "-";
      reliabilitySamples.textContent = data.metrics?.samples ?? "-";
      reliabilityLastValidation.textContent = data.metrics?.last_validation_date ?? "-";

      const labels = points.map(point => {
        const day = point.target_date || point.date || "-";
        return day;
      });
      const chartTextColor = getChartTextColor();

      if (reliabilityChart) {
        reliabilityChart.destroy();
      }

      reliabilityChart = new Chart(canvas, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Nivel real (m)",
              data: points.map(point => point.nivel_real),
              borderWidth: 3,
              tension: 0.25,
              pointRadius: 4,
              borderColor: "#22c55e",
              yAxisID: "yNivel"
            },
            {
              label: "Nivel predicho (m)",
              data: points.map(point => point.nivel_pred),
              borderWidth: 2,
              borderDash: [7, 5],
              tension: 0.25,
              pointRadius: 4,
              borderColor: "#38bdf8",
              yAxisID: "yNivel"
            },
            {
              label: "Caudal real (m³/s)",
              data: points.map(point => point.caudal_real),
              borderWidth: 3,
              tension: 0.25,
              pointRadius: 4,
              borderColor: "#f59e0b",
              yAxisID: "yCaudal"
            },
            {
              label: "Caudal predicho (m³/s)",
              data: points.map(point => point.caudal_pred),
              borderWidth: 2,
              borderDash: [7, 5],
              tension: 0.25,
              pointRadius: 4,
              borderColor: "#a78bfa",
              yAxisID: "yCaudal"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: chartTextColor }
            },
            tooltip: {
              mode: "index",
              intersect: false
            }
          },
          interaction: {
            mode: "index",
            intersect: false
          },
          scales: {
            x: {
              ticks: { color: chartTextColor },
              grid: { color: "rgba(148,163,184,.12)" }
            },
            yNivel: {
              type: "linear",
              position: "left",
              ticks: { color: chartTextColor },
              title: {
                display: true,
                text: "Nivel (m)",
                color: chartTextColor
              }
            },
            yCaudal: {
              type: "linear",
              position: "right",
              ticks: { color: chartTextColor },
              grid: { drawOnChartArea: false },
              title: {
                display: true,
                text: "Caudal (m³/s)",
                color: chartTextColor
              }
            }
          }
        }
      });
    }

    async function loadReliabilityChart(siteId) {
      if (!siteId) {
        clearReliabilityChart();
        return;
      }

      const requestId = ++reliabilityRequestId;
      clearReliabilityChart();
      setChartLoading(reliabilityChartLoading, true, [
        "Buscando predicciones D+1",
        "Leyendo medias diarias reales",
        "Calculando error medio"
      ]);
      reliabilityErr.classList.add("info");
      reliabilityErr.style.display = "block";
      reliabilityErr.textContent = "Actualizando comparativa con datos reales guardados...";

      try {
        const data = await apiJson(`/api/prediction/reliability/${encodeURIComponent(siteId)}`);
        if (requestId !== reliabilityRequestId) {
          return;
        }
        updateReliabilityChart(data);
      } catch (err) {
        if (requestId !== reliabilityRequestId) {
          return;
        }
        clearReliabilityChart(err.message || "No se pudo cargar la fiabilidad del modelo.");
      }
    }

    function formatIaError(error) {
      const text = String(error || "");

      if (text.includes("prediction_recent_saih_unavailable")) {
        return "No se puede generar una predicción actual porque SAIH Ebro no está devolviendo suficientes datos recientes. No es un fallo de la web.";
      }

      if (text.includes("429") || text.toLowerCase().includes("too many requests")) {
        return "SAIH Ebro ha limitado temporalmente las peticiones. No es un fallo de la web; prueba de nuevo en unos minutos.";
      }

      if (text.includes("prediction_model_error") || text.toLowerCase().includes("tensorflow")) {
        return "El motor de prediccion no esta disponible en este entorno. Revisa que TensorFlow y los modelos esten instalados en el despliegue.";
      }

      return `Error IA: ${text}`;
    }

    function formatTrend(value) {
      const text = String(value ?? "").trim();
      const normalized = text.toLowerCase();
      const trends = {
        "abajo": "bajada",
        "derecha": "estable",
        "arriba": "subida"
      };

      return trends[normalized] || text || "-";
    }

    function updateIAState(error, pred, warning = null, source = null) {
      iaRef.textContent = window._lastIaRef ?? "-";
      iaErr.classList.remove("info");

      if (error) {
        setChartLoading(predChartLoading, false);
        iaErr.style.display = "block";
        iaErr.textContent = formatIaError(error);
        iaState.textContent = "Error";
        return;
      }

      iaErr.style.display = "none";
      iaErr.textContent = "";

      const limpia = limpiarPred(pred);

      if (warning && limpia.length > 0) {
        iaErr.classList.add("info");
        iaErr.style.display = "block";
        iaErr.textContent = `Mostrando prediccion guardada. ${formatIaError(warning)}`;
      }

      if (limpia.length > 0) {
        const riesgo = evaluarRiesgoIA(limpia);

        if (source === "persisted" || source === "persisted_d1") {
          iaState.textContent = "Guardada";
        } else if (riesgo.clase === "bad") {
          iaState.textContent = "Riesgo alto";
        } else if (riesgo.clase === "warn") {
          iaState.textContent = "Vigilancia";
        } else if (riesgo.clase === "ok") {
          iaState.textContent = "Estable";
        } else {
          iaState.textContent = "Disponible";
        }
      } else {
        iaState.textContent = predChart ? "Sin datos" : "Calculando";
      }
    }
