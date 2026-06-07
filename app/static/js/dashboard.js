const CLIENT_DEBUG = localStorage.getItem("dashboardDebug") === "1";

function debugLog(...args) {
  if (CLIENT_DEBUG) {
    console.debug(...args);
  }
}

const connDot = document.getElementById("connDot");
    const connText = document.getElementById("connText");
    const siteSelect = document.getElementById("siteSelect");
    const titleEl = document.getElementById("title");

    const sourceEl = document.getElementById("source");
    const tsEl = document.getElementById("ts");
    const refEl = document.getElementById("ref");
    const isNewEl = document.getElementById("isNew");

    const nivelEl = document.getElementById("nivel");
    const caudalEl = document.getElementById("caudal");
    const tnEl = document.getElementById("tn");
    const tcEl = document.getElementById("tc");
    const nivelQuality = document.getElementById("nivelQuality");
    const caudalQuality = document.getElementById("caudalQuality");

    const estadoDot = document.getElementById("estadoDot");
    const estadoTxt = document.getElementById("estadoTxt");
    const saihErr = document.getElementById("saihErr");

    const aemetRef = document.getElementById("aemetRef");
    const aemetErr = document.getElementById("aemetErr");

    const mm6sum = document.getElementById("mm6sum");
    const mm24sum = document.getElementById("mm24sum");

    const p6 = document.getElementById("p6");
    const p24 = document.getElementById("p24");
    const barP6 = document.getElementById("barP6");
    const barP24 = document.getElementById("barP24");

    const rainDot = document.getElementById("rainDot");
    const rainState = document.getElementById("rainState");

    const iaRef = document.getElementById("iaRef");
    const iaErr = document.getElementById("iaErr");
    const iaState = document.getElementById("iaState");

    const NIVEL_WARN = 2.0;
    const NIVEL_BAD  = 3.0;

    const IA_WARN = 2.5;
    const IA_BAD = 3.5;

    let predChart = null;
    let ws = null;
    let currentSite = null;

    let firebaseToken = null;
    let notificationsInitialized = null;
    let lastFirebaseError = null;
    let currentUser = null;
    let lastPredData = null;
    let lastPredPending = false;
    let lastPredInterval = null;
    let emailConfig = null;

    const userPanel = document.getElementById("userPanel");
    const loginPanel = document.getElementById("loginPanel");
    const registerPanel = document.getElementById("registerPanel");
    const userBtn = document.getElementById("userBtn");
    const loginBtn = document.getElementById("loginBtn");
    const registerBtn = document.getElementById("registerBtn");
    const sessionState = document.getElementById("sessionState");
    const authLoggedOut = document.getElementById("authLoggedOut");
    const authLoggedIn = document.getElementById("authLoggedIn");
    const userSummary = document.getElementById("userSummary");
    const alertsPanel = document.getElementById("alertsPanel");
    const alertsAuthNotice = document.getElementById("alertsAuthNotice");
    const alertsControls = document.getElementById("alertsControls");
    const alertProfileState = document.getElementById("alertProfileState");
    const profileName = document.getElementById("profileName");
    const prefChannel = document.getElementById("prefChannel");
    const prefAlertTime = document.getElementById("prefAlertTime");
    const prefTheme = document.getElementById("prefTheme");
    const prefEmail = document.getElementById("prefEmail");
    const emailConfigNotice = document.getElementById("emailConfigNotice");
    const historyPanel = document.getElementById("historyPanel");
    const historySiteSelect = document.getElementById("historySiteSelect");
    const historyStartDate = document.getElementById("historyStartDate");
    const historyEndDate = document.getElementById("historyEndDate");
    const historyVariable = document.getElementById("historyVariable");
    const historyGranularity = document.getElementById("historyGranularity");
    const historyFormat = document.getElementById("historyFormat");
    const historyFileName = document.getElementById("historyFileName");
    const historyEndLabel = document.getElementById("historyEndLabel");
    const historyErr = document.getElementById("historyErr");
    const historyDownloadBtn = document.getElementById("historyDownloadBtn");
    const historyDownloadsList = document.getElementById("historyDownloadsList");
    const historyFilePreview = document.getElementById("historyFilePreview");
    const historyFilePreviewTitle = document.getElementById("historyFilePreviewTitle");
    const historyFilePreviewBody = document.getElementById("historyFilePreviewBody");
    const registerPassword = document.getElementById("registerPassword");
    const registerPasswordConfirm = document.getElementById("registerPasswordConfirm");
    const registerPasswordHelp = document.getElementById("registerPasswordHelp");
    const reliabilityErr = document.getElementById("reliabilityErr");
    const reliabilityNivelMae = document.getElementById("reliabilityNivelMae");
    const reliabilityCaudalMae = document.getElementById("reliabilityCaudalMae");
    const reliabilityNivelRmse = document.getElementById("reliabilityNivelRmse");
    const reliabilityCaudalRmse = document.getElementById("reliabilityCaudalRmse");
    const reliabilitySamples = document.getElementById("reliabilitySamples");
    const reliabilityLastValidation = document.getElementById("reliabilityLastValidation");
    const predChartLoading = document.getElementById("predChartLoading");
    const reliabilityChartLoading = document.getElementById("reliabilityChartLoading");
    const predIntervalInfo = document.getElementById("predIntervalInfo");
    let historyFilenameEdited = false;
    let serverHistoryDownloads = [];
    let historyDownloadsLoadedFor = null;
    let reliabilityChart = null;
    let reliabilityRequestId = 0;
    let reliabilityLoadedSite = null;
    const chartLoadingTimers = new Map();

    function setChartLoading(element, active, messages = "Cargando datos") {
      if (!element) {
        return;
      }

      const previous = chartLoadingTimers.get(element);
      if (previous) {
        clearInterval(previous);
        chartLoadingTimers.delete(element);
      }

      if (!active) {
        element.classList.remove("active");
        element.textContent = "";
        return;
      }

      const queue = Array.isArray(messages) ? messages : [messages];
      let index = 0;
      let dots = 0;

      const paint = () => {
        dots = (dots + 1) % 4;
        element.textContent = `${queue[index % queue.length]}${".".repeat(dots)}`;
        if (dots === 0) {
          index += 1;
        }
      };

      element.classList.add("active");
      paint();
      chartLoadingTimers.set(element, setInterval(paint, 650));
    }

    function setChartMessage(element, message) {
      if (!element) {
        return;
      }

      const previous = chartLoadingTimers.get(element);
      if (previous) {
        clearInterval(previous);
        chartLoadingTimers.delete(element);
      }

      element.classList.add("active");
      element.textContent = message;
    }

    setChartLoading(predChartLoading, true, [
      "Preparando prediccion",
      "Cargando modelo",
      "Dibujando grafica"
    ]);

    function togglePasswordVisibility(inputId, button) {
      const input = document.getElementById(inputId);

      if (!input) {
        return;
      }

      const show = input.type === "password";
      input.type = show ? "text" : "password";
      button?.classList.toggle("is-visible", show);
      button?.setAttribute("aria-label", show ? "Ocultar contraseña" : "Mostrar contraseña");
      button?.setAttribute("title", show ? "Ocultar contraseña" : "Mostrar contraseña");
    }

    window.togglePasswordVisibility = togglePasswordVisibility;

    function openModal(panel) {
      if (!panel) return;
      panel.style.display = "flex";
    }

    function closeModal(panelId) {
      const panel = document.getElementById(panelId);
      if (panel) {
        panel.style.display = "none";
      }
    }

    function openLoginPanel() {
      closeModal("registerPanel");
      closeModal("userPanel");
      closeModal("alertsPanel");
      closeModal("historyPanel");
      openModal(loginPanel);
      setTimeout(() => document.getElementById("loginEmail")?.focus(), 30);
    }

    function openRegisterPanel() {
      closeModal("loginPanel");
      closeModal("userPanel");
      closeModal("alertsPanel");
      closeModal("historyPanel");
      openModal(registerPanel);
      setTimeout(() => document.getElementById("registerName")?.focus(), 30);
    }

    function getPasswordValidation(password, confirmation) {
      const checks = {
        length: password.length >= 8,
        lower: /[a-záéíóúüñ]/.test(password),
        upper: /[A-ZÁÉÍÓÚÜÑ]/.test(password),
        number: /\d/.test(password),
        match: password.length > 0 && password === confirmation
      };

      return {
        ...checks,
        ok: checks.length && checks.lower && checks.upper && checks.number && checks.match
      };
    }

    function updateRegisterPasswordHelp() {
      if (!registerPassword || !registerPasswordConfirm || !registerPasswordHelp) {
        return;
      }

      const validation = getPasswordValidation(registerPassword.value, registerPasswordConfirm.value);
      const pending = [];

      if (!validation.length) pending.push("8 caracteres");
      if (!validation.upper) pending.push("una mayúscula");
      if (!validation.lower) pending.push("una minúscula");
      if (!validation.number) pending.push("un número");
      if (!validation.match && registerPasswordConfirm.value) pending.push("que ambas contraseñas coincidan");

      registerPasswordHelp.classList.toggle("ok", validation.ok);
      registerPasswordHelp.textContent = validation.ok
        ? "Contraseña segura y confirmada."
        : `Debe incluir ${pending.join(", ") || "confirmación de contraseña"}.`;
    }

    window.closeModal = closeModal;
    window.openLoginPanel = openLoginPanel;
    window.openRegisterPanel = openRegisterPanel;

    document.querySelectorAll(".modal-backdrop").forEach(panel => {
      panel.addEventListener("click", (event) => {
        if (event.target === panel) {
          panel.style.display = "none";
        }
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      ["loginPanel", "registerPanel", "userPanel", "alertsPanel", "historyPanel"]
        .forEach(closeModal);
    });

    async function getFirebaseMessagingSupported() {
      try {
        return firebase.messaging.isSupported
          ? await firebase.messaging.isSupported()
          : null;
      } catch (err) {
        return null;
      }
    }

    async function logPushDebug(event, extra = {}) {
      try {
        const selectedSites = getSelectedAlertSites();
        const selectedSitesCount = selectedSites.length;

        await fetch("/api/push-debug", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            event,
            href: location.href,
            userAgent: navigator.userAgent,
            permission: "Notification" in window ? Notification.permission : "unsupported",
            secureContext: window.isSecureContext,
            serviceWorker: "serviceWorker" in navigator,
            pushManager: "PushManager" in window,
            firebaseMessagingSupported: await getFirebaseMessagingSupported(),
            selectedSitesCount,
            selectedSites,
            ...extra
          })
        });
      } catch (err) {
        console.warn("No se pudo enviar push-debug", err);
      }
    }

    function getSelectedAlertSites() {
      return Array.from(
        document.querySelectorAll("#alertsSites input:checked")
      ).map(x => x.value);
    }

    function getChartTextColor() {
      return document.documentElement.dataset.theme === "light"
        ? "rgba(15,23,42,.82)"
        : "white";
    }

    function applyTheme(theme) {
      const normalized = theme === "light" ? "light" : "dark";
      document.documentElement.dataset.theme = normalized;
      localStorage.setItem("dashboardTheme", normalized);
      document.querySelector("meta[name='theme-color']")
        ?.setAttribute("content", normalized === "light" ? "#f6f8fb" : "#05070d");

      if (prefTheme) {
        prefTheme.value = normalized;
      }

      if (lastPredData) {
        updatePredChart(lastPredData, lastPredPending, lastPredInterval);
      }
    }

    function getUserPreferences() {
      return currentUser?.preferences || {
        notification_channel: "push",
        alert_time: "08:00",
        theme: localStorage.getItem("dashboardTheme") || "dark",
        sites: []
      };
    }

let sitesGlobal = []; // 🔥 guardar sitios para mapa

    async function loadSites() {
      const res = await fetch("/api/sites");
      const sites = await res.json();

      sitesGlobal = sites; // 🔥 guardar para usar en mapa

      // =========================
      // SELECTOR
      // =========================
      siteSelect.innerHTML = sites
        .map(s => `<option value="${s.id}">${s.name}</option>`)
        .join("");

      historySiteSelect.innerHTML = sites
        .map(s => `<option value="${s.id}">${s.name}</option>`)
        .join("");

      initHistoryForm();

      currentSite = sites[0]?.id ?? null;

      siteSelect.addEventListener("change", () => {
        currentSite = siteSelect.value;
        reliabilityLoadedSite = null;
        clearReliabilityChart();

        // 🔥 centrar mapa al seleccionar
        const s = sitesGlobal.find(x => x.id === currentSite);
        if (s && s.lat && s.lon) {
          map.setView([s.lat, s.lon], 10);
        }

        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "set_site", site: currentSite }));
        }
      });

      // =========================
      // 🔥 MAPA (crear marcadores iniciales)
      // =========================
      sites.forEach(s => {
        if (s.lat && s.lon) {
          updateMarker(s.id, s.lat, s.lon, s.name, null, null);
        }
      });

      // 🔥 centrar en el primero
      if (sites[0]?.lat && sites[0]?.lon) {
        map.setView([sites[0].lat, sites[0].lon], 7);
      }

      startWS();

      const alertsDiv = document.getElementById("alertsSites");

      alertsDiv.innerHTML = sites.map(s => `
        <label class="site-check">
          <input type="checkbox" value="${s.id}">
          ${s.name}
        </label>
      `).join("");

      // =========================
      // RESTAURAR MUNICIPIOS
      // =========================
      const savedSites = currentUser?.preferences?.sites || JSON.parse(
        localStorage.getItem("alertSites") || "[]"
      );

      document
        .querySelectorAll("#alertsSites input")
        .forEach(input => {

          if (savedSites.includes(input.value)) {
            input.checked = true;
          }
        });
      
      // =========================
      // AUTO GUARDAR CAMBIOS
      // =========================
      document
        .querySelectorAll("#alertsSites input")
        .forEach(input => {

          input.addEventListener("change", async () => {

            debugLog("Cambio en alertas");

            localStorage.setItem(
              "alertSites",
              JSON.stringify(getSelectedAlertSites())
            );
          });
        });

    }

    function setQualityBadge(element, code, label) {
      if (!element) {
        return;
      }

      const normalized = String(code || "api_no_data").trim() || "api_no_data";
      const labels = {
        real: "real",
        persisted: "persistido",
        last_valid: "último válido",
        api_no_data: "API sin datos"
      };
      const descriptions = {
        real: "Dato recibido en el último refresco desde SAIH Ebro.",
        persisted: "Dato cargado desde la base de datos porque ya estaba guardado por el sistema.",
        last_valid: "SAIH no ha devuelto un valor nuevo; se mantiene el último dato válido conocido.",
        api_no_data: "SAIH no devuelve un valor disponible para este municipio o señal."
      };

      element.className = `quality-badge ${normalized}`;
      element.textContent = label || labels[normalized] || "API sin datos";
      element.title = descriptions[normalized] || descriptions.api_no_data;
      element.setAttribute("aria-label", `${element.textContent}: ${element.title}`);
      element.setAttribute("tabindex", "0");
      element.dataset.tooltip = element.title;
    }

    function startWS() {
      const protocol = (location.protocol === "https:") ? "wss" : "ws";
      ws = new WebSocket(`${protocol}://${location.host}/ws`);

      ws.onopen = () => {
        setConn(true);
        if (currentSite) {
          ws.send(JSON.stringify({ type: "set_site", site: currentSite }));
        }
      };

      ws.onclose = () => setConn(false);
      ws.onerror = (e) => debugLog("WS error", e);

      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);

        if (data.site_name) {
          titleEl.textContent = `Dashboard Río Ebro – ${data.site_name}`;
        }

        sourceEl.textContent = data.source ?? "-";
        tsEl.textContent = data.ts ?? "-";
        refEl.textContent = data.refreshed_at ?? "-";
        isNewEl.textContent =
          (data.is_new === true) ? "SI" :
          (data.is_new === false) ? "NO" : "-";

        const nivel = toNum(data.nivel_m);
        const caudal = toNum(data.caudal_m3s);

        nivelEl.textContent = (nivel === null) ? "-" : fmt(nivel, 2);
        caudalEl.textContent = (caudal === null) ? "-" : fmt(caudal, 1);

        tnEl.textContent = formatTrend(data.tendencia_nivel);
        tcEl.textContent = formatTrend(data.tendencia_caudal);
        setQualityBadge(nivelQuality, data.nivel_quality, data.nivel_quality_label);
        setQualityBadge(caudalQuality, data.caudal_quality, data.caudal_quality_label);

        setEstadoRio(nivel);

        const saihError = data.saih_error ?? "";
        if (saihError) {
          saihErr.classList.add("info");
          saihErr.style.display = "block";
          saihErr.textContent = saihError;
          saihErr.title = data.saih_error_detail ?? "";
        } else {
          saihErr.classList.remove("info");
          saihErr.style.display = "none";
          saihErr.textContent = "";
          saihErr.title = "";
        }

        aemetRef.textContent = data.aemet_refreshed_at ?? "-";

        const err = data.aemet_error ?? "";
        if (err) {
          aemetErr.style.display = "block";
          aemetErr.textContent = "Error AEMET: no se pudo actualizar la prediccion. Se reintentara automaticamente.";
          aemetErr.title = err;
        } else {
          aemetErr.style.display = "none";
          aemetErr.textContent = "";
          aemetErr.title = "";
        }

        mm6sum.textContent = data.aemet_mm_6h_sum ?? "-";
        mm24sum.textContent = data.aemet_mm_24h_sum ?? "-";

        p6.textContent = data.aemet_prob_6h_max ?? "-";
        p24.textContent = data.aemet_prob_24h_max ?? "-";

        setProbBars(data.aemet_prob_6h_max, data.aemet_prob_24h_max);
        setRainState(data.aemet_mm_24h_sum, data.aemet_prob_24h_max);

        window._lastIaRef = data.ia_refreshed_at ?? "-";
        debugLog("IA PRED:", data.pred_semana);
        updateIAState(data.ia_error, data.pred_semana, data.ia_warning, data.pred_semana_source);
        updatePredChart(data.pred_semana, data.pred_semana_pending === true, data.prediction_interval);

        if (
          data.site_id &&
          data.site_id === currentSite &&
          data.site_id !== reliabilityLoadedSite
        ) {
          reliabilityLoadedSite = data.site_id;
          loadReliabilityChart(data.site_id);
        }

        // 🔥 ACTUALIZAR MAPA EN TIEMPO REAL
        if (data.site_id && data.lat && data.lon) {
          const nivel = toNum(data.nivel_m);
          const caudal = toNum(data.caudal_m3s);
          updateMarker(
            data.site_id,
            data.lat,
            data.lon,
            data.site_name || data.site_id,
            nivel,
            caudal
          );
        }
      };
    }

    // =========================
    // MAPA
    // =========================

    function refreshAllMarkers() {
      Object.keys(markers).forEach(id => {
        const m = markers[id];

        const nivelText = m.getPopup().getContent();
        const nivelMatch = nivelText.match(/Nivel: ([\d.]+)/);
        const nivel = nivelMatch ? parseFloat(nivelMatch[1]) : null;

        const latlng = m.getLatLng();
        const nombre = m.getPopup().getContent().split("<br>")[0].replace("<b>", "").replace("</b>", "");

        updateMarker(id, latlng.lat, latlng.lng, nombre, nivel, null);
      });
    }

    const map = L.map('map').setView([41.65, -0.88], 6);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    let markers = {}; // guardar marcadores por id

    function getColorNivel(nivel) {
      if (nivel === null || nivel === undefined) return "#999";
      if (nivel >= 3) return "#ef4444";   // rojo
      if (nivel >= 2) return "#f59e0b";   // naranja
      return "#22c55e";                   // verde
    }

    function updateMarker(siteId, lat, lon, nombre, nivel, caudal) {
      const color = getColorNivel(nivel);
      const isActive = (siteId === currentSite);

      const radius = isActive ? 12 : 8;
      const weight = isActive ? 3 : 1.5;

      const popupContent = `
        <b>${nombre}</b><br>
        Nivel: ${nivel ?? "-"} m<br>
        Caudal: ${caudal ?? "-"} m³/s<br>
        ${isActive ? "<span style='color:#22c55e'>● Seleccionado</span>" : ""}
      `;

      if (markers[siteId]) {
        // 🔄 actualizar marcador existente
        markers[siteId].setStyle({
          color: color,
          fillColor: color,
          radius: radius,
          weight: weight
        });

        markers[siteId].setPopupContent(popupContent);

      } else {
        // 🆕 crear marcador
        markers[siteId] = L.circleMarker([lat, lon], {
          radius: radius,
          color: color,
          fillColor: color,
          fillOpacity: 0.85,
          weight: weight
        })
        .addTo(map)
        .bindPopup(popupContent);

        // 🔥 CLICK EN MAPA → CAMBIAR MUNICIPIO
        markers[siteId].on("click", () => {
          currentSite = siteId;

          // sincronizar selector
          siteSelect.value = siteId;

          // centrar mapa
          map.setView([lat, lon], 10);

          // enviar al backend
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "set_site", site: siteId }));
          }

          // 🔄 refrescar todos los marcadores (para resaltar activo)
          refreshAllMarkers();
        });
      }
    }


    const sitesReady = loadSites();
    const currentUserReady = loadCurrentUser();
    const emailConfigReady = loadEmailConfig();

    historyStartDate.addEventListener("change", syncHistoryDateLimits);
    historyEndDate.addEventListener("change", syncHistoryDateLimits);
    historySiteSelect.addEventListener("change", () => updateHistoryFilenameSuggestion());
    historyVariable.addEventListener("change", () => updateHistoryFilenameSuggestion());
    historyGranularity.addEventListener("change", () => updateHistoryFilenameSuggestion());
    historyFormat.addEventListener("change", () => updateHistoryFilenameSuggestion({ force: true }));
    historyFileName.addEventListener("input", () => {
      historyFilenameEdited = true;
    });
    registerPassword?.addEventListener("input", updateRegisterPasswordHelp);
    registerPasswordConfirm?.addEventListener("input", updateRegisterPasswordHelp);
    prefTheme.addEventListener("change", () => applyTheme(prefTheme.value));
    prefChannel.addEventListener("change", () => {
      alertProfileState.textContent = prefChannel.value === "email"
        ? "Las alertas llegarán por correo"
        : "Las alertas llegarán por push";
      updateEmailConfigNotice();
    });
