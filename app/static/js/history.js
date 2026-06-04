function isoDateLocal(date) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    }

    function getHistoryEndDate() {
      const end = new Date();
      end.setDate(end.getDate() - 1);
      return isoDateLocal(end);
    }

    function initHistoryForm() {
      const endDate = getHistoryEndDate();
      historyStartDate.max = endDate;
      historyEndDate.max = endDate;
      historyEndLabel.textContent = endDate;

      if (!historyStartDate.value) {
        historyStartDate.value = endDate;
      }

      if (!historyEndDate.value || historyEndDate.value > endDate) {
        historyEndDate.value = endDate;
      }

      syncHistoryDateLimits();
      updateHistoryFilenameSuggestion();
      renderHistoryDownloads();
    }

    function syncHistoryDateLimits() {
      const maxEndDate = getHistoryEndDate();

      historyStartDate.max = maxEndDate;
      historyEndDate.max = maxEndDate;
      historyEndLabel.textContent = maxEndDate;

      if (historyStartDate.value && historyStartDate.value > maxEndDate) {
        historyStartDate.value = maxEndDate;
      }

      if (historyEndDate.value && historyEndDate.value > maxEndDate) {
        historyEndDate.value = maxEndDate;
      }

      historyEndDate.min = historyStartDate.value || "";

      if (
        historyStartDate.value &&
        historyEndDate.value &&
        historyEndDate.value < historyStartDate.value
      ) {
        historyEndDate.value = historyStartDate.value;
      }

      updateHistoryFilenameSuggestion();
    }

    function toggleHistoryPanel() {
      if (!currentUser) {
        historyPanel.style.display = "none";
        openLoginPanel();
        alert("Inicia sesión para descargar históricos.");
        return;
      }

      historyPanel.style.display = historyPanel.style.display === "none" ? "flex" : "none";
      initHistoryForm();
    }

    function setHistoryError(message, type = "error") {
      if (!message) {
        historyErr.style.display = "none";
        historyErr.textContent = "";
        historyErr.classList.remove("info");
        return;
      }

      historyErr.classList.toggle("info", type === "info");
      historyErr.style.display = "block";
      historyErr.textContent = message;
    }

    function historyDownloadErrorMessage(errorData, status) {
      if (errorData?.message) {
        return errorData.message;
      }

      if (errorData?.error === "saih_history_error") {
        return "La API de SAIH Ebro no ha respondido correctamente. No es un fallo de la web; prueba mas tarde o cambia el tramo de fechas.";
      }

      if (errorData?.error === "history_without_data") {
        return "SAIH Ebro no tiene datos publicados para ese municipio, variable o tramo de fechas. No es un fallo de la web.";
      }

      if (errorData?.error === "site_without_requested_signals") {
        return "Ese municipio no tiene senales SAIH para el dato solicitado. No es un fallo de la web.";
      }

      return errorData?.error || `HTTP ${status}`;
    }

    function filenameFromDisposition(disposition, fallback) {
      if (!disposition) {
        return fallback;
      }

      const match = disposition.match(/filename="?([^";]+)"?/i);
      return match ? match[1] : fallback;
    }

    function slugifyDownloadPart(value) {
      return String(value || "")
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "") || "historico";
    }

    function historyExtension() {
      return historyFormat.value === "csv" ? "csv" : "xlsx";
    }

    function selectedHistorySiteName() {
      return historySiteSelect.options[historySiteSelect.selectedIndex]?.text || historySiteSelect.value;
    }

    function buildHistorySuggestedFilename() {
      const site = slugifyDownloadPart(selectedHistorySiteName());
      const variable = slugifyDownloadPart(historyVariable.value);
      const granularity = slugifyDownloadPart(historyGranularity.value);
      const startDate = historyStartDate.value || getHistoryEndDate();
      const endDate = historyEndDate.value || startDate;

      return `historico_${site}_${variable}_${granularity}_${startDate}_${endDate}.${historyExtension()}`;
    }

    function normalizeHistoryFilename(value) {
      const extension = historyExtension();
      const cleanName = String(value || buildHistorySuggestedFilename())
        .trim()
        .replace(/[\\/:*?"<>|]+/g, "_")
        .replace(/\s+/g, " ");

      if (!cleanName) {
        return buildHistorySuggestedFilename();
      }

      return cleanName.toLowerCase().endsWith(`.${extension}`)
        ? cleanName
        : `${cleanName.replace(/\.(csv|xlsx)$/i, "")}.${extension}`;
    }

    function updateHistoryFilenameSuggestion({ force = false } = {}) {
      if (!historyFileName) {
        return;
      }

      if (force || !historyFilenameEdited || !historyFileName.value.trim()) {
        historyFileName.value = buildHistorySuggestedFilename();
        historyFilenameEdited = false;
      }
    }

    function historyDownloadsKey() {
      return currentUser ? `historyDownloads:${currentUser.id}` : "historyDownloads";
    }

    function createDownloadId() {
      if (crypto.randomUUID) {
        return crypto.randomUUID();
      }

      return `download_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    }

    function openHistoryHandlesDb() {
      return new Promise((resolve, reject) => {
        if (!window.indexedDB) {
          reject(new Error("indexeddb_unavailable"));
          return;
        }

        const request = indexedDB.open("rio-history-downloads", 1);

        request.onupgradeneeded = () => {
          request.result.createObjectStore("handles");
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("indexeddb_open_failed"));
      });
    }

    async function saveHistoryFileHandle(downloadId, handle) {
      if (!downloadId || !handle) {
        return false;
      }

      try {
        const db = await openHistoryHandlesDb();

        await new Promise((resolve, reject) => {
          const tx = db.transaction("handles", "readwrite");
          tx.objectStore("handles").put(handle, downloadId);
          tx.oncomplete = resolve;
          tx.onerror = () => reject(tx.error || new Error("indexeddb_save_failed"));
        });

        db.close();
        return true;
      } catch (err) {
        console.warn("No se pudo guardar la referencia local del archivo:", err);
        return false;
      }
    }

    async function getHistoryFileHandle(downloadId) {
      if (!downloadId) {
        return null;
      }

      try {
        const db = await openHistoryHandlesDb();
        const handle = await new Promise((resolve, reject) => {
          const tx = db.transaction("handles", "readonly");
          const request = tx.objectStore("handles").get(downloadId);
          request.onsuccess = () => resolve(request.result || null);
          request.onerror = () => reject(request.error || new Error("indexeddb_read_failed"));
        });
        db.close();
        return handle;
      } catch (err) {
        console.warn("No se pudo recuperar la referencia local del archivo:", err);
        return null;
      }
    }

    async function ensureHistoryFilePermission(handle) {
      if (!handle || !handle.queryPermission || !handle.requestPermission) {
        return true;
      }

      const options = { mode: "read" };

      if (await handle.queryPermission(options) === "granted") {
        return true;
      }

      return await handle.requestPermission(options) === "granted";
    }

    function readLocalHistoryDownloads() {
      try {
        let changed = false;
        const downloads = JSON.parse(localStorage.getItem(historyDownloadsKey()) || "[]")
          .map(download => {
            if (download && !download.id) {
              changed = true;
              return {
                ...download,
                id: createDownloadId(),
                hasLocalHandle: false
              };
            }

            return download;
          });

        if (changed) {
          localStorage.setItem(historyDownloadsKey(), JSON.stringify(downloads));
        }

        return downloads;
      } catch (err) {
        return [];
      }
    }

    function writeLocalHistoryDownloads(downloads) {
      localStorage.setItem(historyDownloadsKey(), JSON.stringify(downloads.slice(0, 100)));
    }

    function mergeHistoryDownloads(localDownloads, remoteDownloads) {
      const merged = new Map();

      (remoteDownloads || []).forEach(download => {
        if (download?.id) {
          merged.set(download.id, {
            ...download,
            hasLocalHandle: false
          });
        }
      });

      (localDownloads || []).forEach(download => {
        if (download?.id) {
          merged.set(download.id, {
            ...(merged.get(download.id) || {}),
            ...download
          });
        }
      });

      return Array.from(merged.values())
        .sort((a, b) => String(b.downloadedAt || "").localeCompare(String(a.downloadedAt || "")));
    }

    function readHistoryDownloads() {
      return mergeHistoryDownloads(readLocalHistoryDownloads(), serverHistoryDownloads);
    }

    async function syncHistoryDownloadToServer(download) {
      if (!currentUser || !download?.id) {
        return null;
      }

      try {
        const data = await apiJson("/api/users/downloads", {
          method: "POST",
          body: JSON.stringify(download)
        });

        if (data.download) {
          serverHistoryDownloads = mergeHistoryDownloads([data.download], serverHistoryDownloads);
        }

        return data.download || null;
      } catch (err) {
        console.warn("No se pudo sincronizar la descarga con el servidor:", err);
        return null;
      }
    }

    async function loadHistoryDownloads() {
      if (!currentUser) {
        serverHistoryDownloads = [];
        historyDownloadsLoadedFor = null;
        renderHistoryDownloads();
        return;
      }

      if (historyDownloadsLoadedFor === currentUser.id) {
        return;
      }

      historyDownloadsLoadedFor = currentUser.id;

      try {
        const data = await apiJson("/api/users/downloads");
        serverHistoryDownloads = Array.isArray(data.downloads) ? data.downloads : [];
      } catch (err) {
        console.warn("No se pudo cargar el historial del servidor:", err);
        serverHistoryDownloads = [];
      }

      renderHistoryDownloads();
    }

    function updateHistoryDownload(downloadId, patch) {
      const downloads = readLocalHistoryDownloads();
      const existsLocally = downloads.some(download => download.id === downloadId);
      const baseDownload = readHistoryDownloads().find(download => download.id === downloadId);
      const updated = existsLocally
        ? downloads.map(download => (
            download.id === downloadId
              ? { ...download, ...patch }
              : download
          ))
        : baseDownload
          ? [{ ...baseDownload, ...patch }, ...downloads]
          : downloads;
      const updatedDownload = updated.find(download => download.id === downloadId);

      writeLocalHistoryDownloads(updated);
      renderHistoryDownloads();

      if (updatedDownload) {
        syncHistoryDownloadToServer(updatedDownload);
      }
    }

    function formatBytes(bytes) {
      const size = Number(bytes) || 0;
      if (size < 1024) return `${size} B`;
      if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
      return `${(size / 1024 / 1024).toFixed(1)} MB`;
    }

    function closeHistoryFilePreview() {
      historyFilePreview.style.display = "none";
      historyFilePreviewTitle.textContent = "-";
      historyFilePreviewBody.textContent = "";
    }

    window.closeHistoryFilePreview = closeHistoryFilePreview;

    function parseCsvPreview(text, maxRows = 120) {
      const rows = [];
      let row = [];
      let cell = "";
      let inQuotes = false;

      for (let i = 0; i < text.length; i += 1) {
        const char = text[i];
        const next = text[i + 1];

        if (char === "\"") {
          if (inQuotes && next === "\"") {
            cell += "\"";
            i += 1;
          } else {
            inQuotes = !inQuotes;
          }
          continue;
        }

        if (char === "," && !inQuotes) {
          row.push(cell);
          cell = "";
          continue;
        }

        if ((char === "\n" || char === "\r") && !inQuotes) {
          if (char === "\r" && next === "\n") {
            i += 1;
          }
          row.push(cell);
          rows.push(row);
          row = [];
          cell = "";

          if (rows.length >= maxRows) {
            break;
          }
          continue;
        }

        cell += char;
      }

      if (rows.length < maxRows && (cell || row.length)) {
        row.push(cell);
        rows.push(row);
      }

      return rows;
    }

    function renderPreviewTable(rows, filename) {
      historyFilePreviewTitle.textContent = filename;
      historyFilePreviewBody.textContent = "";

      if (!rows || rows.length === 0) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.style.padding = "12px";
        empty.textContent = "El archivo no contiene filas para mostrar.";
        historyFilePreviewBody.appendChild(empty);
        historyFilePreview.style.display = "block";
        return;
      }

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const tbody = document.createElement("tbody");
      const headerRow = document.createElement("tr");
      const headers = rows[0] || [];

      headers.forEach(header => {
        const th = document.createElement("th");
        th.textContent = header || "-";
        headerRow.appendChild(th);
      });

      thead.appendChild(headerRow);

      rows.slice(1).forEach(row => {
        const tr = document.createElement("tr");
        const width = Math.max(headers.length, row.length);

        for (let i = 0; i < width; i += 1) {
          const td = document.createElement("td");
          td.textContent = row[i] ?? "";
          tr.appendChild(td);
        }

        tbody.appendChild(tr);
      });

      table.appendChild(thead);
      table.appendChild(tbody);
      historyFilePreviewBody.appendChild(table);
      historyFilePreview.style.display = "block";
      historyFilePreview.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    async function previewHistoryFile(file) {
      const filename = file.name || "archivo";
      const lowerName = filename.toLowerCase();

      if (lowerName.endsWith(".csv") || file.type === "text/csv") {
        const text = await file.text();
        renderPreviewTable(parseCsvPreview(text), filename);
        return;
      }

      if (lowerName.endsWith(".xlsx")) {
        if (!window.XLSX) {
          alert("No se puede previsualizar Excel porque no se ha cargado el lector XLSX.");
          return;
        }

        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: "array" });
        const firstSheetName = workbook.SheetNames[0];
        const sheet = workbook.Sheets[firstSheetName];
        const rows = XLSX.utils.sheet_to_json(sheet, {
          header: 1,
          blankrows: false,
          defval: ""
        }).slice(0, 120);

        renderPreviewTable(rows, `${filename} · ${firstSheetName}`);
        return;
      }

      alert("Este tipo de archivo no se puede previsualizar dentro del dashboard.");
    }

    async function openDownloadedHistoryFile(downloadId) {
      const handle = await getHistoryFileHandle(downloadId);

      if (!handle) {
        alert("No hay una ubicacion guardada para este archivo. Pulsa Buscar y selecciona el archivo para recordarla.");
        return;
      }

      try {
        const hasPermission = await ensureHistoryFilePermission(handle);

        if (!hasPermission) {
          alert("No se puede abrir el archivo porque el navegador no tiene permiso para acceder a esa ruta.");
          return;
        }

        const file = await handle.getFile();
        await previewHistoryFile(file);
      } catch (err) {
        console.error("No se pudo abrir el archivo descargado:", err);
        alert("No se puede encontrar el archivo en la ruta original. Pulsa Buscar y selecciona su nueva ubicacion.");
      }
    }

    window.openDownloadedHistoryFile = openDownloadedHistoryFile;

    function pickerTypesForDownload(download) {
      return download?.format === "csv" || String(download?.filename || "").toLowerCase().endsWith(".csv")
        ? [{
            description: "CSV",
            accept: { "text/csv": [".csv"] }
          }]
        : [{
            description: "Excel",
            accept: {
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"]
            }
          }];
    }

    async function relinkDownloadedHistoryFile(downloadId) {
      const download = readHistoryDownloads().find(item => item.id === downloadId);

      if (!download) {
        alert("No se puede encontrar esta descarga en el historial.");
        return;
      }

      if (!window.showOpenFilePicker) {
        alert("Este navegador no permite recordar una nueva ubicacion del archivo desde la web.");
        return;
      }

      try {
        const [handle] = await window.showOpenFilePicker({
          multiple: false,
          types: pickerTypesForDownload(download)
        });

        if (!handle) {
          return;
        }

        const file = await handle.getFile();
        const saved = await saveHistoryFileHandle(downloadId, handle);

        if (!saved) {
          alert("No se ha podido guardar la nueva ubicacion del archivo.");
          return;
        }

        updateHistoryDownload(downloadId, {
          filename: file.name || handle.name || download.filename,
          bytes: file.size || download.bytes,
          hasLocalHandle: true,
          relinkedAt: new Date().toISOString()
        });

        alert("Nueva ubicacion guardada. Ya puedes abrir el archivo desde el historial.");
      } catch (err) {
        if (err?.name === "AbortError") {
          return;
        }

        console.error("No se pudo recordar la nueva ubicacion:", err);
        alert("No se ha podido recordar la nueva ubicacion del archivo.");
      }
    }

    window.relinkDownloadedHistoryFile = relinkDownloadedHistoryFile;

    function renderHistoryDownloads() {
      if (!historyDownloadsList) {
        return;
      }

      historyDownloadsList.textContent = "";

      if (!currentUser) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "Inicia sesion para ver tus descargas.";
        historyDownloadsList.appendChild(empty);
        return;
      }

      const downloads = readHistoryDownloads();

      if (!downloads.length) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "Aun no has descargado ningun archivo.";
        historyDownloadsList.appendChild(empty);
        return;
      }

      downloads.slice(0, 8).forEach(download => {
        const item = document.createElement("div");
        item.className = "download-item";

        const text = document.createElement("div");
        text.className = "download-info";
        const name = document.createElement("div");
        name.className = "download-name";
        name.textContent = download.filename;

        const meta = document.createElement("div");
        meta.className = "download-meta";
        meta.textContent = `${download.site} · ${download.startDate} / ${download.endDate} · ${download.granularity} · ${formatBytes(download.bytes)}`;

        const date = document.createElement("div");
        date.className = "download-meta mono";
        date.textContent = new Date(download.downloadedAt).toLocaleString("es-ES");

        const actions = document.createElement("div");
        actions.className = "download-actions";

        const openButton = document.createElement("button");
        openButton.type = "button";
        openButton.className = "download-open";
        openButton.textContent = "Ver";
        openButton.addEventListener("click", () => openDownloadedHistoryFile(download.id));

        const relinkButton = document.createElement("button");
        relinkButton.type = "button";
        relinkButton.className = "download-open secondary";
        relinkButton.textContent = "Buscar";
        relinkButton.addEventListener("click", () => relinkDownloadedHistoryFile(download.id));

        actions.appendChild(openButton);
        actions.appendChild(relinkButton);
        actions.appendChild(date);
        text.appendChild(name);
        text.appendChild(meta);
        item.appendChild(text);
        item.appendChild(actions);
        historyDownloadsList.appendChild(item);
      });
    }

    async function recordHistoryDownload(download) {
      const downloads = readLocalHistoryDownloads();
      downloads.unshift(download);
      writeLocalHistoryDownloads(downloads.slice(0, 20));
      renderHistoryDownloads();
      await syncHistoryDownloadToServer(download);
      renderHistoryDownloads();
    }

    function pickerTypesForHistory() {
      return historyFormat.value === "csv"
        ? [{
            description: "CSV",
            accept: { "text/csv": [".csv"] }
          }]
        : [{
            description: "Excel",
            accept: {
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"]
            }
          }];
    }

    async function downloadHistory() {
      setHistoryError("");

      if (!currentUser) {
        openLoginPanel();
        setHistoryError("Inicia sesión para descargar históricos.");
        return;
      }

      syncHistoryDateLimits();

      const maxEndDate = getHistoryEndDate();
      const startDate = historyStartDate.value;
      const endDate = historyEndDate.value;

      if (!startDate || !endDate) {
        setHistoryError("Selecciona fecha de inicio y fecha final.");
        return;
      }

      if (startDate > endDate) {
        setHistoryError("La fecha desde no puede ser posterior a la fecha hasta.");
        return;
      }

      if (endDate > maxEndDate) {
        setHistoryError("La fecha hasta debe ser, como maximo, el dia anterior a hoy.");
        return;
      }

      const selectedFilename = normalizeHistoryFilename(historyFileName.value);
      historyFileName.value = selectedFilename;

      let saveHandle = null;

      if (window.showSaveFilePicker) {
        try {
          saveHandle = await window.showSaveFilePicker({
            suggestedName: selectedFilename,
            types: pickerTypesForHistory()
          });
        } catch (err) {
          if (err?.name === "AbortError") {
            return;
          }
          console.error("Error abriendo selector de descarga:", err);
          setHistoryError(err.message || "No se pudo abrir el selector de descarga.");
          return;
        }
      }

      const params = new URLSearchParams({
        site_id: historySiteSelect.value,
        start_date: startDate,
        end_date: endDate,
        variable: historyVariable.value,
        granularity: historyGranularity.value,
        file_format: historyFormat.value
      });

      historyDownloadBtn.disabled = true;
      historyDownloadBtn.textContent = "Preparando...";

      try {
        const response = await fetch(`/api/history/download?${params.toString()}`);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(historyDownloadErrorMessage(errorData, response.status));
        }

        const blob = await response.blob();
        const historyWarning = response.headers.get("X-History-Warning");
        const savedFilename = saveHandle?.name || selectedFilename;

        if (saveHandle) {
          const writable = await saveHandle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = savedFilename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
        }

        const downloadId = createDownloadId();
        const handleSaved = await saveHistoryFileHandle(downloadId, saveHandle);

        await recordHistoryDownload({
          id: downloadId,
          filename: savedFilename,
          siteId: historySiteSelect.value,
          site: selectedHistorySiteName(),
          startDate,
          endDate,
          variable: historyVariable.value,
          granularity: historyGranularity.value === "daily" ? "diario" : "horario",
          format: historyFormat.value,
          bytes: blob.size,
          downloadedAt: new Date().toISOString(),
          hasLocalHandle: handleSaved,
          savedWithPicker: Boolean(saveHandle)
        });

        if (historyWarning) {
          setHistoryError(historyWarning, "info");
        }

      } catch (err) {
        console.error("Error descargando historico:", err);
        setHistoryError(err.message || "No se pudo descargar el histórico.");
      } finally {
        historyDownloadBtn.disabled = false;
        historyDownloadBtn.textContent = "Descargar histórico";
      }
    }
