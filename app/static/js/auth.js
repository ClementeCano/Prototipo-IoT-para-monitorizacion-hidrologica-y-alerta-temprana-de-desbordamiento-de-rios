function setSelectedAlertSites(siteIds) {
      const selected = new Set(Array.isArray(siteIds) ? siteIds : []);

      document
        .querySelectorAll("#alertsSites input")
        .forEach(input => {
          input.checked = selected.has(input.value);
        });
    }

    function collectPreferencesFromUi() {
      return {
        notification_channel: prefChannel.value,
        alert_time: prefAlertTime.value || "08:00",
        theme: prefTheme.value || "dark",
        sites: getSelectedAlertSites()
      };
    }

    function setAlertButtonState(enabled) {
      const btn = document.getElementById("alertBtn");
      alertsEnabled = Boolean(enabled);
      btn.classList.toggle("active", alertsEnabled);
      btn.textContent = alertsEnabled ? "✅ Alertas activadas" : "🔔 Activar alertas";
    }

    function updateAuthUI() {
      const logged = Boolean(currentUser);
      authLoggedOut.style.display = "none";
      authLoggedIn.style.display = logged ? "block" : "none";
      alertsAuthNotice.style.display = logged ? "none" : "block";
      alertsControls.style.display = logged ? "block" : "none";

      loginBtn.style.display = logged ? "none" : "inline-flex";
      registerBtn.style.display = logged ? "none" : "inline-flex";
      userBtn.style.display = logged ? "inline-flex" : "none";

      sessionState.textContent = logged ? "Sesión activa" : "Sin sesión";
      alertProfileState.textContent = logged ? "Selecciona los municipios" : "Inicia sesión para guardar";
      userBtn.textContent = logged ? (currentUser.name || "Usuario") : "Usuario";
      renderHistoryDownloads();
      loadHistoryDownloads();

      if (!logged) {
        alertsPanel.style.display = "none";
        historyPanel.style.display = "none";
        userPanel.style.display = "none";
        serverHistoryDownloads = [];
        historyDownloadsLoadedFor = null;
        setAlertButtonState(false);
        return;
      }

      const preferences = getUserPreferences();
      userSummary.textContent = `${currentUser.name} · ${currentUser.email}`;
      profileName.value = currentUser.name || "";
      prefEmail.value = currentUser.email || "";
      prefChannel.value = preferences.notification_channel || "push";
      prefAlertTime.value = preferences.alert_time || "08:00";
      prefTheme.value = preferences.theme || "dark";
      applyTheme(preferences.theme || "dark");
      setSelectedAlertSites(preferences.sites || []);
      setAlertButtonState(Array.isArray(preferences.sites) && preferences.sites.length > 0);
      updateEmailConfigNotice();
    }

    function updateEmailConfigNotice() {
      if (!emailConfigNotice || !prefChannel) {
        return;
      }

      const showNotice =
        prefChannel.value === "email" &&
        emailConfig &&
        emailConfig.smtp &&
        !emailConfig.smtp.configured;

      if (!showNotice) {
        emailConfigNotice.style.display = "none";
        emailConfigNotice.textContent = "";
        return;
      }

      emailConfigNotice.style.display = "block";
      emailConfigNotice.textContent = emailConfig.message;
    }

    async function apiJson(url, options = {}) {
      const res = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {})
        }
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok || data.ok === false) {
        throw new Error(data.message || data.error || data.detail || `HTTP ${res.status}`);
      }

      return data;
    }

    function authErrorMessage(error) {
      const message = error?.message || String(error || "");
      const messages = {
        password_too_short: "La contraseña debe tener al menos 8 caracteres.",
        password_not_secure: "La contraseña debe incluir mayúscula, minúscula y número.",
        email_already_registered: "Ya existe una cuenta con ese email.",
        invalid_email: "Introduce un email válido.",
        invalid_name: "Introduce un nombre válido.",
        invalid_credentials: "Email o contraseña incorrectos."
      };
      return messages[message] || message;
    }

    async function loadCurrentUser() {
      try {
        const data = await apiJson("/api/users/me");
        currentUser = data.authenticated ? data.user : null;
        updateAuthUI();
      } catch (err) {
        console.warn("No se pudo cargar usuario", err);
        currentUser = null;
        updateAuthUI();
      }
    }

    async function loadEmailConfig() {
      try {
        emailConfig = await apiJson("/api/email/config");
      } catch (err) {
        emailConfig = {
          smtp: { configured: false },
          message: "No se pudo comprobar la configuración SMTP del servidor."
        };
      }

      updateEmailConfigNotice();
    }

    async function registerUser() {
      try {
        const password = document.getElementById("registerPassword").value;
        const confirmation = document.getElementById("registerPasswordConfirm").value;
        const validation = getPasswordValidation(password, confirmation);

        if (!validation.ok) {
          updateRegisterPasswordHelp();
          alert("Revisa la contraseña: debe ser segura y coincidir en ambos campos.");
          return;
        }

        const data = await apiJson("/api/users/register", {
          method: "POST",
          body: JSON.stringify({
            name: document.getElementById("registerName").value,
            email: document.getElementById("registerEmail").value,
            password
          })
        });
        currentUser = data.user;
        updateAuthUI();
        closeModal("registerPanel");
        openModal(userPanel);
        alert("Cuenta creada");
      } catch (err) {
        alert(`No se pudo crear la cuenta: ${authErrorMessage(err)}`);
      }
    }

    async function loginUser() {
      try {
        const data = await apiJson("/api/users/login", {
          method: "POST",
          body: JSON.stringify({
            email: document.getElementById("loginEmail").value,
            password: document.getElementById("loginPassword").value
          })
        });
        currentUser = data.user;
        updateAuthUI();
        closeModal("loginPanel");
        openModal(userPanel);
      } catch (err) {
        alert(authErrorMessage(err));
      }
    }

    async function logoutUser() {
      try {
        await apiJson("/api/users/logout", { method: "POST", body: "{}" });
      } catch (err) {
        console.warn("No se pudo cerrar sesión", err);
      }

      currentUser = null;
      firebaseToken = null;
      updateAuthUI();
      closeModal("userPanel");
    }

    async function savePreferencesToServer() {
      if (!currentUser) {
        openLoginPanel();
        throw new Error("auth_required");
      }

      const data = await apiJson("/api/users/profile", {
        method: "PUT",
        body: JSON.stringify({
          name: profileName.value,
          preferences: collectPreferencesFromUi()
        })
      });
      currentUser = data.user;
      updateAuthUI();
      return currentUser;
    }

    async function saveUserProfile() {
      try {
        await savePreferencesToServer();
        alert("Perfil guardado");
      } catch (err) {
        console.error("Error guardando perfil:", err);
        alert(err.message === "auth_required" ? "Inicia sesión para guardar" : "No se pudo guardar el perfil");
      }
    }

    function toggleUserPanel() {
      if (!currentUser) {
        openLoginPanel();
        return;
      }

      userPanel.style.display = userPanel.style.display === "none" ? "flex" : "none";
    }
