const firebaseConfig = {
      apiKey: "AIzaSyCXNKG8wb5IsTLaL6WLPIiRCPtuF4MIlLo",
      authDomain: "rio-ebro.firebaseapp.com",
      projectId: "rio-ebro",
      storageBucket: "rio-ebro.firebasestorage.app",
      messagingSenderId: "867230279445",
      appId: "1:867230279445:web:5afb433821606547276b1c",
      measurementId: "G-N9PV3N2M2J"
    };

    const FCM_VAPID_KEY =
      "BCFznYCtRHRklZcr-XJ2jjnjvzIFV0OBPqcZXdZWJdJI7YT8GClCn-uc8Qi1fFT_zyicRQNrRxG7qo6uKJwWGng";

    firebase.initializeApp(firebaseConfig);

    let messaging = null;
    let foregroundHandlerRegistered = false;
    let serviceWorkerRegistrationPromise = null;

    async function showPushNotification(title, options = {}) {
      if (Notification.permission !== "granted") return;

      const notificationOptions = {
        body: options.body || "",
        icon: options.icon || "/static/icon.png",
        badge: options.badge || "/static/icon.png",
        tag: options.tag || "rio-ebro-alert",
        renotify: true,
        requireInteraction: true,
        data: {
          url: options.url || "/"
        }
      };

      try {
        const registration = await navigator.serviceWorker.ready;
        await registration.showNotification(title, notificationOptions);
        return;
      } catch (err) {
        console.warn("No se pudo mostrar notificacion via Service Worker", err);
      }

      try {
        new Notification(title, notificationOptions);
      } catch (err) {
        console.warn("No se pudo mostrar notificacion en foreground", err);
      }
    }

    async function getMessagingInstance() {
      if (messaging) return messaging;

      if (!window.isSecureContext) {
        throw new Error("FCM Web Push necesita HTTPS o localhost.");
      }

      if (!("Notification" in window)) {
        throw new Error("Este navegador no soporta notificaciones web.");
      }

      if (!("serviceWorker" in navigator)) {
        throw new Error("Este navegador no soporta service workers.");
      }

      if (firebase.messaging.isSupported) {
        const supported = await firebase.messaging.isSupported();
        if (!supported) {
          throw new Error("Firebase Messaging no esta soportado en este navegador.");
        }
      }

      messaging = firebase.messaging();

      if (!foregroundHandlerRegistered) {
        messaging.onMessage((payload) => {
          debugLog("Foreground message:", payload);

          const title = payload.notification?.title || payload.data?.title || "Alerta Rio";
          const body = payload.notification?.body || payload.data?.body || "";

          showPushNotification(title, {
            body,
            icon: payload.notification?.icon || payload.data?.icon || "/static/icon.png",
            tag: payload.data?.tag || "rio-ebro-alert",
            url: payload.fcmOptions?.link || payload.data?.url || "/"
          });
        });

        foregroundHandlerRegistered = true;
      }

      return messaging;
    }

async function initNotifications({ allowPrompt = true } = {}) {

      try {

        // =========================
        // EVITAR INICIALIZAR 20 VECES
        // =========================
        if (notificationsInitialized && firebaseToken) {
          return firebaseToken;
        }

        debugLog("Iniciando notificaciones");
        await logPushDebug("init_start", { allowPrompt });

        const messagingInstance = await getMessagingInstance();
        await logPushDebug("messaging_ready");

        // =========================
        // PERMISOS
        // =========================
        let permission = Notification.permission;

        if (permission === "default") {
          if (!allowPrompt) {
            debugLog("Permiso pendiente; no se solicita fuera de una accion del usuario.");
            await logPushDebug("permission_default_no_prompt");
            return null;
          }

          permission = await Notification.requestPermission();
        }

        debugLog("Permiso:", permission);
        await logPushDebug("permission_result", { permission });

        if (permission !== "granted") {

          debugLog("Permiso denegado");
          await logPushDebug("permission_not_granted", { permission });
          return null;
        }

        // =========================
        // SERVICE WORKER
        // =========================
        if (!serviceWorkerRegistrationPromise) {

          serviceWorkerRegistrationPromise = navigator.serviceWorker.register(
            "/firebase-messaging-sw.js",
            { scope: "/" }
          );
        }

        const registered = await serviceWorkerRegistrationPromise;
        await registered.update();

        debugLog("Service Worker registrado:", registered.scope);
        await logPushDebug("service_worker_registered", {
          serviceWorkerScope: registered.scope
        });

        // =========================
        // ESPERAR SW
        // =========================
        const registration = await navigator.serviceWorker.ready;

        // =========================
        // TOKEN FIREBASE
        // =========================
        await logPushDebug("get_token_start");

        firebaseToken = await messagingInstance.getToken({
          vapidKey: FCM_VAPID_KEY,
          serviceWorkerRegistration: registration
        });

        if (!firebaseToken) {

          debugLog("No se genero token");
          await logPushDebug("get_token_empty");
          return null;
        }

        debugLog("Token Firebase obtenido:", firebaseToken.slice(0, 15));
        await logPushDebug("get_token_ok", {
          tokenPrefix: firebaseToken.slice(0, 15)
        });

        notificationsInitialized = true;

        return firebaseToken;

      } catch (err) {

        lastFirebaseError = {
          code: err?.code || null,
          message: err?.message || String(err),
          stack: err?.stack || null
        };

        console.error("❌ ERROR Firebase:", lastFirebaseError);
        await logPushDebug("firebase_error", {
          errorCode: lastFirebaseError.code,
          errorMessage: lastFirebaseError.message
        });

        alert(
          "Error Firebase:\n" +
          [
            lastFirebaseError.code,
            lastFirebaseError.message
          ].filter(Boolean).join("\n")
        );

        return null;
      }
    }

    window.debugPush = async function debugPush() {
      const result = {
        href: location.href,
        origin: location.origin,
        protocol: location.protocol,
        secureContext: window.isSecureContext,
        notification: "Notification" in window ? Notification.permission : "unsupported",
        serviceWorker: "serviceWorker" in navigator,
        pushManager: "PushManager" in window,
        firebaseMessagingSupported: null,
        serviceWorkerReady: false,
        tokenPrefix: null,
        lastFirebaseError
      };

      try {
        if (firebase.messaging.isSupported) {
          result.firebaseMessagingSupported = await firebase.messaging.isSupported();
        }

        const registration = await navigator.serviceWorker.getRegistration("/");
        result.serviceWorkerRegistered = Boolean(registration);
        result.serviceWorkerScope = registration?.scope || null;

        const ready = await navigator.serviceWorker.ready;
        result.serviceWorkerReady = Boolean(ready);

        const token = await initNotifications({ allowPrompt: true });
        result.tokenPrefix = token ? token.slice(0, 18) : null;
      } catch (err) {
        result.lastFirebaseError = {
          code: err?.code || null,
          message: err?.message || String(err),
          stack: err?.stack || null
        };
      }

      debugLog("Diagnostico push:", result);
      return result;
    };

    async function enviarSuscripciones({ allowPrompt = true, requireSites = false } = {}) {

      try {

        // =========================
        // ASEGURAR TOKEN
        // =========================
        await sitesReady;
        await logPushDebug("subscription_start", { allowPrompt, requireSites });

        if (!currentUser) {
          openLoginPanel();
          throw new Error("auth_required");
        }

        if (!firebaseToken) {

          const token = await initNotifications({ allowPrompt });

          if (!token) {
            await logPushDebug("subscription_without_token");
            if (requireSites) {
              throw new Error("No se ha podido generar el token push en este dispositivo.");
            }
            return null;
          }
        }

        // =========================
        // MUNICIPIOS
        // =========================
        const selectedSites = getSelectedAlertSites();

        debugLog("Municipios:", selectedSites);
        await logPushDebug("subscription_sites_selected", {
          selectedSitesCount: selectedSites.length
        });

        if (requireSites && selectedSites.length === 0) {
          debugLog("No se envia token: no hay municipios seleccionados.");
          await logPushDebug("subscription_no_sites");
          throw new Error("Selecciona al menos un municipio");
        }

        // =========================
        // GUARDAR LOCAL
        // =========================
        localStorage.setItem(
          "alertSites",
          JSON.stringify(selectedSites)
        );

        // =========================
        // ENVIAR BACKEND
        // =========================
        debugLog("Enviando token al backend");

        await logPushDebug("api_token_post_start", {
          selectedSitesCount: selectedSites.length,
          tokenPrefix: firebaseToken.slice(0, 15)
        });

        const res = await fetch("/api/token", {
          

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            token: firebaseToken,
            sites: selectedSites,
            userAgent: navigator.userAgent,
            platform: navigator.platform || null
          })
        });

        const data = await res.json();
        debugLog("Respuesta token backend:", {
          status: res.status,
          ok: data?.ok,
          token_saved: data?.token_saved,
          total_tokens: data?.total_tokens
        });
        await logPushDebug("api_token_post_response", {
          selectedSitesCount: selectedSites.length,
          responseStatus: res.status,
          responseOk: res.ok,
          tokenSaved: data?.token_saved,
          tokenPrefix: firebaseToken.slice(0, 15)
        });

        if (!res.ok || !data.ok) {
          throw new Error(data.message || data.error || `HTTP ${res.status}`);
        }

        if (requireSites && !data.token_saved) {
          throw new Error("El servidor no ha guardado el token push.");
        }

        if (data.user) {
          currentUser = data.user;
          updateAuthUI();
        }

        debugLog("Suscripciones guardadas:", {
          token_saved: data?.token_saved,
          sites: data?.sites
        });
        return data;

      } catch (err) {

        console.error("❌ Error enviando suscripciones:", err);
        await logPushDebug("subscription_error", {
          errorMessage: err?.message || String(err)
        });
        throw err;
      }
    }

    let alertsEnabled = false;

    function toggleAlerts() {

      const panel = document.getElementById("alertsPanel");

      if (!currentUser) {
        openLoginPanel();
        alert("Inicia sesión para programar alertas.");
        return;
      }

      panel.style.display = panel.style.display === "none" ? "flex" : "none";
    }

    window.addEventListener("load", () => {
      Promise.all([sitesReady, currentUserReady]).then(() => {
        const preferences = getUserPreferences();

        if (
          currentUser &&
          preferences.notification_channel === "push" &&
          Array.isArray(preferences.sites) &&
          preferences.sites.length > 0
        ) {
          initNotifications({ allowPrompt: false }).then((token) => {
            if (token) {
              return enviarSuscripciones({ allowPrompt: false, requireSites: true });
            }
            return null;
          }).catch((err) => {
            console.warn("No se han podido restaurar las suscripciones push:", err);
          });
        }
      });
    });

  window.sendSelectedAlertsTest = async function() {
    const selectedSites = getSelectedAlertSites();

    if (!currentUser) {
      openLoginPanel();
      throw new Error("auth_required");
    }

    if (selectedSites.length === 0) {
      throw new Error("Selecciona al menos un municipio");
    }

    await savePreferencesToServer();

    const preferences = getUserPreferences();
    let token = null;

    if (preferences.notification_channel === "email") {
      if (emailConfig && emailConfig.smtp && !emailConfig.smtp.configured) {
        throw new Error(emailConfig.message);
      }
    } else {
      token = await initNotifications({ allowPrompt: true });

      if (!token) {
        throw new Error("No se ha podido generar token en este dispositivo");
      }
    }

    await logPushDebug("test_selected_alerts_start", {
      tokenPrefix: token ? token.slice(0, 15) : null,
      selectedSites,
      channel: preferences.notification_channel
    });

    const data = await apiJson("/api/test-selected-alerts", {
      method: "POST",
      body: JSON.stringify({
        token,
        sites: selectedSites,
        userAgent: navigator.userAgent,
        platform: navigator.platform || null
      })
    });

    await logPushDebug("test_selected_alerts_response", {
      tokenPrefix: token ? token.slice(0, 15) : null,
      sent: data?.sent,
      selectedSites: data?.sites || selectedSites,
      channel: data?.channel
    });

    if (!data.sent && data.errors?.length) {
      throw new Error(data.errors.join("\n"));
    }

    return data;
  };

  window.testCurrentDevicePush = async function() {
    try {
      const data = await window.sendSelectedAlertsTest();
      alert(data.channel === "email" ? "Email de prueba enviado" : "Push de prueba enviado");
    } catch (err) {
      console.error("Error probando canal:", err);
      alert(err.message || "Error probando canal");
    }
  };

  window.testSelectedAlertsOnCurrentDevice = async function() {
    try {
      const data = await window.sendSelectedAlertsTest();
      alert(`Prueba enviada a ${data.processed_sites} municipios seleccionados`);
    } catch (err) {
      console.error("Error probando municipios seleccionados:", err);
      alert(err.message || "Error probando municipios seleccionados");
    }
  };

  window.removeCurrentDeviceAlerts = async function() {
    try {
      if (!currentUser) {
        openLoginPanel();
        alert("Inicia sesión para modificar tus alertas");
        return;
      }

      document
        .querySelectorAll("#alertsSites input")
        .forEach(input => {
          input.checked = false;
        });

      if (prefChannel.value === "push") {
        const token = await initNotifications({ allowPrompt: true });

        if (token) {
          firebaseToken = token;
          await enviarSuscripciones({ allowPrompt: true, requireSites: false });
        }
      }

      await savePreferencesToServer();

      localStorage.setItem("alertSites", "[]");
      localStorage.setItem("alerts", "off");

      setAlertButtonState(false);
      document.getElementById("alertsPanel").style.display = "none";

      alert("Alertas desactivadas");

    } catch (err) {
      console.error("Error quitando alertas:", err);
      alert("Error quitando alertas");
    }
  };

  window.saveAlerts = async function() {
    try {
      if (!currentUser) {
        openLoginPanel();
        alert("Inicia sesión para guardar tus alertas");
        return;
      }

      const selectedSites = getSelectedAlertSites();

      if (selectedSites.length === 0) {
        alert("Selecciona al menos un municipio");
        return;
      }

      await savePreferencesToServer();

      if (prefChannel.value === "push") {
        await enviarSuscripciones({ allowPrompt: true, requireSites: true });
      }

      document.getElementById("alertsPanel").style.display = "none";

      localStorage.setItem("alertSites", JSON.stringify(selectedSites));
      localStorage.setItem("alerts", "on");

      setAlertButtonState(true);

      alert("✅ Alertas guardadas correctamente");

    } catch (err) {
      console.error(err);
      alert(err.message === "auth_required" ? "Inicia sesión para guardar" : (err.message || "Error guardando alertas"));
    }
  };
