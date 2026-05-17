importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js");

firebase.initializeApp({
  apiKey: "AIzaSyCXNKG8wb5IsTLaL6WLPIiRCPtuF4MIlLo",
  authDomain: "rio-ebro.firebaseapp.com",
  projectId: "rio-ebro",
  storageBucket: "rio-ebro.firebasestorage.app",
  messagingSenderId: "867230279445",
  appId: "1:867230279445:web:5afb433821606547276b1c",
  measurementId: "G-N9PV3N2M2J"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
  console.log("[firebase-messaging-sw.js] Background message", payload);

  const notificationTitle =
    payload.notification?.title || payload.data?.title || "Alerta Rio";

  const notificationOptions = {
    body: payload.notification?.body || payload.data?.body || "",
    icon: payload.notification?.icon || payload.data?.icon || "/static/icon.png",
    badge: payload.notification?.badge || payload.data?.badge || "/static/icon.png",
    tag: payload.data?.tag || "rio-ebro-alert",
    renotify: true,
    requireInteraction: true,
    data: {
      url: payload.fcmOptions?.link || payload.data?.url || "/"
    }
  };

  self.registration.showNotification(notificationTitle, notificationOptions);
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const targetUrl = new URL(
    event.notification.data?.url || "/",
    self.location.origin
  ).href;

  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === targetUrl && "focus" in client) {
          return client.focus();
        }
      }

      if (clients.openWindow) {
        return clients.openWindow(targetUrl);
      }
    })
  );
});
