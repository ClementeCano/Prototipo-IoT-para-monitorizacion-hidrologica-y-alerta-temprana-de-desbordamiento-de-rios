importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js');

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

// =========================
// BACKGROUND NOTIFICATIONS
// =========================
messaging.onBackgroundMessage((payload) => {

  console.log(
    '[firebase-messaging-sw.js] Background message ',
    payload
  );

  const notificationTitle =
    payload.notification?.title || "Alerta Río";

  const notificationOptions = {
    body: payload.notification?.body || "",
    icon: "/static/icon.png"
  };

  self.registration.showNotification(
    notificationTitle,
    notificationOptions
  );
});