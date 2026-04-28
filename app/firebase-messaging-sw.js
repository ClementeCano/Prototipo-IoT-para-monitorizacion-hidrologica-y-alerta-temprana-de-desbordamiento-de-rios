importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/10.12.2/firebase-messaging-compat.js");

firebase.initializeApp({
  apiKey: "AIzaSyCXNKG8wb5IsTLaL6WLPIiRCPtuF4MIlLo",
  authDomain: "rio-ebro.firebaseapp.com",
  projectId: "rio-ebro",
  messagingSenderId: "867230279445",
  appId: "1:867230279445:web:5afb433821606547276b1c"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage(function(payload) {
  self.registration.showNotification(payload.notification.title, {
    body: payload.notification.body,
  });
});