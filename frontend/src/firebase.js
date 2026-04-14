// ── Firebase configuration ────────────────────────────────
// Replace the values below with your own Firebase project config.
// Go to: https://console.firebase.google.com → your project → ⚙️ Settings → General → Your apps
import { initializeApp } from "firebase/app";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  signOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  updateProfile,
} from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCGYm8_GwPSOJKnas17m2PauJReZN6gilw",
  authDomain: "speech-to-text-698a4.firebaseapp.com",  // must be this, not localhost
  projectId: "speech-to-text-698a4",
  storageBucket: "speech-to-text-698a4.firebasestorage.app",
  messagingSenderId: "272378279755",
  appId: "1:272378279755:web:853dd691cb0baff0729ace",
  measurementId: "G-FVNSBSJC0H"
};

const app  = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

export {
  auth,
  googleProvider,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  updateProfile,
};
