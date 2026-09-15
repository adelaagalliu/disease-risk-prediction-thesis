import { useEffect, useRef, useState } from "react";
import { predictAll } from "./api";
import { createAccount, getSession, login, logout, loadDashboard } from "./storage";
import AuthScreen from "./components/AuthScreen";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [user, setUser] = useState(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const generation = useRef(0);
  const locked = useRef(false);
  useEffect(() => {
    let active = true;
    getSession().then(value => {if (active) setUser(value);})
      .catch(e => {if (active) setError(e.message);})
      .finally(() => {if (active) setLoading(false);});
    const expired = () => {generation.current++; setUser(null); setError("Sesioni juaj ka skaduar. Identifikohuni sërish për të vazhduar.");};
    window.addEventListener("health-session-expired", expired);
    return () => {active = false; window.removeEventListener("health-session-expired", expired);};
  }, []);
  const authenticate = async (credentials, signup = false) => {
    if (locked.current) return;
    locked.current = true; setBusy(true); setError("");
    try {setUser(await (signup ? createAccount(credentials) : login(credentials.email, credentials.password)));}
    catch(e) {setError(e.message);}
    finally {locked.current = false; setBusy(false);}
  };
  const updateProfile = async profile => {
    if (locked.current) return;
    locked.current = true; setBusy(true); setError("");
    const run = generation.current;
    try {
      const result = await predictAll(profile);
      if (run !== generation.current) throw new Error("Ju lutem identifikohuni serish.");
      if (!Object.keys(result.predictions).length) throw new Error(Object.values(result.errors).join("; "));
      try {
        const restored = await loadDashboard();
        if (run === generation.current) setUser({...restored, errors:{...restored.errors,...result.errors}});
      } catch(e) {
        if (run === generation.current) {
          setUser(previous => ({...previous, profile, predictions:{...previous.predictions,...result.predictions}, errors:result.errors}));
          setError("Vlerësimi u ruajt, por historia nuk mund të rifreskohej. Përdorni opsionin 'Rifresko historikun'. " + e.message);
        }
      }
    } finally {locked.current = false; setBusy(false);}
  };
  const refresh = async () => {
    if (locked.current) return;
    locked.current = true; setBusy(true); setError("");
    const run = generation.current;
    try {const value = await loadDashboard(); if(run === generation.current) setUser(value);}
    catch(e) {setError(e.message);}
    finally {locked.current = false; setBusy(false);}
  };
  const handleLogout = () => {generation.current++; logout(); setUser(null); setError("");};
  if (loading) return <main className="empty-state" role="status">Duke ngarkuar profilin tuaj...</main>;
  if (user) return <Dashboard user={user} onLogout={handleLogout} onUpdateProfile={updateProfile} onRefresh={refresh} busy={busy} error={error} />;
  return <AuthScreen onLogin={credentials => authenticate(credentials)} onBeginSignup={credentials => authenticate(credentials,true)} busy={busy} error={error} />;
}
