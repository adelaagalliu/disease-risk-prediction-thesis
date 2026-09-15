import { useState } from "react";

export default function AuthScreen({ onLogin, onBeginSignup, busy, error }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "" });

  const update = (event) =>
    setForm((current) => ({ ...current, [event.target.name]: event.target.value }));

  const [validationError, setValidationError] = useState("");
  const submit = (event) => {
    event.preventDefault();
    setValidationError("");
    if (mode === "signup" && (!/\d/.test(form.password) || !/[^\p{L}\p{N}\s]/u.test(form.password) || /\s/.test(form.password))) {
      setValidationError("Përdor më shumë se 8 karaktere, duke përfshirë të paktën një numër dhe një simbol të veçantë, pa hapësira."); return;
    }
    if (mode === "signup") onBeginSignup(form);
    else onLogin(form);
  };

  return (
    <main className="auth-shell">
      <section className="auth-story">
        <div className="brand brand-light">
          <span className="brand-mark">HL</span>
          <span>Health Lens</span>
        </div>
        <div className="story-copy">
          <span className="section-label light">Profili yt shëndetësor</span>
          <h1>Kupto më qartë faktorët që ndikojnë në shëndetin tënd.</h1>
          <p>
            Ndiq vlerësimet e katër modeleve kërkimore, kupto faktorët me ndikimin më të madh
            dhe shiko si mund të ndryshojnë vlerësimet me ndryshime në stilin e jetesës me kalimin e kohës.
          </p>
        </div>
        <div className="story-note">
          <span className="pulse-dot" />
          Vetëm vlerësime të bazuara në kërkime shkencore. Ky mjet nuk jep diagnoza.
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-card">
          <div className="mobile-brand brand">
            <span className="brand-mark">C</span>
            <span>Health Lens</span>
          </div>
          <span className="section-label">{mode === "login" ? "Mirë se u ktheve" : "Krijo profilin tënd"}</span>
          <h2>{mode === "login" ? "Hyr në profilin tënd" : "Fillo me të dhënat bazë"}</h2>
          <p className="muted">
            {mode === "login"
              ? "Hyr për të parë vlerësimet e ruajtura dhe historikun tënd."
              : "Krijo llogarinë tënde, pastaj plotëso vlerësimin e parë shëndetësor."}
          </p>

          <div className="auth-tabs" role="group" aria-label="Account action">
            <button className={mode === "login" ? "active" : ""} disabled={busy} onClick={() => {setMode("login"); setValidationError("");}} type="button">
              Identifikohu
            </button>
            <button className={mode === "signup" ? "active" : ""} disabled={busy} onClick={() => {setMode("signup"); setValidationError("");}} type="button">
              Regjistrohu
            </button>
          </div>

          <form onSubmit={submit} className="auth-form">
            {mode === "signup" && <p className="muted">Fjalëkalimi: më shumë se 8 karaktere, me të paktën një numër dhe një simbol të veçantë, pa hapësira.</p>}
            <label>
              Adresa e email-it
              <input name="email" value={form.email} onChange={update} type="email" autoComplete="email" required />
            </label>
            <label>
              Fjalëkalimi
              <input name="password" value={form.password} onChange={update} type="password" minLength={mode === "signup" ? 8 : undefined} maxLength={120} autoComplete={mode === "login" ? "current-password" : "new-password"} required />
            </label>
            {(error || validationError) && <p className="form-error" role="alert">{validationError || error}</p>}
            <button className="primary-button wide" disabled={busy} type="submit">
              {busy ? "Ju lutem prisni..." : mode === "login" ? "Hap profilin" : "Krijo llogarinë"}
              {!busy && <span aria-hidden="true">→</span>}
            </button>
          </form>
          <p className="privacy-note">
            Llogaria dhe historiku i vlerësimeve ruhen edhe pasi del. Për t'i parë sërish, duhet të hysh në llogari.
          </p>
        </div>
      </section>
    </main>
  );
}
