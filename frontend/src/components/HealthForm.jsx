import { useEffect, useMemo, useState } from "react";
import { defaultProfile, profileSections, validateProfile } from "../profile";

function Toggle({ value, onChange, label }) {
  return (
    <div className="toggle-field">
      <span>{label}</span>
      <div className="segmented">
        <button type="button" aria-pressed={value === true} className={value === true ? "active" : ""} onClick={() => onChange(true)}>Po</button>
        <button type="button" aria-pressed={value === false} className={value === false ? "active" : ""} onClick={() => onChange(false)}>Jo</button>
      </div>
    </div>
  );
}

function InputField({ field, value, update, profile }) {
  if (field.showWhen && profile[field.showWhen[0]] !== field.showWhen[1]) return null;
  if (field.type === "boolean") {
    const label = field.name === "heavy_alcohol" ? `Konsum i lartë i alkoolit (Po, nëse konsumon më shumë se ${profile.sex === "female" ? 7 : 14} pije në javë)` : field.label;
    return <Toggle label={label} value={value} onChange={(next) => update(field.name, next)} />;
  }
  if (field.type === "select") {
    return (
      <label className="input-field">
        <span>{field.label}</span>
        <select required value={value} onChange={(event) => update(field.name, event.target.value)}>
          <option value="" disabled>Zgjidh një përgjigje</option>
          {field.options.map(([optionValue, label]) => (
            <option key={optionValue} value={optionValue}>{label}</option>
          ))}
        </select>
      </label>
    );
  }
  return (
    <label className="input-field">
      <span>{field.label}</span>
      <div className="number-wrap">
        <input
          type="number"
          value={value}
          min={field.min}
          max={field.max}
          step={field.step || (["age", "mental_health_days", "physical_health_days"].includes(field.name) ? 1 : "any")}
          required
          onChange={(event) => update(field.name, event.target.value)}
        />
        {field.suffix && <small>{field.suffix}</small>}
      </div>
      {field.note && <em className="field-note">{field.note}</em>}
    </label>
  );
}

export default function HealthForm({ initialProfile, onSubmit, onCancel, busy, mode = "signup" }) {
  const [profile, setProfile] = useState({ ...defaultProfile, ...initialProfile });
  const [sectionIndex, setSectionIndex] = useState(0);
  const [error, setError] = useState("");
  const section = profileSections[sectionIndex];
  const progress = ((sectionIndex + 1) / profileSections.length) * 100;

  useEffect(() => {
    if (profile.smoking_status !== "current" && Number(profile.cigarettes_per_day) !== 0) {
      setProfile((current) => ({ ...current, cigarettes_per_day: 0 }));
    }
  }, [profile.smoking_status, profile.cigarettes_per_day]);

  const bmi = useMemo(() => {
    const meters = Number(profile.height_cm) / 100;
    return meters > 0 && Number(profile.weight_kg) > 0 ? (Number(profile.weight_kg) / meters ** 2).toFixed(1) : "—";
  }, [profile.height_cm, profile.weight_kg]);

  const update = (name, value) => {
    setError("");
    setProfile((current) => ({ ...current, [name]: value }));
  };

  const next = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    const issue = validateProfile(profile, sectionIndex === profileSections.length - 1 ? profileSections : [section]);
    if (issue) {setSectionIndex(issue.section); setError(issue.message); return;}
    if (sectionIndex < profileSections.length - 1) setSectionIndex(value => value + 1);
    else {try {await onSubmit(profile);} catch(e) {setError(e.message);}}

  };

  return (
    <main className={mode === "edit" ? "questionnaire edit-form" : "questionnaire"}>
      <header className="questionnaire-header">
        <div className="brand">
          <span className="brand-mark">HL</span>
          <span>Health Lens</span>
        </div>
        {mode === "edit" ? (
          <button className="icon-button" type="button" disabled={busy} onClick={onCancel} aria-label="Mbyll">×</button>
        ) : (
          <button className="text-button" type="button" onClick={onCancel}>Kthehu te hyrja</button>
        )}
      </header>

      <div className="questionnaire-layout">
        <aside className="question-nav">
          <span className="section-label">{mode === "edit" ? "Përditëso profilin" : "Profili shëndetësor"}</span>
          <h2>{mode === "edit" ? "Mbaji të dhënat e tua të përditësuara." : "Le të njohim shëndetin tënd."}</h2>
          <p>Përgjigjet e tua përdoren nga modelet kërkimore për të vlerësuar rreziqet shëndetësore.</p>
          <ol>
            {profileSections.map((item, index) => (
              <li key={item.title} className={index === sectionIndex ? "active" : index < sectionIndex ? "done" : ""}>
                <button type="button" disabled={busy || index > sectionIndex} onClick={() => setSectionIndex(index)}>
                  <span>{index < sectionIndex ? "✓" : index + 1}</span>{item.title}
                </button>
              </li>
            ))}
          </ol>
          <div className="bmi-chip"><span>BMI e llogaritur</span><strong>{bmi}</strong></div>
        </aside>

        <section className="question-card">
          <div className="progress-track"><span style={{ width: progress + "%" }} /></div>
          <div className="step-label">Hapi {sectionIndex + 1} nga {profileSections.length}</div>
          <h1>{section.title}</h1>
          <p className="muted">{section.description}</p>
          <form onSubmit={next}>
            <fieldset disabled={busy} className="form-fieldset">
            <div className="field-grid">
              {section.fields.map((field) => (
                <InputField key={field.name} field={field} value={profile[field.name]} update={update} profile={profile} />
              ))}
            </div>
            {error && <p className="form-error" role="alert">{error}</p>}
            <div className="form-actions">
              <button className="secondary-button" type="button" disabled={busy || sectionIndex === 0} onClick={() => setSectionIndex((value) => value - 1)}>
                Prapa
              </button>
              <button className="primary-button" type="submit" disabled={busy}>
                {busy ? "Duke llogaritur..." : sectionIndex === profileSections.length - 1 ? (mode === "edit" ? "Ruaj dhe llogarit përsëri" : "Shiko panelin tim") : "Vazhdo"}
                {!busy && <span aria-hidden="true">→</span>}
              </button>
            </div>
            </fieldset>
          </form>
        </section>
      </div>
    </main>
  );
}