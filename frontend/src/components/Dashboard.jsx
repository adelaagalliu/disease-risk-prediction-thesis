import { useEffect, useMemo, useRef, useState } from "react";
import { diseaseMeta } from "../api";
import { formatDisplayValue } from "../profile";
import HealthForm from "./HealthForm";
import HealthTrends from "./HealthTrends";

const diseaseOrder = ["diabetes", "heart_disease", "hypertension", "stroke"];
const percent = (value) => Number((Number(value) * 100).toFixed(1));
const dateLabel = (value) =>
  new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(new Date(value));

function RiskCard({ disease, prediction, error }) {
  const meta = diseaseMeta[disease];
  const score = prediction ? percent(prediction.probability) : 0;
  return (
    <article className="risk-card">
      <div className="risk-heading">
        <span className="risk-icon" style={{ color: meta.color, background: meta.color + "16" }}>{meta.short}</span>
        <div>
          <h3>{meta.label}</h3>
          <span>{prediction ? (prediction.classification === "elevated" ? "Rrezik i rritur" : "Rrezik jo i rritur") : "Nuk disponohet"}</span>
        </div>
      </div>
      <div className="risk-score">
        <div className="risk-ring" style={{ "--score": score, "--ring": meta.color }}>
          <strong>{prediction ? score + "%" : "—"}</strong>
        </div>
        <div>
          <span>Model threshold</span>
          <strong>{prediction ? percent(prediction.threshold) + "%" : "—"}</strong>
        </div>
      </div>
      {prediction?.created_at && <p className="panel-caption">Saved {new Date(prediction.created_at).toLocaleString()}</p>}
      {error && <p className="card-error" role="alert">{error}</p>}
    </article>
  );
}

function HistoryChart({ history, disease }) {
  const data = history[disease] || [];
  if (!data.length) {
    return <div className="empty-state">Rezultati i parë i llogaritur me sukses do të shfaqet këtu.</div>;
  }
  const width = 680;
  const height = 220;
  const left = 40;
  const right = 18;
  const top = 14;
  const bottom = 34;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const x = (index) => left + (data.length === 1 ? innerWidth / 2 : (index / (data.length - 1)) * innerWidth);
  const y = (probability) => top + (1 - probability) * innerHeight;
  const points = data.map((item, index) => x(index) + "," + y(item.probability)).join(" ");
  const meta = diseaseMeta[disease];

  return (
    <div className="chart-wrap">
      <svg viewBox={"0 0 " + width + " " + height} role="img" aria-label={meta.label + " estimate history"}>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="grid-line" />
            <text x={left - 8} y={y(tick) + 4} textAnchor="end" className="axis-label">{Math.round(tick * 100)}%</text>
          </g>
        ))}
        {data.length > 1 && <polyline points={points} fill="none" stroke={meta.color} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />}
        {data.map((item, index) => (
          <g key={item.date + index}>
            <circle cx={x(index)} cy={y(item.probability)} r="6" fill="#fff" stroke={meta.color} strokeWidth="4">
              <title>{dateLabel(item.date) + ": " + percent(item.probability) + "%"}</title>
            </circle>
            {(data.length <= 6 || index === 0 || index === data.length - 1) && (
              <text x={x(index)} y={height - 8} textAnchor="middle" className="axis-label">{dateLabel(item.date)}</text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

function RecommendationList({ predictions }) {
  const recommendations = useMemo(
    () =>
      Object.entries(predictions)
        .flatMap(([disease, prediction]) =>
          (prediction.recommendations || []).map((item) => ({ ...item, disease })),
        )
        .sort((a, b) => b.reduction_percentage_points - a.reduction_percentage_points)
        .slice(0, 5),
    [predictions],
  );

  if (!recommendations.length) {
    return (
      <div className="empty-state compact">
        Nuk ka skenarë të disponueshëm për këto rezultate. Plotëso një vlerësim për të parë ndryshimet e mundshme që sugjeron modeli.
      </div>
    );
  }

  return (
    <div className="recommendation-list">
      {recommendations.map((item, index) => (
        <article className="recommendation" key={item.disease + item.action}>
          <span className="recommendation-number">{String(index + 1).padStart(2, "0")}</span>
          <div>
            <div className="recommendation-meta">
              <span>{diseaseMeta[item.disease].label}</span>
              <span>{item.scenario_type.replaceAll("_", " ")}</span>
            </div>
            <h3>{item.message}</h3>
            {item.ai_explanation && (
              <p className="ai-explanation">
                {item.ai_generated && <span className="ai-badge">AI</span>}
                {item.ai_explanation}
              </p>
            )}
            <p>
              Ndryshimi i parashikuar nga modeli: <strong>−{item.reduction_percentage_points.toFixed(1)} pikë përqindjeje</strong>
              {" "}({item.relative_reduction_percent.toFixed(1)}% relative)
            </p>
          </div>
          <div className="recommendation-result">
            <span>Vlerësimi i ri</span>
            <strong>{percent(item.new_probability)}%</strong>
          </div>
        </article>
      ))}
      <p className="causal-note">Shifrat llogariten nga modeli logjistik. AI përdoret vetëm për formulimin e shpjegimit dhe nuk ndryshon rezultatet. Skenarët nuk provojnë shkakësi.</p>
    </div>
  );
}

function FactorList({ prediction }) {
  const factors = prediction?.top_factors?.slice(0, 5) || [];
  if (!factors.length) return <div className="empty-state compact">Nuk ka të dhëna shpjeguese të disponueshme.</div>;
  const max = Math.max(...factors.map((item) => item.importance), 0.001);
  return (
    <div className="factor-list">
      {factors.map((factor) => (
        <div className="factor" key={factor.feature}>
          <div>
            <strong>{factor.label}</strong>
            <span>{String(factor.value)}</span>
          </div>
          <div className="factor-bar">
            <span
              className={factor.direction === "increases_estimate" ? "up" : "down"}
              style={{ width: (factor.importance / max) * 100 + "%" }}
            />
          </div>
          <span className={factor.direction === "increases_estimate" ? "direction up-text" : "direction down-text"}>
            {factor.direction === "increases_estimate" ? "E rrit" : "E zvogelon"}
          </span>
        </div>
      ))}
    </div>
  );
}

function ProfileSummary({ profile }) {
  const items = [
    ["age", "Mosha"],
    ["sex", "Gjinia"],
    ["height_cm", "Gjatësia"],
    ["weight_kg", "Pesha"],
    ["glucose", "Glukoza"],
    ["systolic_bp", "Tensioni sistolik"],
    ["diastolic_bp", "Tensioni diastolik"],
    ["cholesterol", "Kolesteroli"],
    ["smoking_status", "Konsumimi i duhanit"],
    ["heavy_alcohol", "Konsum i lartë i alkoolit"],
  ];
  return (
    <dl className="profile-summary">
      {items.map(([key, label]) => (
        <div key={key}><dt>{label}</dt><dd>{formatDisplayValue(key, profile[key])}</dd></div>
      ))}
    </dl>
  );
}

export default function Dashboard({ user, onLogout, onUpdateProfile, onRefresh, busy, error }) {
  const [historyDisease, setHistoryDisease] = useState("diabetes");
  const [factorDisease, setFactorDisease] = useState("diabetes");
  const [editing, setEditing] = useState(false);
  const dialogRef = useRef(null);
  useEffect(() => {
    if (!editing) return;
    const previous = document.activeElement;
    const priorOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.querySelector("button")?.focus();
    const keyboard = event => {
      if (event.key === "Escape" && !busy) setEditing(false);
      if (event.key !== "Tab") return;
      const elements = [...dialogRef.current.querySelectorAll("button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex='0']")].filter(el => el.getClientRects().length);
      const first=elements[0],last=elements.at(-1);
      if (!first) {event.preventDefault(); return;}
      if(event.shiftKey && document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first.focus();}
    };
    document.addEventListener("keydown",keyboard);
    return ()=>{document.body.style.overflow=priorOverflow;document.removeEventListener("keydown",keyboard);previous?.focus();};
  },[editing,busy]);
  const predictions = user.predictions || {};
  const errors = user.errors || {};
  const lastUpdated = Object.values(user.history || {})
    .flat()
    .map((item) => new Date(item.date))
    .sort((a, b) => b - a)[0];

  const update = async (profile) => {
    await onUpdateProfile(profile);
    setEditing(false);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">HL</span><span>Health Lens</span></div>
        <nav aria-label="Primary navigation"><a className="active" href="#overview">Përmbledhje</a><a href="#history">Historiku</a><a href="#recommendations">Rekomandime</a></nav>
        <div className="user-menu">
          <span>{user.name.split(" ").map((part) => part[0]).join("").slice(0, 2).toUpperCase()}</span>
          <div><strong>{user.name}</strong><small>{user.email}</small></div>
          <button type="button" disabled={busy} onClick={onLogout}>Dil</button>
        </div>
      </header>

      <main className="dashboard">
        <section className="dashboard-intro" id="overview">
          <div>
            <span className="section-label">Përmbledhje e gjendjes shëndetësore</span>
            <h1>Mirë{new Date().getHours() < 12 ? "mëngjes" : new Date().getHours() < 18 ? "dita" : "mbrëma"}, {user.name.split(" ")[0]}.</h1>
            <p>Vlerësimi yt më i fundit dhe faktorët kryesorë që duhet të kesh parasysh.</p>
          </div>
          <div className="intro-actions">
            <button type="button" className="secondary-button" disabled={busy} onClick={onRefresh}>{busy ? "Ju lutem prisni..." : "Rifresko historikun"}</button>
            <span>{lastUpdated ? "Përditësuar më " + dateLabel(lastUpdated) : "Ende pa u llogaritur"}</span>
            <button className="primary-button" disabled={busy} onClick={() => setEditing(true)} type="button">{Object.keys(predictions).length ? "Shiko dhe përditëso të dhënat" : "Fillo vlerësimin e parë"}</button>
          </div>
        </section>

        {error && <div className="api-notice" role="alert">{error}</div>}
        {Object.keys(errors).length > 0 && (
          <div className="api-notice">
            Disa vlerësime nuk mundën të përditësoheshin. Vlerësimet e ruajtura më parë ruajnë datën e tyre fillestare.
          </div>
        )}

        <section className="risk-grid" aria-label="Latest disease estimates">
          {diseaseOrder.map((disease) => (
            <RiskCard key={disease} disease={disease} prediction={predictions[disease]} error={errors[disease]} />
          ))}
        </section>

        <div className="dashboard-grid">
          <section className="panel history-panel" id="history">
            <div className="panel-heading">
              <div><span className="section-label">Me kalimin e kohës</span><h2>Historiku i vlerësimeve</h2></div>
              <div className="disease-tabs">
                {diseaseOrder.map((disease) => (
                  <button type="button" key={disease} className={historyDisease === disease ? "active" : ""} onClick={() => setHistoryDisease(disease)}>
                    {diseaseMeta[disease].label.replace(" pattern", "")}
                  </button>
                ))}
              </div>
            </div>
            <HistoryChart history={user.history || {}} disease={historyDisease} />
            <p className="causal-note">500 vlerësimet më të fundit për të gjitha sëmundjet. Mund të ndiqni renditjen e vlerësimeve.</p>
          </section>

          <aside className="panel profile-panel">
            <div className="panel-heading"><div><span className="section-label">Të dhënat aktuale</span><h2>Parametrat e tua</h2></div><button className="text-button" type="button" onClick={() => setEditing(true)}>Ndrysho</button></div>
            <ProfileSummary profile={user.profile} />
            <button className="secondary-button wide" type="button" onClick={() => setEditing(true)}>Hap të gjithë parametrat</button>
          </aside>
        </div>

        <HealthTrends revision={user} />

        <section className="panel recommendations-panel" id="recommendations">
          <div className="panel-heading">
            <div><span className="section-label">Hapat e ardhshëm të modeluar</span><h2>Skenarët me ndikimin më të madh</h2></div>
            <span className="panel-caption">Ndryshimet me ndikim më të madh në të gjitha vlerësimet</span>
          </div>
          <RecommendationList predictions={predictions} />
        </section>

        <section className="panel factors-panel">
          <div className="panel-heading">
            <div><span className="section-label">Shpjegueshmëria</span><h2>Çfarë ndikoi në vlerësim</h2></div>
            <select aria-label="Disease explanation" value={factorDisease} onChange={(event) => setFactorDisease(event.target.value)}>
              {diseaseOrder.map((disease) => <option key={disease} value={disease}>{diseaseMeta[disease].label}</option>)}
            </select>
          </div>
          <FactorList prediction={predictions[factorDisease]} />
          <p className="causal-note">Faktorët SHAP tregojnë se si inputet e kanë zhvendosur këtë rezultat të modelit në raport me vlerën e tij bazë. Ato nuk përbëjnë shpjegime shkakësore.</p>
        </section>

        <footer className="dashboard-footer">
          <strong>Ky është një vlerësim kërkimor dhe jo një diagnozë mjekësore.</strong>
          <span>Diskutoni shqetësimet shëndetësore dhe matjet klinike me një profesionist të kualifikuar të kujdesit shëndetësor.</span>
        </footer>
      </main>

      {editing && (
        <div ref={dialogRef} className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Ndrysho parametër">
          <div className="modal-sheet">
            <HealthForm initialProfile={user.profile} onSubmit={update} onCancel={() => setEditing(false)} busy={busy} mode="edit" />
          </div>
        </div>
      )}
    </div>
  );
}
