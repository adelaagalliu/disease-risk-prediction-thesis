const API_URL = (import.meta.env?.VITE_API_URL || "/api").replace(/\/$/, "");

export const diseaseMeta = {
  diabetes: { label: "Diabeti", short: "DB", color: "#e16f47" },
  heart_disease: { label: "Sëmundja e zemrës", short: "HD", color: "#bb5b68" },
  hypertension: { label: "Hipertensioni", short: "BP", color: "#607aa7" },
  stroke: { label: "Goditje në tru (insult cerebral)", short: "ST", color: "#9170a5" },
};

const base = (p) => ({
  sex: p.sex,
  age: Number(p.age),
  height_cm: Number(p.height_cm),
  weight_kg: Number(p.weight_kg),
});

const brfss = (p) => ({
  ...base(p),
  smoking_status: p.smoking_status,
  heavy_alcohol: p.heavy_alcohol,
  physical_activity: p.physical_activity,
  education: p.education,
  income: p.income,
  income_display_currency: p.income_display_currency || "ALL",
  high_bp: p.diagnosed_hypertension,
  high_chol: p.high_chol,
  chol_check: p.chol_check,
  fruits: p.fruits,
  veggies: p.veggies,
  general_health: p.general_health,
  mental_health_days: Number(p.mental_health_days),
  physical_health_days: Number(p.physical_health_days),
  difficulty_walking: p.difficulty_walking,
  stroke: p.diagnosed_stroke,
});

export const buildPayloads = (p) => ({
  diabetes: { ...brfss(p), heart_disease: p.diagnosed_heart_disease },
  heart_disease: { ...brfss(p), diabetes: p.diagnosed_diabetes },
  hypertension: {
    ...base(p),
    diabetes: p.diagnosed_diabetes,
    smoking_status: p.smoking_status,
    cigarettes_per_day: p.smoking_status === "current" ? Number(p.cigarettes_per_day) : 0,
    systolic_bp: Number(p.systolic_bp),
    diastolic_bp: Number(p.diastolic_bp),
    heart_rate: Number(p.heart_rate),
    bp_meds: p.bp_meds,
    cholesterol: Number(p.cholesterol),
    glucose: Number(p.glucose),
  },
  stroke: {
    ...base(p),
    heart_disease: p.diagnosed_heart_disease,
    hypertension: p.diagnosed_hypertension,
    smoking_status: p.smoking_status,
    glucose: Number(p.glucose),
    ever_married: p.ever_married,
    work_type: p.work_type,
    residence_type: p.residence_type,
  },
});

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}
const TOKEN_KEY = "health-access-token-v2";
export const getToken = () => sessionStorage.getItem(TOKEN_KEY);
export const clearToken = () => sessionStorage.removeItem(TOKEN_KEY);
export async function request(endpoint, { method = "GET", body, authenticated = true, signal } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (authenticated && getToken()) headers.Authorization = "Bearer " + getToken();
  let response;
  try {
    response = await fetch(API_URL + endpoint, {
      method, headers, body: body === undefined ? undefined : JSON.stringify(body),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(120000)]) : AbortSignal.timeout(120000),
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    const message = endpoint.startsWith("/predict/")
      ? "Nuk u arrit lidhja me serverin ose përgjigjja po vonon. Vlerësimi i dërguar mund të jetë ruajtur tashmë; rifresko historikun përpara se të provosh përsëri."
      : "Nuk mund të lidhet me serverin. Kontrollo që është aktiv dhe provo përsëri.";
    throw new ApiError(message, 0);
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (authenticated && response.status === 401) {
      clearToken();
      window.dispatchEvent(new Event("health-session-expired"));
    }
    const detail = Array.isArray(data?.detail)
      ? data.detail.map(item => [item.loc?.filter(x => x !== "body").join(" → "), item.msg].filter(Boolean).join(": ")).join("; ")
      : typeof data?.detail === "string" ? data.detail : "Kërkesa dështoi (" + response.status + ").";
    throw new ApiError(detail, response.status);
  }
  if (data === null) throw new ApiError("Serveri ktheu një përgjigje të papritur. Kontrollo lidhjen me API.", 0);
  return data;
}
export async function signIn(email, password) {
  const token = await request("/auth/login", { method: "POST", body: {email, password}, authenticated: false });
  sessionStorage.setItem(TOKEN_KEY, token.access_token);
}
export const register = (email, password) => request("/auth/register", {method:"POST", body:{email,password}, authenticated:false});
export const currentUser = () => request("/users/me");
export const getHistory = () => request("/history/predictions?limit=500");
export const getPrediction = id => request("/history/predictions/" + encodeURIComponent(id));
export const getTrend = (metric, signal) => request("/history/trends/" + encodeURIComponent(metric), {signal});
const predictOne = (disease, payload) => request("/predict/" + disease, {method:"POST", body:payload});

export async function predictAll(profile) {
  const payloads = buildPayloads(profile);
  const diseases = Object.keys(payloads);
  const settled = await Promise.allSettled(
    diseases.map((disease) => predictOne(disease, payloads[disease])),
  );

  return settled.reduce(
    (result, item, index) => {
      const disease = diseases[index];
      if (item.status === "fulfilled") result.predictions[disease] = item.value;
      else result.errors[disease] = item.reason?.message || "Prediction unavailable";
      return result;
    },
    { predictions: {}, errors: {} },
  );
}
