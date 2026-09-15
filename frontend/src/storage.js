import { currentUser, getHistory, getPrediction, getToken, clearToken, signIn, register } from "./api.js";
import { defaultProfile } from "./profile.js";

export async function loadDashboard() {
  const account = await currentUser();
  const rows = await getHistory();
  const history = {}, latest = {}, profile = {...defaultProfile};
  for (const row of [...rows].reverse()) {
    (history[row.disease] ||= []).push({date: row.created_at, probability: row.probability});
    latest[row.disease] = row;
    const inputs = {...row.inputs};
    if ('high_bp' in inputs) inputs.diagnosed_hypertension = inputs.high_bp;
    if ('hypertension' in inputs) inputs.diagnosed_hypertension = inputs.hypertension;
    if ('diabetes' in inputs) inputs.diagnosed_diabetes = inputs.diabetes;
    if ('heart_disease' in inputs) inputs.diagnosed_heart_disease = inputs.heart_disease;
    if ('stroke' in inputs) inputs.diagnosed_stroke = inputs.stroke;
    Object.keys(profile).forEach(key => { if (key in inputs) profile[key] = inputs[key]; });
  }
  profile.income_display_currency = "ALL";
  const predictions = {}, errors = {};
  await Promise.all(Object.entries(latest).map(async ([disease, row]) => {
    try { predictions[disease] = await getPrediction(row.id); }
    catch (error) {
      if (error.status === 401) throw error;
      predictions[disease] = {...row, created_at:row.created_at};
      errors[disease] = "Vlerësimi i ruajtur u ngarkua, por shpjegimi i tij nuk mund të merrej: " + error.message;
    }
  }));
  return {...account, name:account.email.split('@')[0], profile, predictions, errors, history};
}
export const getSession = () => getToken() ? loadDashboard() : Promise.resolve(null);
export async function login(email,password) { await signIn(email,password); return loadDashboard(); }
export async function createAccount({email,password}) {
  await register(email,password);
  try { return await login(email,password); }
  catch (error) { throw new Error("Llogaria u krijua. Të lutem hyr për të vazhduar. " + error.message); }
}
export function logout() { clearToken(); }
