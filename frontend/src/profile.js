export const defaultProfile = {
  age: "",
  sex: "",
  height_cm: "",
  weight_kg: "",
  diagnosed_diabetes: null,
  diagnosed_heart_disease: null,
  diagnosed_hypertension: null,
  diagnosed_stroke: null,
  high_chol: null,
  chol_check: null,
  smoking_status: "",
  cigarettes_per_day: 0,
  heavy_alcohol: null,
  income_display_currency: "ALL",
  physical_activity: null,
  fruits: null,
  veggies: null,
  general_health: "",
  mental_health_days: "",
  physical_health_days: "",
  difficulty_walking: null,
  education: "",
  income: "",
  glucose: "",
  systolic_bp: "",
  diastolic_bp: "",
  heart_rate: "",
  bp_meds: null,
  cholesterol: "",
  ever_married: null,
  work_type: "",
  residence_type: "",
};

export const profileSections = [
  {
    title: "Rreth teje",
    description: "Matjet bazë që përdoren për të përcaktuar grupmoshën dhe indeksin e masës trupore (BMI).",
    fields: [
      { name: "age", label: "Mosha", type: "number", min: 18, max: 120, suffix: "vjeç" },
      {
        name: "sex",
        label: "Gjinia",
        type: "select",
        options: [["female", "Femër"], ["male", "Mashkull"]],
      },
      { name: "height_cm", label: "Gjatësia", type: "number", min: 100, max: 250, suffix: "cm" },
      { name: "weight_kg", label: "Pesha", type: "number", min: 30, max: 300, suffix: "kg" },
    ],
  },
  {
    title: "Historiku shëndetësor",
    description: "Na trego për diagnozat e mëparshme dhe kontrollet rutinë.",
    fields: [
      { name: "diagnosed_diabetes", label: "Ke pasur më parë një diagnozë të diabetit", type: "boolean" },
      { name: "diagnosed_heart_disease", label: "Ke pasur më parë një diagnozë të sëmundjes së zemrës", type: "boolean" },
      { name: "diagnosed_hypertension", label: "Ke pasur më parë një diagnozë të tensionit të lartë të gjakut", type: "boolean" },
      { name: "diagnosed_stroke", label: "Ke pësuar më parë një goditje në tru", type: "boolean" },
      { name: "high_chol", label: "Ke pasur më parë një diagnozë të kolesterolit të lartë", type: "boolean" },
      { name: "chol_check", label: "Ke kontrolluar kolesterolin gjatë pesë viteve të fundit", type: "boolean" },
      { name: "bp_meds", label: "Aktualisht merr ilaçe për tensionin e gjakut", type: "boolean" },
    ],
  },
  {
    title: "Matjet aktuale",
    description: "Vendos vlerat nga matjet e fundit. Mos hamendëso vlerat e analizave ose të tensionit të gjakut që nuk i ke.",
    fields: [
      { name: "systolic_bp", label: "Tensioni sistolik", type: "number", min: 70, max: 260, suffix: "mmHg" },
      { name: "diastolic_bp", label: "Tensioni diastolik", type: "number", min: 40, max: 160, suffix: "mmHg" },
      { name: "heart_rate", label: "Rrahjet e zemrës", type: "number", min: 30, max: 220, suffix: "rrahje/min" },
      { name: "cholesterol", label: "Kolesteroli", type: "number", min: 80, max: 700, suffix: "mg/dL" },
      { name: "glucose", label: "Glukoza", type: "number", min: 30, max: 700, suffix: "mg/dL" },
    ],
  },
  {
    title: "Stili i jetesës",
    description: "Këto përgjigje përdoren si të dhëna për faktorët e stilit të jetesës në model.",
    fields: [
      { name: "fruits", label: "Konsumon fruta të paktën një herë në ditë?", type: "boolean" },
      { name: "veggies", label: "Konsumon perime të paktën një herë në ditë?", type: "boolean" },
      { name: "physical_activity", label: "Ke kryer aktivitet fizik të rregullt gjatë 30 ditëve të fundit", type: "boolean" },
      { name: "heavy_alcohol", label: "Konsum i lartë i alkoolit", type: "boolean" },
      {
        name: "smoking_status",
        label: "Konsumimi i duhanit",
        type: "select",
        options: [["never", "Nuk kam pirë kurrë duhan"], ["former", "Kam pirë më parë, por e kam lënë"], ["current", "Pi duhan aktualisht"], ["unknown", "Preferoj të mos përgjigjem"]],
      },
      { name: "cigarettes_per_day", label: "Cigare në ditë", type: "number", min: 0, max: 100, suffix: "në ditë", showWhen: ["smoking_status", "current"] },
    ],
  },
  {
    title: "Mirëqenia",
    description: "Një pasqyrë e shkurtër e shëndetit të përgjithshëm, fizik dhe mendor.",
    fields: [
      {
        name: "general_health",
        label: "Si do ta vlerësoje gjendjen tënde të përgjithshme shëndetësore?",
        type: "select",
        options: [["excellent", "E shkëlqyer"], ["very_good", "Shumë e mirë"], ["good", "E mirë"], ["fair", "Mesatare"], ["poor", "E dobët"]],
      },
      { name: "mental_health_days", label: "Gjatë muajit të fundit, për sa ditë shëndeti yt mendor nuk ka qenë i mirë?", type: "number", min: 0, max: 30, suffix: "ditë" },
      { name: "physical_health_days", label: "Gjatë muajit të fundit, për sa ditë shëndeti yt fizik nuk ka qenë i mirë?", type: "number", min: 0, max: 30, suffix: "ditë" },
      { name: "difficulty_walking", label: "Ke vështirësi të mëdha në ecje ose në ngjitjen e shkallëve?", type: "boolean" },
    ],
  },
  {
    title: "Të dhëna të tjera",
    description: "Të dhëna shtesë që kërkohen nga modelet kërkimore.",
    fields: [
      {
        name: "education",
        label: "Cili është niveli më i lartë arsimor që ke ndjekur?",
        type: "select",
        options: [
          ["never_attended_or_kindergarten", "Pa shkollim ose vetëm kopsht"],
          ["elementary", "Arsim fillor"],
          ["some_high_school", "Arsim i mesëm i papërfunduar"],
          ["high_school_graduate", "Arsim i mesëm i përfunduar"],
          ["some_college_or_technical_school", "Studime universitare të papërfunduara ose shkollë teknike"],
          ["college_graduate", "Arsim universitar i përfunduar"],
        ],
      },
      {
        name: "income",
        label: "Të ardhurat vjetore të familjes",
        type: "select",
        options: [
          ["less_than_10000", "Nën 800,000 lekë"],
          ["10000_to_14999", "800,000–1,200,000 lekë"],
          ["15000_to_19999", "1,200,000–1,600,000 lekë"],
          ["20000_to_24999", "1,600,000–2,000,000 lekë"],
          ["25000_to_34999", "2,000,000–2,800,000 lekë"],
          ["35000_to_49999", "2,800,000–4,000,000 lekë"],
          ["50000_to_74999", "4,000,000–6,000,000 lekë"],
          ["75000_or_more", "6,000,000 lekë ose më shumë"],
        ],
      },
      { name: "ever_married", label: "A ke qenë ndonjëherë i/e martuar?", type: "boolean" },
      {
        name: "work_type",
        label: "Lloji i punësimit",
        type: "select",
        options: [["private", "Sektor privat"], ["self_employed", "Vetëpunësim"], ["government_job", "Sektor shtetëror"], ["never_worked", "Nuk kam punuar asnjëherë"]],
      },
      {
        name: "residence_type",
        label: "Vendbanimi",
        type: "select",
        options: [["urban", "Zonë urbane"], ["rural", "Zonë rurale"]],
      },
    ],
  },
];

export const formatDisplayValue = (key, value) => {
  if (value === "" || value == null) return "Nuk është dhënë";
  if (typeof value === "boolean") return value ? "Po" : "Jo";
  const translatedValues = {
    female: "Femër",
    male: "Mashkull",
    current: "Aktualisht",
    former: "I mëparshëm",
    never: "Asnjëherë",
    unknown: "E panjohur",
  };
  const labels = {
    height_cm: value + " cm",
    weight_kg: value + " kg",
    glucose: value + " mg/dL",
    systolic_bp: value + " mmHg",
    diastolic_bp: value + " mmHg",
    heart_rate: value + " rrahje/min",
    cholesterol: value + " mg/dL",
    cigarettes_per_day: value + " / ditë",
    alcohol_drinks_per_day: value + " / ditë",
  };
  return labels[key] || translatedValues[value] || String(value).replaceAll("_", " ");
};

export function validateProfile(profile, sections = profileSections) {
  for (const section of sections) for (const field of section.fields) {
    if (field.showWhen && profile[field.showWhen[0]] !== field.showWhen[1]) continue;
    const value = profile[field.name];
    if (field.type === "boolean" && typeof value !== "boolean") return {section:profileSections.indexOf(section), message:"Të lutem përgjigju: " + field.label};
    if (field.type === "select" && !field.options.some(([key]) => key === value)) return {section:profileSections.indexOf(section), message:"Të lutem zgjidh: " + field.label};
    if (field.type === "number" && (value === "" || value == null || !Number.isFinite(Number(value)) || Number(value) < field.min || Number(value) > field.max || (["age","mental_health_days","physical_health_days"].includes(field.name) && !Number.isInteger(Number(value))))) return {section:profileSections.indexOf(section), message:field.label + " duhet të jetë midis " + field.min + " dhe " + field.max + "."};
  }
  if (sections.some(s => s.fields.some(f => f.name === "diastolic_bp")) && Number(profile.diastolic_bp) >= Number(profile.systolic_bp)) return {section:2,message:"Tensioni diastolik duhet të jetë më i ulët se tensioni sistolik."};
  if (sections.some(s => s.fields.some(f => f.name === "cigarettes_per_day")) && profile.smoking_status === "current" && Number(profile.cigarettes_per_day) <= 0) return {section:3,message:"Nëse pi duhan aktualisht, vendos numrin e cigareve që konsumon në ditë."};
  return null;
}