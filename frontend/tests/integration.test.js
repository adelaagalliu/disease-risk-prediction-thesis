import test from 'node:test';
import assert from 'node:assert/strict';
import {buildPayloads, request, signIn, predictAll, getToken} from '../src/api.js';
import {defaultProfile, validateProfile, profileSections} from '../src/profile.js';
import {loadDashboard} from '../src/storage.js';

const profile = {...defaultProfile, age:45, sex:'female', height_cm:168, weight_kg:72,
  diagnosed_diabetes:false, diagnosed_heart_disease:false, diagnosed_hypertension:false,
  diagnosed_stroke:false, high_chol:false, chol_check:true, smoking_status:'never',
  heavy_alcohol:false, physical_activity:true, fruits:true, veggies:true, general_health:'good',
  mental_health_days:2, physical_health_days:2, difficulty_walking:false, education:'college_graduate',
  income:'50000_to_74999', glucose:95, systolic_bp:120, diastolic_bp:80, heart_rate:72,
  bp_meds:false, cholesterol:190, ever_married:false, work_type:'private', residence_type:'urban'};

test('questionnaire requires actual answers and validates all sections', () => {
  assert.ok(validateProfile(defaultProfile));
  assert.equal(validateProfile(profile), null);
  assert.equal(validateProfile({...profile,age:45.5}).section,0);
  assert.equal(validateProfile({...profile,weight_kg:''}).section,0);
  assert.equal(validateProfile({...profile,diastolic_bp:130}).section,2);
  assert.equal(validateProfile({...profile,smoking_status:'current',cigarettes_per_day:0}).section,3);
  assert.equal(validateProfile({...profile,physical_activity:null}).section,3);
  // A later invalid field must not prevent navigating the current section.
  assert.equal(validateProfile({...profile,smoking_status:'current'},[profileSections[0]]),null);
});

test('payloads use server field names, types, currency and per-disease fields', () => {
  const payloads=buildPayloads({...profile,age:'45',height_cm:'168.5',weight_kg:'72'});
  assert.equal(payloads.diabetes.age,45);
  assert.equal(payloads.diabetes.height_cm,168.5);
  assert.equal(payloads.diabetes.income_display_currency,'ALL');
  assert.equal(payloads.diabetes.heavy_alcohol,false);
  assert.equal(payloads.hypertension.cigarettes_per_day,0);
  assert.equal(payloads.stroke.hypertension,false);
  assert.equal('diagnosed_diabetes' in payloads.heart_disease,false);
  assert.equal('diabetes' in payloads.diabetes,false);
  assert.equal('income' in payloads.stroke,false);
  assert.equal('high_bp' in payloads.hypertension,false);
});

test('auth token, partial prediction failures and expired sessions', async () => {
  const values=new Map();
  globalThis.sessionStorage={getItem:key=>values.get(key)||null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  let expired=false;
  globalThis.window={dispatchEvent:()=>{expired=true;}};
  const calls=[];
  globalThis.fetch=async(url,options)=>{
    calls.push({url,...options});
    if(url.endsWith('/auth/login'))return Response.json({access_token:'test-token'});
    return Response.json({detail:[{loc:['body','age'],msg:'Invalid age'}]},{status:422});
  };
  await signIn('person@example.com','test-password');
  assert.equal(getToken(),'test-token');
  assert.equal(calls[0].headers.Authorization,undefined);
  await assert.rejects(request('/predict/diabetes',{method:'POST',body:{}}),/age: Invalid age/);
  assert.equal(calls[1].headers.Authorization,'Bearer test-token');
  globalThis.fetch=async url=>url.endsWith('/stroke')?Response.json({detail:'Failed'},{status:500}):Response.json({probability:.25});
  const result=await predictAll(profile);
  assert.equal(Object.keys(result.predictions).length,3);
  assert.equal(result.errors.stroke,'Failed');
  globalThis.fetch=async()=>Response.json({detail:'Expired'},{status:401});
  await assert.rejects(request('/users/me'),/Expired/);
  assert.equal(getToken(),null);
  assert.equal(expired,true);
});

test('database history restores inputs and full saved explanations without new predictions', async () => {
  const calls=[];
  globalThis.fetch=async url=>{
    calls.push(url);
    if(url.endsWith('/users/me'))return Response.json({id:'user-1',email:'person@example.com'});
    if(url.includes('?limit=500'))return Response.json([
      {id:'new',disease:'stroke',created_at:'2026-09-14T12:00:00Z',probability:0.2,inputs:{age:46,hypertension:true,glucose:100}},
      {id:'old',disease:'diabetes',created_at:'2026-09-13T12:00:00Z',probability:0.3,inputs:{age:45,high_bp:false,heavy_alcohol:false,income_display_currency:'EUR'}}
    ]);
    return Response.json({top_factors:[{feature:'age'}],recommendations:[],probability:0.2});
  };
  const user=await loadDashboard();
  assert.equal(user.profile.age,46);
  assert.equal(user.profile.diagnosed_hypertension,true);
  assert.equal(user.profile.glucose,100);
  assert.equal(user.profile.income_display_currency,'ALL');
  assert.equal(user.profile.weight_kg,'');
  assert.equal(user.history.diabetes[0].date,'2026-09-13T12:00:00Z');
  assert.equal(user.predictions.stroke.top_factors[0].feature,'age');
  assert.equal(calls.some(url=>url.includes('/predict/')),false);
});
