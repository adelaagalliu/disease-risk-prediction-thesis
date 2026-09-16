import {useEffect,useState} from "react";
import {getTrend} from "../api";
const metrics={bmi:["BMI","kg/m²"],blood_pressure:["Tensioni i gjakut","mmHg"],heart_rate:["Rrahjet e zemrës","rrahje/min"],glucose:["Glukoza","mg/dL"],cholesterol:["Kolesteroli","mg/dL"]};
export default function HealthTrends({revision}) {
  const [metric,setMetric]=useState("bmi"),[state,setState]=useState({loading:true,series:[],error:""});
  useEffect(()=>{
    const controller=new AbortController();
    setState({loading:true,series:[],error:""});
    const measurements=metric==="blood_pressure"?[
      {key:"systolic_bp",label:"Sistolik",color:"#397f70"},
      {key:"diastolic_bp",label:"Diastolik",color:"#bb5b68"}
    ]:[{key:metric,label:metrics[metric][0],color:"#397f70"}];
    Promise.all(measurements.map(async measurement=>{
      const data=await getTrend(measurement.key,controller.signal);
      const rows=data.filter(row=>row.value!==null&&row.value!==undefined&&row.value!==""&&Number.isFinite(Number(row.value))&&Number.isFinite(new Date(row.recorded_at).getTime())).map(row=>({...row,value:Number(row.value)})).sort((a,b)=>new Date(a.recorded_at)-new Date(b.recorded_at));
      return {...measurement,rows};
    })).then(series=>{if(!controller.signal.aborted)setState({loading:false,series,error:""});})
      .catch(error=>{if(!controller.signal.aborted)setState({loading:false,series:[],error:error.message});});
    return ()=>controller.abort();
  },[metric,revision]);
  const series=state.series,allRows=series.flatMap(item=>item.rows),unit=metrics[metric][1];
  const values=allRows.map(row=>row.value);
  const low=allRows.length?Math.min(...values):0,high=allRows.length?Math.max(...values):1;
  const padding=Math.max((high-low)*0.15,1),minimum=low-padding,range=high-low+2*padding;
  const dates=allRows.map(row=>new Date(row.recorded_at).getTime());
  const start=allRows.length?Math.min(...dates):0,end=allRows.length?Math.max(...dates):0;
  const x=row=>end===start?350:60+(new Date(row.recorded_at).getTime()-start)/(end-start)*590;
  const y=row=>175-(row.value-minimum)/range*145;
  const tableMap=new Map();
  series.forEach(item=>item.rows.forEach((row,index)=>{
    const timestamp=new Date(row.recorded_at).getTime();
    const key=row.health_record_id!=null?`${row.health_record_id}:${timestamp}`:`${item.key}:${index}:${timestamp}`;
    if(!tableMap.has(key))tableMap.set(key,{key,recorded_at:row.recorded_at,values:{}});
    tableMap.get(key).values[item.key]=row.value;
  }));
  const tableRows=[...tableMap.values()].sort((a,b)=>new Date(b.recorded_at)-new Date(a.recorded_at));
  return <section className="panel trends-panel">
    <div className="panel-heading"><div><span className="section-label">Matjet me kalimin e kohës</span><h2>Historiku i matjeve</h2></div>
      <label>Matja <select value={metric} onChange={e=>setMetric(e.target.value)}>{Object.entries(metrics).map(([key,[label]])=><option key={key} value={key}>{label}</option>)}</select></label></div>
    {state.loading?<p role="status" className="empty-state">Duke ngarkuar matjet...</p>:state.error?<p role="alert" className="form-error">{state.error}</p>:!allRows.length?<p className="empty-state">Nuk ka ende matje të ruajtura për këtë tregues.</p>:<>
      <div className="muted">{series.map(item=>{const latest=item.rows.at(-1);return <p key={item.key}><strong style={{color:item.color}}>{item.label}</strong>{latest?`: ${latest.value} ${unit} · ${new Date(latest.recorded_at).toLocaleString("sq-AL")}`:": Nuk ka matje të ruajtura."}</p>;})}</div>
      <div className="chart-wrap"><svg viewBox="0 0 680 215" role="img" aria-label={"Historiku i matjeve për "+metrics[metric][0]}>
        {[0,0.5,1].map(t=><g key={t}><line x1="60" x2="650" y1={175-t*145} y2={175-t*145} className="grid-line"/><text x="50" y={179-t*145} textAnchor="end" className="axis-label">{(minimum+t*range).toFixed(1)}</text></g>)}
        {series.map(item=><g key={item.key}>
          {item.rows.length>1&&<polyline points={item.rows.map(row=>x(row)+","+y(row)).join(" ")} fill="none" stroke={item.color} strokeWidth="3" strokeDasharray={item.key==="diastolic_bp"?"6 4":undefined}/>}
          {item.rows.map((row,index)=><circle key={item.key+":"+index} cx={x(row)} cy={y(row)} r="4" fill={item.color}><title>{item.label+": "+row.value+" "+unit+" · "+new Date(row.recorded_at).toLocaleString("sq-AL")}</title></circle>)}
        </g>)}
        <text x="60" y="205" className="axis-label">{new Date(start).toLocaleDateString("sq-AL")}</text><text x="650" y="205" textAnchor="end" className="axis-label">{new Date(end).toLocaleDateString("sq-AL")}</text>
      </svg></div>
      <details><summary>Shiko tabelën e matjeve</summary><div className="table-scroll"><table><thead><tr><th>Data</th>{series.map(item=><th key={item.key}>{item.label} ({unit})</th>)}</tr></thead><tbody>{tableRows.map(row=><tr key={row.key}><td>{new Date(row.recorded_at).toLocaleString("sq-AL")}</td>{series.map(item=><td key={item.key}>{row.values[item.key]??"—"}</td>)}</tr>)}</tbody></table></div></details>
    </>}
    <p className="causal-note">Shfaqen deri në 250 matje të fundit për secilin tregues. Çdo vlerësim shëndetësor ruan matjet e veta, ndaj një paraqitje mund të përmbajë matje të përsëritura.</p>
  </section>;
}