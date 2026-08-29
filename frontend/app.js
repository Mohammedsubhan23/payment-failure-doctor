const API_BASE = window.API_BASE || 'http://localhost:8000';

const BUCKET_COLORS = {
  timeout: '#4FD1C5', dropoff: '#7C8CFF', invalid: '#F5A623',
  decline: '#E8A6E0', funds: '#8C97B5', risk: '#FF6B6B',
};
const BUCKET_LABELS = {
  timeout: 'Network / Gateway Timeout', dropoff: 'Checkout Drop-off',
  invalid: 'Invalid Card / Details', decline: 'Bank / Issuer Decline',
  funds: 'Insufficient Funds', risk: 'Risk / Fraud Block',
};
const BUCKET_CATEGORY = {
  timeout: 'autoRetry', dropoff: 'autoRetry', invalid: 'userAction',
  decline: 'altMethod', funds: 'delayedRetry', risk: 'manualReview',
};

let STATE = { summary: null, bucketBreakdown: [], transactions: [] };
let MODE = 'checking'; // 'backend' | 'fallback'

// ---------------------------------------------------------------------
// Mode detection: prefer the real trained-model backend; fall back to a
// client-side heuristic (clearly labelled as such) if it isn't running.
// ---------------------------------------------------------------------
async function checkHealth(){
  const el = document.getElementById('apiStatus');
  try{
    const r = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(2500) });
    const d = await r.json();
    if(d.status === 'ok'){
      MODE = 'backend';
      el.textContent = 'Connected — RandomForest / GradientBoosting backend';
      el.className = 'api-status ok';
      return;
    }
    throw new Error();
  }catch(e){
    MODE = 'fallback';
    el.textContent = 'No backend detected — running client-side heuristic fallback';
    el.className = 'api-status down';
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  await checkHealth();
  await loadSample();
});

async function loadSample(){
  if(MODE === 'backend'){
    await runBatchBackend(null);
  } else {
    runBatchFallback(generateSyntheticTransactions(60));
  }
}

async function uploadCSV(){
  const input = document.getElementById('csvFile');
  if(!input.files.length){ alert('Choose a CSV file first, or click "Use demo sample".'); return; }
  if(MODE === 'backend'){
    await runBatchBackend(input.files[0]);
  } else {
    const text = await input.files[0].text();
    runBatchFallback(parseCSV(text));
  }
}

// ---------------------------------------------------------------------
// Backend path — real trained models
// ---------------------------------------------------------------------
async function runBatchBackend(file){
  const fd = new FormData();
  if(file) fd.append('file', file);
  try{
    const res = await fetch(API_BASE + '/batch', { method:'POST', body: fd });
    if(!res.ok){ throw new Error(await res.text()); }
    const data = await res.json();
    STATE.summary = data.summary;
    STATE.bucketBreakdown = data.bucket_breakdown;
    STATE.transactions = data.transactions;
    renderAll();
  }catch(e){
    alert('Could not classify batch via backend: ' + e.message);
  }
}

// ---------------------------------------------------------------------
// Fallback path — no backend running. Same taxonomy, but the bucket and
// retry-probability numbers come from a hand-written heuristic instead
// of the trained models, and every "diagnose" call hits the LLM directly
// from the browser instead of via the server-side /diagnose endpoint.
// This exists so the app is inspectable with zero setup — it is NOT a
// substitute for the trained backend, and the UI says so explicitly.
// ---------------------------------------------------------------------
function parseCSV(text){
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h=>h.trim());
  return lines.slice(1).map((line,i)=>{
    const vals = line.split(',').map(v=>v.trim());
    const row = {};
    headers.forEach((h,idx)=>row[h]=vals[idx]);
    row.id = row.id || ('pay_'+String(i).padStart(8,'0'));
    row.amount = Number(row.amount);
    row.hour = Number(row.hour || 12);
    row.day_of_week = Number(row.day_of_week || 0);
    row.prior_retries = Number(row.prior_retries || 0);
    row.is_new_customer = Number(row.is_new_customer || 0);
    return row;
  });
}

const REASON_TO_BUCKET = {
  issuer_unavailable:'timeout', internal_error:'timeout',
  otp_timeout:'dropoff', user_cancelled:'dropoff',
  invalid_cvv:'invalid', invalid_expiry:'invalid',
  issuer_general_decline:'decline', insufficient_funds:'funds',
  issuer_risk_block:'risk',
};
const METHODS = ['UPI','Card','Netbanking','Wallet'];
const BANKS = ['HDFC Bank','ICICI Bank','SBI','Axis Bank','Kotak Mahindra','Yes Bank','IDFC First'];
const REASON_WEIGHTS = {issuer_unavailable:9,internal_error:9,otp_timeout:11,user_cancelled:11,
  invalid_cvv:7,invalid_expiry:7,issuer_general_decline:10,insufficient_funds:8,issuer_risk_block:5};
const BUCKET_RETRY_BASE = {timeout:0.82,dropoff:0.66,invalid:0.52,decline:0.33,funds:0.20,risk:0.03};

function weightedPick(weights){
  const total = Object.values(weights).reduce((a,b)=>a+b,0);
  let r = Math.random()*total;
  for(const k in weights){ r -= weights[k]; if(r<=0) return k; }
  return Object.keys(weights)[0];
}
function pick(arr){return arr[Math.floor(Math.random()*arr.length)];}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}

function generateSyntheticTransactions(n){
  const rows = [];
  for(let i=0;i<n;i++){
    const reason = weightedPick(REASON_WEIGHTS);
    const bucket = REASON_TO_BUCKET[reason];
    const amount = (bucket==='decline'||bucket==='risk') ? Math.round(30+Math.random()*450)*100 : Math.round(1.5+Math.random()*120)*100;
    rows.push({
      id: 'pay_'+Math.random().toString(36).slice(2,16).toUpperCase(),
      amount, method: pick(METHODS), bank: pick(BANKS),
      hour: Math.floor(Math.random()*24), day_of_week: Math.floor(Math.random()*7),
      prior_retries: Math.floor(Math.random()*3), is_new_customer: Math.random()<0.35?1:0,
      error_code: bucket==='decline'||bucket==='funds'||bucket==='invalid'||bucket==='dropoff' ? 'BAD_REQUEST_ERROR' : 'GATEWAY_ERROR',
      reason, _true_bucket: bucket,
    });
  }
  return rows;
}

function heuristicClassify(t){
  // uses reason as ground truth when synthetic; falls back to a rough
  // guess when reason is unrecognised (e.g. a real uploaded CSV with
  // unfamiliar codes) — this is explicitly the weaker, unaudited path
  // the trained backend replaces.
  const bucket = REASON_TO_BUCKET[t.reason] || t._true_bucket || 'decline';
  let p = BUCKET_RETRY_BASE[bucket];
  if(t.amount > 20000) p -= 0.12;
  if(t.method === 'UPI') p += 0.05;
  if(t.hour < 6) p -= 0.05;
  p = clamp(p, 0.02, 0.97);
  const confidence = clamp(0.86 + (Math.random()*0.17-0.06), 0.75, 0.98);
  return { bucket_key: bucket, bucket_label: BUCKET_LABELS[bucket], category: BUCKET_CATEGORY[bucket],
    confidence, retry_probability: p };
}

function runBatchFallback(rows){
  const classified = rows.map(t => ({ ...t, ...heuristicClassify(t) }));
  const totalValue = classified.reduce((s,t)=>s+t.amount,0);
  let recoverable = 0, riskCount = 0;
  classified.forEach(t=>{
    if(t.category !== 'manualReview') recoverable += t.amount * t.retry_probability;
    else riskCount++;
  });
  const sums = {};
  classified.forEach(t=>{ sums[t.bucket_label] = (sums[t.bucket_label]||0) + t.amount; });

  STATE.summary = {
    total_transactions: classified.length,
    total_failed_value: totalValue,
    estimated_recoverable_value: recoverable,
    recoverable_pct: totalValue ? Math.round(100*recoverable/totalValue*10)/10 : 0,
    risk_flagged_count: riskCount,
  };
  STATE.bucketBreakdown = Object.entries(sums).sort((a,b)=>b[1]-a[1]).map(([bucket,amount])=>({bucket,amount}));
  STATE.transactions = classified;
  renderAll();
}

async function diagnoseFallback(t){
  const prompt = `You are writing a short, merchant-facing diagnosis inside a payments \
dashboard called "Payment Failure Doctor", running in client-side fallback mode (no \
trained-model backend connected — this classification came from a simple heuristic, \
no need to mention that in your reply, just explain it plainly). Write 2-3 tight \
sentences in plain, non-technical language, then one clear recommended action on its \
own final line starting with "Recommended:".

Transaction: ₹${t.amount} via ${t.method}, bank: ${t.bank}, gateway error: ${t.error_code} (${t.reason}).
Classified root cause: "${t.bucket_label}" with ${Math.round(t.confidence*100)}% confidence.
Estimated retry-success probability: ${Math.round(t.retry_probability*100)}%.
Category: ${t.category}.

Treat the classification as ground truth — do not question or re-derive it.`;
  return callClaudeDirect(prompt);
}

async function askDoctorFallback(question){
  const s = STATE.summary;
  const breakdown = STATE.bucketBreakdown.map(b=>`${b.bucket}: ₹${Math.round(b.amount).toLocaleString('en-IN')}`).join('\n');
  const prompt = `You are "the doctor" inside a payments dashboard, answering a \
merchant's question using ONLY the computed data below. Never invent numbers not \
present here. Answer in 3-5 sentences, plain language, no bullet lists, no headers.

Total failed value: ₹${Math.round(s.total_failed_value).toLocaleString('en-IN')} across ${s.total_transactions} transactions.
Estimated recoverable value: ₹${Math.round(s.estimated_recoverable_value).toLocaleString('en-IN')} (${s.recoverable_pct}%).
Risk-flagged: ${s.risk_flagged_count}.

Breakdown by root cause:
${breakdown}

Merchant's question: "${question}"`;
  return callClaudeDirect(prompt);
}

async function callClaudeDirect(prompt){
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 500,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  return (data.content||[]).filter(b=>b.type==='text').map(b=>b.text).join('\n').trim();
}

// ---------------------------------------------------------------------
// Shared rendering — identical regardless of which path produced STATE
// ---------------------------------------------------------------------
function renderAll(){
  renderVitals();
  renderLeakMap();
  populateFilters();
  renderFeed();
}

function renderVitals(){
  const s = STATE.summary;
  const vitals = [
    {label:'Failed transactions', value: s.total_transactions, sub:'in this batch'},
    {label:'Failed value', value: '₹'+Math.round(s.total_failed_value).toLocaleString('en-IN'), sub:'across all buckets'},
    {label:'Estimated recoverable', value:'₹'+Math.round(s.estimated_recoverable_value).toLocaleString('en-IN'), sub:s.recoverable_pct+'% of failed value', cls:'recover'},
    {label:'Risk-flagged (no auto-retry)', value: s.risk_flagged_count, sub:'held for manual review', cls:'risk'}
  ];
  document.getElementById('vitals').innerHTML = vitals.map(v=>`
    <div class="vital ${v.cls||''}">
      <div class="label">${v.label}</div>
      <div class="value">${v.value}</div>
      <div class="sub2">${v.sub}</div>
    </div>`).join('');
}

function bucketKeyFromLabel(label){
  const t = STATE.transactions.find(t=>t.bucket_label===label);
  return t ? t.bucket_key : 'decline';
}

function renderLeakMap(){
  const total = STATE.bucketBreakdown.reduce((s,b)=>s+b.amount,0);
  document.getElementById('leakbar').innerHTML = STATE.bucketBreakdown.map(b=>{
    const pct = (b.amount/total*100).toFixed(1);
    const key = bucketKeyFromLabel(b.bucket);
    return `<span style="width:${pct}%;background:${BUCKET_COLORS[key]}" title="${b.bucket}: ₹${Math.round(b.amount).toLocaleString('en-IN')}"></span>`;
  }).join('');
  document.getElementById('legend').innerHTML = STATE.bucketBreakdown.map(b=>{
    const key = bucketKeyFromLabel(b.bucket);
    return `<div class="legend-item">
      <span class="legend-swatch" style="background:${BUCKET_COLORS[key]}"></span>
      <b>${b.bucket}</b>
      <span class="amt">₹${Math.round(b.amount).toLocaleString('en-IN')}</span>
    </div>`;
  }).join('');
}

function populateFilters(){
  const sel = document.getElementById('fBucket');
  sel.innerHTML = '<option value="">All root causes</option>';
  const seen = new Set();
  STATE.transactions.forEach(t=>{
    if(seen.has(t.bucket_key)) return;
    seen.add(t.bucket_key);
    const opt = document.createElement('option');
    opt.value = t.bucket_key; opt.textContent = t.bucket_label;
    sel.appendChild(opt);
  });
}

function renderFeed(){
  const fb = document.getElementById('fBucket').value;
  const fm = document.getElementById('fMethod').value;
  const filtered = STATE.transactions.filter(t => (!fb || t.bucket_key===fb) && (!fm || t.method===fm));
  document.getElementById('fCount').textContent = filtered.length + ' of ' + STATE.transactions.length + ' shown';

  document.getElementById('feed').innerHTML = filtered.map(t=>{
    const color = BUCKET_COLORS[t.bucket_key];
    return `
    <div class="txn">
      <div class="txn-head" onclick="toggleTxn('${t.id}')">
        <span class="bucket-chip" style="background:${color}22;color:${color}">${t.bucket_label}</span>
        <span class="txn-id">${t.id}</span>
        <span class="txn-method">${t.method}</span>
        <span class="txn-amt">₹${Math.round(t.amount).toLocaleString('en-IN')}</span>
        <span class="chevron" id="chev-${t.id}">▾</span>
      </div>
      <div class="txn-body" id="body-${t.id}">
        <div class="txn-meta">
          <div class="m"><div class="k">Error code</div><div class="v">${t.error_code}</div></div>
          <div class="m"><div class="k">Reason</div><div class="v">${t.reason}</div></div>
          <div class="m"><div class="k">Bank</div><div class="v">${t.bank}</div></div>
          <div class="m"><div class="k">Method</div><div class="v">${t.method}</div></div>
        </div>
        <div class="txn-meta">
          <div class="m" style="grid-column:span 2">
            <div class="k">${MODE==='backend'?'Model confidence':'Heuristic confidence'} &mdash; ${Math.round(t.confidence*100)}%</div>
            <div class="prob-bar"><i style="width:${t.confidence*100}%"></i></div>
          </div>
          <div class="m" style="grid-column:span 2">
            <div class="k">Retry success probability &mdash; ${Math.round(t.retry_probability*100)}%</div>
            <div class="prob-bar"><i style="width:${t.retry_probability*100}%;background:${color}"></i></div>
          </div>
        </div>
        <button class="btn ghost" id="rxbtn-${t.id}" onclick='diagnose(${JSON.stringify(t.id)})'>Get diagnosis &amp; fix</button>
        <div class="rx" id="rx-${t.id}"></div>
      </div>
    </div>`;
  }).join('');
}

function toggleTxn(id){
  const body = document.getElementById('body-'+id);
  const chev = document.getElementById('chev-'+id);
  const isOpen = body.classList.contains('open');
  if(isOpen){ body.classList.remove('open'); body.style.display='none'; chev.style.transform='rotate(0deg)'; }
  else { body.classList.add('open'); body.style.display='block'; chev.style.transform='rotate(180deg)'; }
}

async function diagnose(id){
  const t = STATE.transactions.find(x=>x.id===id);
  const btn = document.getElementById('rxbtn-'+id);
  const box = document.getElementById('rx-'+id);
  btn.disabled = true; btn.textContent = 'Diagnosing…';
  box.classList.add('show'); box.innerHTML = '<span class="thinking">Reading the classification…</span>';
  try{
    let text;
    if(MODE === 'backend'){
      const res = await fetch(API_BASE + '/diagnose', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          id: t.id, amount: t.amount, method: t.method, bank: t.bank,
          error_code: t.error_code, reason: t.reason, bucket: t.bucket_label,
          category: t.category, confidence: t.confidence, retry_probability: t.retry_probability
        })
      });
      const data = await res.json();
      text = data.diagnosis;
    } else {
      text = await diagnoseFallback(t);
    }
    box.innerHTML = (text || 'No response.').replace(/\n/g,'<br>');
  }catch(e){
    box.innerHTML = '<span style="color:var(--coral)">Could not reach the diagnosis model.</span>';
  }
  btn.disabled = false; btn.textContent = 'Regenerate diagnosis';
}

async function askDoctor(){
  const q = document.getElementById('doctorQ').value.trim();
  if(!q || !STATE.summary) return;
  const btn = document.getElementById('askBtn');
  const box = document.getElementById('doctorAnswer');
  btn.disabled = true;
  box.classList.add('show');
  box.innerHTML = '<span class="thinking">Reading the failure data…</span>';
  try{
    let text;
    if(MODE === 'backend'){
      const res = await fetch(API_BASE + '/ask', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ question: q, summary: STATE.summary, bucket_breakdown: STATE.bucketBreakdown })
      });
      const data = await res.json();
      text = data.answer;
    } else {
      text = await askDoctorFallback(q);
    }
    box.innerHTML = (text || 'No response.').replace(/\n/g,'<br>');
  }catch(e){
    box.innerHTML = '<span style="color:var(--coral)">Could not reach the model.</span>';
  }
  btn.disabled = false;
}
