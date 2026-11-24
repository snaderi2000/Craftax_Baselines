/* Standalone UI: no libraries, network, or labels in cluster fitting. */
'use strict';
(() => {
  const D = JSON.parse(document.getElementById('dataset').textContent);
  const $ = id => document.getElementById(id);
  const pretty = s => s.replaceAll('_', ' ');
  const fmt = n => n == null || !Number.isFinite(n) ? '—' : (100 * n).toFixed(1) + '%';
  const num = n => Number(n).toLocaleString();
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const color = i => `hsl(${(i * 137.508 + 20) % 360} 65% 65%)`;
  let C, cluster = null, page = 0, examples = [], points = [], fitBusy = false;
  const textures = {};
  const imagesReady = Promise.all(Object.entries(D.textures.images).map(([name, src]) => new Promise(resolve => {
    const img = new Image(); img.onload = resolve; img.onerror = resolve; img.src = src; textures[name] = img;
  })));
  $('provenance').textContent = `${num(D.meta.episodes)} episodes · checkpoint ${num(D.meta.checkpoint)} · ${num(D.meta.achievement_reward)} reward = +1 transitions · ${num(D.rows.length)} inspectable examples`;
  D.names.forEach((name, i) => $('event').add(new Option(pretty(name), String(i))));
  $('protocol').innerHTML = `<p>Policy: <code>${esc(D.meta.policy_run)}</code>, checkpoint <code>${D.meta.checkpoint}</code>, ${esc(D.meta.environment)}. The source batch contains ${num(D.meta.original_transitions)} original transitions from ${num(D.meta.episodes)} episodes: ${num(D.meta.reference_episodes)} reference and ${num(D.meta.test_episodes)} held out. Episode IDs never overlap between splits.</p>
    <p>The source bank retains every nonzero scalar reward plus uniformly sampled timesteps. This explorer selects ONLY reward = +1 (absolute tolerance 1e-6, no relative tolerance): ${num(D.meta.achievement_reference)} reference and ${num(D.meta.achievement_test)} held-out transitions. All selected transitions in this dataset unlock one new achievement. Health-only +0.1 transitions are excluded from fitting, scoring and inspection. The ${num(D.meta.achievement_with_health_change)} achievement transitions with simultaneous health changes (such as +0.9 or +1.1 reward) are also excluded.</p>
    <p>Each clustered point is a TRANSITION DIFFERENCE, not an isolated state: h(s′) − h(s) for actor/critic hidden layers, or s′ − s for raw observations. The action/inventory control concatenates action one-hot with a normalized inventory change. Every vector is L2-normalized before Euclidean k-means. Centroids were refitted from scratch on the +1 reference transitions using CPU k-means++, 25 Lloyd iterations and a fixed 32/64/128 sweep. Reward selects transitions but is not a feature; achievement flags annotate diagnostics only.</p>
    <p>Cached scores evaluate all eligible held-out transitions. Accuracy uses the majority label learned from reference assignments. Macro F1 averages supported achievements, excluding health-only / other events. Purity uses held-out labels to describe composition. NMI and ARI compare partitions without assuming cluster IDs have meaning. Multiple achievements are a separate diagnostic class.</p>
    <p>Inspection is a subset: ${esc(D.meta.sampling)} This preserves small clusters for inspection but changes visible frequencies. Counts, heatmaps and scores use the full experiment. The PCA view is fitted on at most 2,048 reference examples from the inspection subset; the plotted two dimensions are never used to fit k-means. Distances are squared distances in the original normalized representation.</p>
    <p>Images reconstruct the agent’s local 7×9 symbolic observation using Craftax textures. Gold outlines mark changed cells in observation coordinates; movement can shift the whole view. Inventory and internal levels are in game units. The true post-terminal observation is retained. Hidden world state is not reconstructed; arrows use a generic icon because symbolic observations do not encode arrow direction.</p>
    <p>These are mature-policy, reward-enriched offline diagnostics. They measure grouping and event recovery on held-out episodes, and do not establish causal concepts or improved policy performance. Comparing many settings on the same held-out split is exploratory.</p>`;

  function availableSeeds() {
    const seeds = [...new Set(D.configs.filter(c => c.method === $('method').value && c.mode === $('mode').value && c.k === +$('k').value).map(c => c.seed))].sort((a,b) => a-b);
    const old = +$('seed').value;
    $('seed').replaceChildren(...seeds.map(s => new Option(s, s)));
    if (seeds.includes(old)) $('seed').value = old;
    $('seed').disabled = seeds.length === 0;
  }
  function selectConfig() {
    availableSeeds();
    C = D.configs.find(c => c.method === $('method').value && c.mode === $('mode').value && c.k === +$('k').value && c.seed === +$('seed').value);
    if (!C) {
      $('status').textContent = 'No cached model for this filter. Open “Fit a new k-means model” below and use the local server.';
      $('metrics').replaceChildren(); $('cluster-table').replaceChildren(); $('heatmap').replaceChildren(); $('class-table').replaceChildren(); $('example').textContent = 'Select a cached configuration or fit this filter.'; $('composition').replaceChildren();
      $('scatter').getContext('2d').clearRect(0,0,$('scatter').width,$('scatter').height); points = [];
      return;
    }
    cluster = null; page = 0;
    $('status').textContent = `${C.cached ? 'Cached +1-only fit' : 'New +1-only CPU fit'} · fit ${num(C.fit_n)} reference transitions · assign ${num(C.reference_n)} reference / ${num(C.test_n)} held out · k=${C.k} / seed=${C.seed}`;
    const m = C.metrics;
    const cards = [['Held-out accuracy', fmt(m.mapped_accuracy), 'Fixed reference-majority mapping'],
      ['Achievement macro F1', fmt(m.achievement_macro_f1), 'Supported achievements only'],
      ['Held-out purity', fmt(m.heldout_cluster_purity), 'Descriptive; not predictive accuracy'],
      ['NMI', m.nmi.toFixed(3), 'Normalized mutual information'],
      ['ARI', m.ari.toFixed(3), 'Adjusted Rand index'],
      ['Held-out transitions', num(C.test_n), `${m.occupied_test_clusters} occupied clusters`]];
    $('metrics').innerHTML = cards.map(([label,v,hint]) => `<div class="metric"><small>${label}</small><strong>${v}</strong><small>${hint}</small></div>`).join('');
    renderHeatmap(); renderClasses(); renderSelection();
  }
  function sum(a) { return a.reduce((x,y) => x+y, 0); }
  function clusterPurity(c) { const a=C.test_table[c], n=sum(a); return n ? Math.max(...a)/n : null; }
  function renderClusters() {
    if (!C) return;
    let ids=Array.from({length:C.k},(_,i)=>i), query=$('cluster-search').value.toLowerCase();
    ids=ids.filter(i => `${i} ${pretty(D.names[C.mapping[i]])}`.includes(query));
    const mode=$('cluster-sort').value;
    ids.sort((a,b) => mode==='id' ? a-b : mode==='purity' ? (clusterPurity(a)??2)-(clusterPurity(b)??2) : sum(C.test_table[b])-sum(C.test_table[a]));
    $('cluster-table').innerHTML=ids.map(i=>`<tr data-cluster="${i}" class="${i===cluster?'selected':''}"><td><button data-cluster="${i}" aria-label="Inspect cluster ${i}"><span class="dot" style="background:${color(i)}"></span>${i}</button></td><td>${pretty(D.names[C.mapping[i]])}</td><td class="nowrap">${num(sum(C.reference_table[i]))} / ${num(sum(C.test_table[i]))}</td><td>${fmt(clusterPurity(i))}</td></tr>`).join('');
  }
  function chooseCluster(i) { cluster=i; page=0; renderSelection(); }
  function renderSelection() { if (!C) return; renderClusters(); drawScatter(); renderExamples(); }
  function filteredRows() {
    if (!C) return [];
    return D.rows.map((row,i)=>i).filter(i => C.assignments[i]>=0 && (cluster===null || C.assignments[i]===cluster) &&
      ($('split').value==='all' || D.rows[i].split===+$('split').value) && ($('event').value==='all' || D.rows[i].label===+$('event').value));
  }
  function drawScatter() {
    if (!C) return;
    const canvas=$('scatter'), ctx=canvas.getContext('2d'), w=canvas.clientWidth, h=360, ratio=window.devicePixelRatio||1;
    canvas.width=w*ratio; canvas.height=h*ratio; ctx.scale(ratio,ratio); ctx.clearRect(0,0,w,h);
    const projection=D.projections[C.method], xy=projection.points;
    const visible=D.rows.map((r,i)=>i).filter(i=>C.assignments[i]>=0 && ($('split').value==='all'||D.rows[i].split===+$('split').value));
    if (!visible.length) return;
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    visible.forEach(i=>{xmin=Math.min(xmin,xy[i][0]);xmax=Math.max(xmax,xy[i][0]);ymin=Math.min(ymin,xy[i][1]);ymax=Math.max(ymax,xy[i][1]);});
    points=[];
    for (const i of visible) {
      const x=18+(xy[i][0]-xmin)/Math.max(xmax-xmin,1e-8)*(w-36), y=18+(ymax-xy[i][1])/Math.max(ymax-ymin,1e-8)*(h-36);
      const active=(cluster===null || C.assignments[i]===cluster) && ($('event').value==='all'||D.rows[i].label===+$('event').value);
      ctx.globalAlpha=active?.75:.08;
      ctx.fillStyle=$('color').value==='label'?color(D.rows[i].label):$('color').value==='reward'?(D.rows[i].reward>1e-7?'#94dabd':D.rows[i].reward< -1e-7?'#ffc77d':'#7c91ac'):color(C.assignments[i]);
      ctx.beginPath(); ctx.arc(x,y,active?2.4:1.6,0,Math.PI*2); ctx.fill(); points.push({x,y,i,active});
    }
    ctx.globalAlpha=1;
    $('projection-note').textContent=`Reference PCA · ${fmt(projection.variance)} variance in 2D · ${num(visible.length)} displayed examples. Click a point to inspect it; use the cluster table for precise selection.`;
  }
  function renderExamples() {
    if (!C) return;
    $('inspect-title').textContent=cluster===null?'Inspect transitions':`Inspect cluster ${cluster}`;
    if (cluster!==null) {
      const counts=$('split').value==='0'?C.reference_table[cluster]:$('split').value==='all'?C.reference_table[cluster].map((v,i)=>v+C.test_table[cluster][i]):C.test_table[cluster];
      const total=sum(counts), top=counts.map((n,i)=>({n,i})).filter(x=>x.n).sort((a,b)=>b.n-a.n);
      $('composition').innerHTML=`<div class="hint">Reference majority: ${pretty(D.names[C.mapping[cluster]])} · full selected-split count ${num(total)}</div><div class="bar">${top.map(({n,i})=>`<span title="${pretty(D.names[i])}: ${n}" style="width:${100*n/Math.max(total,1)}%;background:${color(i)}"></span>`).join('')}</div><div class="hint">${top.slice(0,4).map(({n,i})=>`${pretty(D.names[i])} ${fmt(n/total)}`).join(' · ')}</div>`;
    } else $('composition').textContent='Select a cluster to see its full composition, or browse all eligible examples.';
    examples=filteredRows();
    const order=$('order').value;
    const hash=i=>((Math.imul(D.rows[i].id+1,2654435761)^C.seed)>>>0);
    examples.sort((a,b)=>order==='far'?C.distances[b]-C.distances[a]:order==='random'?hash(a)-hash(b):order==='mismatch'?
      Number(D.rows[b].label!==C.mapping[C.assignments[b]])-Number(D.rows[a].label!==C.mapping[C.assignments[a]]) || C.distances[a]-C.distances[b]:C.distances[a]-C.distances[b]);
    if (!examples.length) { $('example').innerHTML='<div class="empty">No inspectable examples match these filters. Full counts may still be nonzero because inspection uses a subset.</div>'; return; }
    page=Math.min(page, examples.length-1); showExample();
  }
  function stateCanvas(canvas,state,other,highlight) {
    const ctx=canvas.getContext('2d'); canvas.width=288;canvas.height=224;ctx.imageSmoothingEnabled=false;
    for(let j=0;j<63;j++) {
      const x=(j%9)*32,y=Math.floor(j/9)*32;
      ctx.fillStyle=state.blocks[j]===1?'#808080':'#13271b';ctx.fillRect(x,y,32,32);
      if(state.blocks[j]!==1)ctx.drawImage(textures[D.textures.blocks[state.blocks[j]]],x,y,32,32);
      for(let m=0;m<4;m++)if(state.mobs[j]&(1<<m))ctx.drawImage(textures[D.textures.mobs[m]],x,y,32,32);
      if(highlight && (state.blocks[j]!==other.blocks[j]||state.mobs[j]!==other.mobs[j])){ctx.strokeStyle='#ffd280';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,30,30);}
    }
    ctx.drawImage(textures[D.textures.players[state.sleeping?4:state.direction]],128,96,32,32);
    ctx.fillStyle=`rgba(0,0,0,${.55*(1-state.light)})`;ctx.fillRect(0,0,288,224);
  }
  function showExample() {
    if (!C || !examples.length) return;
    const i=examples[page], r=D.rows[i], c=C.assignments[i], predicted=C.mapping[c], changes=[];
    r.before.stats.forEach((v,j)=>{const a=r.after.stats[j],delta=a-v; if(Math.abs(delta)>1e-5)changes.push(`<tr><td>${pretty(D.stat_names[j])}</td><td>${v.toFixed(1)}</td><td>${a.toFixed(1)}</td><td class="${delta>0?'delta-positive':'delta-negative'}">${delta>0?'+':''}${delta.toFixed(1)}</td></tr>`);});
    const allstats=r.before.stats.map((v,j)=>`<tr><td>${pretty(D.stat_names[j])}</td><td>${v.toFixed(1)}</td><td>${r.after.stats[j].toFixed(1)}</td></tr>`).join('');
    const dataset=Math.floor(r.episode/100000), ep=r.episode%100000;
    $('example').innerHTML=`<div class="hint">Example ${num(page+1)} / ${num(examples.length)} · ${r.split?'held out':'reference'} · dataset ${dataset}, episode ${ep}, step ${r.step} · bank row ${r.id}</div>
      <div><span class="pill">Cluster ${c}</span><span class="pill">Action: ${pretty(D.action_names[r.action])}</span><span class="pill">Reward ${r.reward.toFixed(3)}</span></div>
      <div class="pair"><div class="state"><div class="caption">Before · s</div><canvas id="before"></canvas></div><div class="state"><div class="caption">After · s′</div><canvas id="after"></canvas></div></div>
      <div class="hint">Observed: <strong>${pretty(D.names[r.label])}</strong> · cluster meaning: ${pretty(D.names[predicted])} ${r.label===predicted?'✓':'(mismatch)'} · squared distance ${C.distances[i].toFixed(5)}</div>
      <table class="changes"><thead><tr><th>Changed level</th><th>Before</th><th>After</th><th>Δ</th></tr></thead><tbody>${changes.join('')||'<tr><td colspan="4">No inventory or internal-level change</td></tr>'}</tbody></table>
      <details><summary>All levels & view details</summary><table><thead><tr><th>Level</th><th>Before</th><th>After</th></tr></thead><tbody>${allstats}</tbody></table><p class="hint">Facing ${['left','right','up','down'][r.before.direction]} → ${['left','right','up','down'][r.after.direction]}; light ${r.before.light.toFixed(3)} → ${r.after.light.toFixed(3)}; sleeping ${r.before.sleeping} → ${r.after.sleeping}. Gold outlines mark changed tiles / mob masks.</p></details>`;
    imagesReady.then(()=>{if($('before')){stateCanvas($('before'),r.before,r.after,false);stateCanvas($('after'),r.after,r.before,true);}});
  }
  function renderHeatmap() {
    if (!C) return;
    const head=Array.from({length:C.k},(_,i)=>`<th><button data-cluster="${i}" title="Inspect cluster ${i}" style="padding:2px">${i}</button></th>`).join('');
    $('heatmap').innerHTML=`<table><thead><tr><th>Diagnostic event</th>${head}</tr></thead><tbody>${D.names.map((name,label)=>{
      const max=Math.max(1,...C.test_table.map(a=>a[label]));
      return `<tr><td>${pretty(name)}</td>${C.test_table.map((a,c)=>`<td style="background:rgba(148,218,189,${a[label]? .12+.7*a[label]/max:0})"><button data-cell="${c},${label}" title="${pretty(name)} · cluster ${c} · ${a[label]} transitions" style="padding:2px;border:0;background:transparent">${a[label]||'·'}</button></td>`).join('')}</tr>`;
    }).join('')}</tbody></table>`;
  }
  function renderClasses() {
    if(!C)return;
    $('class-table').innerHTML=C.classes.map(r=>`<tr><td>${pretty(D.names[r.label])}</td><td>${num(sum(C.reference_table.map(a=>a[r.label])))}</td><td>${num(r.support)}</td><td>${r.support?fmt(r.precision):'—'}</td><td>${r.support?fmt(r.recall):'—'}</td><td>${r.support?fmt(r.f1):'—'}</td></tr>`).join('');
  }
  function download(name,content,type) {
    const url=URL.createObjectURL(new Blob([content],{type})), link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  ['method','mode','k'].forEach(id=>$(id).addEventListener('change',selectConfig));
  $('seed').addEventListener('change',selectConfig);
  $('color').addEventListener('change',drawScatter);
  ['split','event','order'].forEach(id=>$(id).addEventListener('change',()=>{page=0;renderSelection();}));
  ['cluster-search','cluster-sort'].forEach(id=>$(id).addEventListener('input',renderClusters));
  $('clear').onclick=()=>{cluster=null;page=0;$('event').value='all';renderSelection();};
  $('prev').onclick=()=>{if(examples.length){page=(page-1+examples.length)%examples.length;showExample();}};
  $('next').onclick=()=>{if(examples.length){page=(page+1)%examples.length;showExample();}};
  $('random').onclick=()=>{if(examples.length){page=Math.floor(Math.random()*examples.length);showExample();}};
  $('cluster-table').onclick=e=>{const node=e.target.closest('[data-cluster]');if(node)chooseCluster(+node.dataset.cluster);};
  $('heatmap').onclick=e=>{const cell=e.target.closest('[data-cell]'),node=e.target.closest('[data-cluster]');if(cell){const [c,l]=cell.dataset.cell.split(',').map(Number);$('event').value=l;$('split').value='1';chooseCluster(c);}else if(node)chooseCluster(+node.dataset.cluster);};
  $('scatter').onclick=e=>{
    if(!C)return;const rect=$('scatter').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
    let closest=null,distance=100;for(const p of points){if(!p.active)continue;const d=(p.x-x)**2+(p.y-y)**2;if(d<distance){distance=d;closest=p;}}
    if(closest){chooseCluster(C.assignments[closest.i]);const pos=examples.indexOf(closest.i);if(pos>=0){page=pos;showExample();}}
  };
  $('export-examples').onclick=()=>{if(!C)return;download(`${C.id}_examples.csv`,'bank_row,dataset,episode,step,split,reward,action,cluster,observed_event,mapped_event,squared_distance\n'+examples.map(i=>{const r=D.rows[i],c=C.assignments[i];return [r.id,Math.floor(r.episode/100000),r.episode%100000,r.step,r.split?'test':'reference',r.reward,D.action_names[r.action],c,D.names[r.label],D.names[C.mapping[c]],C.distances[i]].join(',');}).join('\n'),'text/csv');};
  $('export-metrics').onclick=()=>{if(!C)return;const {assignments,distances,...result}=C;download(`${C.id}.json`,JSON.stringify({provenance:D.meta,experiment:result},null,2),'application/json');};
  $('fit').onclick=async()=>{
    if(fitBusy)return;
    if(!['http:','https:'].includes(location.protocol)){ $('status').textContent='New fits need the local server. Run serve.py and open http://127.0.0.1:8765. Cached experiments remain available in this file.';return; }
    const params={method:$('method').value,mode:$('mode').value,k:+$('new-k').value,seed:+$('new-seed').value,cap:+$('cap').value,iterations:+$('iterations').value};
    const usedMethod=params.method, usedMode=params.mode;
    fitBusy=true;$('fit').disabled=true;$('status').textContent='Fitting on reference transitions… the page remains available for inspection.';
    try{
      const response=await fetch('/api/fit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
      if(!response.ok)throw new Error(await response.text());
      const model=await response.json();D.configs=D.configs.filter(c=>!(c.method===model.method&&c.mode===model.mode&&c.k===model.k&&c.seed===model.seed));D.configs.push(model);$('method').value=usedMethod;$('mode').value=usedMode;
      if(![...$('k').options].some(o=>+o.value===model.k))$('k').add(new Option(model.k,model.k));
      $('k').value=model.k;availableSeeds();$('seed').value=model.seed;selectConfig();
    }catch(e){$('status').textContent=`Fit failed: ${e.message}. Use the provided Python server for new fits.`;}
    finally{fitBusy=false;$('fit').disabled=false;}
  };
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawScatter,150);});
  selectConfig();
})();
