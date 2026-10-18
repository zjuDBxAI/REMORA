"""Dependency-free paper plots; only measured nondominated frontier points.

Lines are straight segments, never invented performance measurements. Full raw
and frontier CSVs remain beside the SVGs. Primary lines dashed; REMORA solid.
"""
from collections import defaultdict
from html import escape
from pathlib import Path
import math

PALETTE=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#333333']


def frontier(rows,metric):
    minimize=metric!='qps'; points=[]
    for r in rows:
        x,y=float(r['recall_at_10']),float(r[metric])
        if math.isfinite(x) and math.isfinite(y) and y>0: points.append((x,y))
    by_x={}
    for x,y in points:
        by_x[x]=(min(by_x.get(x,math.inf),y) if minimize else max(by_x.get(x,-math.inf),y))
    out=[]; best=math.inf if minimize else -math.inf
    for x,y in sorted(by_x.items(),reverse=True):
        if (y<best) if minimize else (y>best): out.append((x,y)); best=y
    return list(reversed(out))


def chart(path,series,title,ylabel,minimum):
    width,height=900,520; left,right,top,bottom=90,625,50,445
    points=[p for _,_,_,pp in series for p in pp]
    if not points:
        Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520"><text x="30" y="50">{escape(title)}: no measured points at Recall ≥ {minimum:.0%}</text></svg>\n'); return
    xmin=minimum; xmax=max(1.,max(x for x,_ in points)); ymax=max(y for _,y in points)*1.08
    xx=lambda x:left+(right-left)*(x-xmin)/(xmax-xmin)
    yy=lambda y:bottom-(bottom-top)*y/ymax
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
          '<rect width="100%" height="100%" fill="white"/>',
          '<g font-family="Arial, sans-serif" font-size="15" fill="#222">',
          f'<text x="90" y="28" font-size="19">{escape(title)}</text>']
    for i in range(6):
        x=xmin+(xmax-xmin)*i/5; px=xx(x); y=ymax*i/5; py=yy(y)
        body += [f'<path d="M{left},{py:.2f}H{right}" stroke="#ddd" fill="none"/>',
                 f'<text x="{left-12}" y="{py+5:.2f}" text-anchor="end">{y:.3g}</text>',
                 f'<text x="{px:.2f}" y="{bottom+25}" text-anchor="middle">{x:.2f}</text>']
    body += [f'<path d="M{left},{top}V{bottom}H{right}" fill="none" stroke="#222"/>',
             f'<text x="{(left+right)/2}" y="{bottom+60}" text-anchor="middle">Recall@10</text>',
             f'<text transform="translate(22,{(top+bottom)/2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']
    for i,(label,color,dashed,pp) in enumerate(series):
        dash=' stroke-dasharray="7 4"' if dashed else ''
        coords=' '.join(f'{xx(x):.2f},{yy(y):.2f}' for x,y in pp)
        body += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.3"{dash}/>',
                 f'<path d="M650,{top+i*29}h30" stroke="{color}" stroke-width="2.3"{dash}/>',
                 f'<text x="688" y="{top+i*29+5}" font-size="12">{escape(label)}</text>']
    body.append('</g></svg>'); Path(path).write_text('\n'.join(body)+'\n')


def draw_reports(out,rows,memories,minimum,dataset='T2I-10M'):
    dest=Path(out)/'experiment1/figures'; dest.mkdir(parents=True,exist_ok=True)
    methods=sorted({r['primary'] for r in rows}); colors={m:PALETTE[i%len(PALETTE)] for i,m in enumerate(methods)}
    for mode in ('no-rerank','rerank'):
        good=[r for r in rows if int(r['experiment'])==1 and r.get('timing_complete') and
              bool(int(r['rerank_depth']))==(mode=='rerank') and float(r['recall_at_10'])>=minimum]
        groups=defaultdict(list)
        for r in good: groups[(r['primary'],r['variant'])].append(r)
        for metric,label in [('qps','QPS'),('mean_service_latency_ms','Mean service latency (ms)'),('p50_ms','P50 service latency (ms)'),('p95_ms','P95 service latency (ms)'),('p99_ms','P99 service latency (ms)')]:
            series=[]
            for (m,v),rr in sorted(groups.items()):
                series.append((m+' / '+v,colors[m],v=='primary',frontier(rr,metric)))
            chart(dest/f'{metric}-recall-{mode}.svg',series,f'{dataset} — {mode}',label,minimum)
    memory_bars(dest/'memory-breakdown.svg',memories)


def memory_bars(path,rows):
    # One observed footprint per branch, no mixing graph degree between runs.
    grouped={}
    for r in sorted(rows,key=lambda x:(int(x['beam']),int(x['rerank_depth']))):
        if int(r['experiment'])==1: grouped.setdefault((r['primary'],r['variant']),r)
    if not grouped: return
    fields=['graph_memory_bytes','primary_code_bytes','refiner_code_bytes','codebooks_transform_bytes','runtime_buffer_bytes','fp_hot_bytes']
    labels=['Graph','Primary codes','Refiner codes','Codebooks / transform','Runtime buffers','FP32 hot data']
    width=max(950,120*len(grouped)+250); height=540; left,top,bottom=80,50,370
    maximum=max(sum(float(r.get(k) or 0) for k in fields) for r in grouped.values())/1e9*1.1
    if maximum<=0: return
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
          '<rect width="100%" height="100%" fill="white"/>','<g font-family="Arial,sans-serif" font-size="13">',
          '<text x="80" y="25">Measured index footprint (decimal GB); runtime payload estimate, allocator overhead tracked separately by RSS</text>']
    for i in range(6):
        y=bottom-(bottom-top)*i/5
        body += [f'<path d="M{left},{y}H{width-220}" stroke="#ddd"/>',f'<text x="{left-12}" y="{y+4}" text-anchor="end">{maximum*i/5:.2f}</text>']
    for i,((m,v),r) in enumerate(sorted(grouped.items())):
        x=left+20+i*120; y=bottom
        for k,color in zip(fields,PALETTE):
            h=float(r.get(k) or 0)/1e9/maximum*(bottom-top); y-=h
            body.append(f'<rect x="{x}" y="{y:.2f}" width="70" height="{h:.2f}" fill="{color}"/>')
        body.append(f'<text transform="translate({x+35},{bottom+15}) rotate(35)">{escape(m+" / "+v)}</text>')
    for i,(label,color) in enumerate(zip(labels,PALETTE)):
        body += [f'<rect x="{width-200}" y="{top+i*28}" width="15" height="15" fill="{color}"/>',f'<text x="{width-175}" y="{top+i*28+12}">{escape(label)}</text>']
    body.append('</g></svg>'); Path(path).write_text('\n'.join(body)+'\n')
