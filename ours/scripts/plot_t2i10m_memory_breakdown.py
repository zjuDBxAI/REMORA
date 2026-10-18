#!/usr/bin/env python3
"""Plot the five-component hot-memory ledger at a common operating point.

Does not rerun benchmarks or modify their results. The ledger is not an exact
RSS attribution; unclassified process overhead is retained in the output CSV.
"""
import argparse
import collections
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


METHODS = [('opq-pq', 'OPQ-PQ'), ('qaq', 'QAQ'),
           ('extended-rabitq', 'Extended RaBitQ'), ('saq', 'SAQ'),
           ('leanvec-ood', 'LeanVec-OOD')]
PARTS = [('graph_memory_bytes', 'Graph', '#4C78A8'),
         ('primary_code_bytes', 'Primary codes', '#F58518'),
         ('refiner_code_bytes', 'Refiner codes', '#54A24B'),
         ('codebooks_transform_bytes', 'Codebooks / transform', '#B279A2'),
         ('runtime_buffer_bytes', 'Runtime buffer', '#9D9DA1')]


def records(root, beam, depth):
    with (root / 'results.csv').open(newline='') as handle:
        raw = [r for r in csv.DictReader(handle) if r['experiment'] == '1'
               and int(r['beam']) == beam and int(r['rerank_depth']) == depth
               and r['status'] == 'OK']
    groups = collections.defaultdict(list)
    for row in raw:
        groups[(row['primary'], row['variant'])].append(row)
    output = []
    for method, name in METHODS:
        for variant in ['primary', 'remora']:
            runs = groups[(method, variant)]
            if len(runs) < 3 or len({r['repeat'] for r in runs}) != len(runs):
                raise ValueError(f'Missing/duplicate timing repeats: {method}/{variant}')
            if any(int(r['threads']) != 8 or float(r['fp_hot_bytes']) != 0 for r in runs):
                raise ValueError('Expected 8 threads and no hot FP32 database vectors')
            row = {'primary': method, 'variant': variant,
                   'method': name + (' Primary' if variant == 'primary' else ' + REMORA-64'),
                   'beam': beam, 'rerank_depth': depth, 'threads': 8, 'repeats': len(runs)}
            values = {field: statistics.median(float(r[field]) for r in runs)
                      for field, _, _ in PARTS}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError('Invalid memory component')
            total = sum(values.values())
            accounted = statistics.median(float(r['accounted_memory_bytes']) for r in runs)
            if abs(total - accounted) > 1:
                raise ValueError('Five components do not sum to the native memory ledger')
            if (values['refiner_code_bytes'] > 0) != (variant == 'remora'):
                raise ValueError('Refiner-code accounting disagrees with method')
            for field, _, _ in PARTS:
                row[field] = int(values[field])
                row[field + '_GB'] = values[field] / 1e9
                row[field + '_percent'] = 100 * values[field] / total
            rss = statistics.median(float(r['peak_rss_bytes']) for r in runs)
            row.update(total_accounted_bytes=int(total), total_accounted_GB=total / 1e9,
                       median_peak_rss_bytes=int(rss), median_peak_rss_GB=rss / 1e9,
                       max_peak_rss_bytes=max(int(r['peak_rss_bytes']) for r in runs),
                       rss_minus_accounted_bytes=int(rss-total),
                       accounting='median native capacity ledger; not exact RSS attribution',
                       cold_fp32_excluded=True)
            output.append(row)
    return output


def draw(path, rows, beam, depth, normalized=False):
    width, height = 1320, 670
    left, right, top, bottom = 88, 1270, 105, 475
    group_width = (right - left) / len(METHODS)
    ymax = 100 if normalized else math.ceil(max(r['total_accounted_GB'] for r in rows)*1.08*2)/2
    scale = (bottom-top)/ymax
    title = 'T2I-10M — Hot-memory structure shares' if normalized else 'T2I-10M — Hot-memory breakdown'
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="670" viewBox="0 0 1320 670">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Arial,Helvetica,sans-serif" fill="#222">',
            f'<text x="679" y="27" text-anchor="middle" font-size="21" font-weight="600">{title}</text>',
            f'<text x="679" y="50" text-anchor="middle" font-size="13">8 CPU threads · beam={beam} · rerank L={depth} · 3-run component medians</text>']
    legend_x = [135, 335, 565, 780, 1080]
    for x, (_, label, color) in zip(legend_x, PARTS):
        body.extend([f'<rect x="{x}" y="68" width="17" height="14" fill="{color}"/>',
                     f'<text x="{x+25}" y="80" font-size="13">{escape(label)}</text>'])
    for i in range(6 if normalized else 9):
        value = ymax * i / (5 if normalized else 8)
        y = bottom - value * scale
        body.extend([f'<path d="M{left},{y:.3f}H{right}" stroke="#e5e7eb" fill="none"/>',
                     f'<text x="{left-12}" y="{y+4:.3f}" text-anchor="end" font-size="12">{value:.0f}%</text>' if normalized else
                     f'<text x="{left-12}" y="{y+4:.3f}" text-anchor="end" font-size="12">{value:.1f}</text>'])
    for index, row in enumerate(rows):
        group = index // 2
        center = left + (group + .5) * group_width
        x = center - 77 if index % 2 == 0 else center + 13
        bar_width = 64
        y = bottom
        for field, _, color in PARTS:
            value = row[field+'_percent'] if normalized else row[field+'_GB']
            h = value * scale
            y -= h
            body.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{bar_width}" height="{h:.5f}" fill="{color}"/>')
            # Labels must fit actual segments. Never enlarge tiny components.
            if h >= 30:
                fill = 'white' if field == 'graph_memory_bytes' else '#222'
                percent = row[field+'_percent']
                body.append(f'<text x="{x+bar_width/2:.3f}" y="{y+h/2+4:.3f}" text-anchor="middle" font-size="12" fill="{fill}">{percent:.1f}%</text>')
        if not normalized:
            body.append(f'<text x="{x+bar_width/2:.3f}" y="{y-8:.3f}" text-anchor="middle" font-size="14" font-weight="600">{row["total_accounted_GB"]:.3f}</text>')
        variant = 'Primary' if row['variant']=='primary' else '+ REMORA'
        body.append(f'<text x="{x+bar_width/2:.3f}" y="500" text-anchor="middle" font-size="13">{variant}</text>')
        if index % 2 == 1:
            label = METHODS[group][1]
            body.extend([f'<path d="M{center-89:.3f},514H{center+89:.3f}" stroke="#777" fill="none"/>',
                         f'<text x="{center:.3f}" y="541" text-anchor="middle" font-size="15">{escape(label)}</text>'])
    ylabel = 'Share of accounted memory (%)' if normalized else 'Memory (GB)'
    body.extend([f'<path d="M{left},{top}V{bottom}H{right}" stroke="#111" fill="none"/>',
                 f'<text transform="translate(25,290) rotate(-90)" text-anchor="middle" font-size="16">{ylabel}</text>',
                 '<text x="88" y="588" font-size="13">Accounted hot-memory components, not a decomposition of process RSS. Decimal GB (1 GB = 10^9 bytes).</text>',
                 '<text x="88" y="612" font-size="13">Cold FP32 vectors excluded. Runtime includes query/result buffers, worker workspaces and refiner caches.</text>',
                 '<text x="88" y="636" font-size="13">Codebooks / transform occupies &lt;0.1%; thin segments are drawn at their true scale. Exact shares are in the CSV.</text>',
                 '</g></svg>'])
    path.write_text('\n'.join(body)+'\n')
    ET.parse(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--beam', type=int, default=100)
    parser.add_argument('--rerank-depth', type=int, default=50)
    args = parser.parse_args()
    rows = records(args.root, args.beam, args.rerank_depth)
    dest = args.root/'experiment1/analysis'
    dest.mkdir(exist_ok=True)
    name = f'memory-breakdown-b{args.beam}-l{args.rerank_depth}'
    with (dest/(name+'.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(dest/(name+'.svg'), rows, args.beam, args.rerank_depth)
    draw(dest/(name+'-share.svg'), rows, args.beam, args.rerank_depth, normalized=True)
    metadata = {'source': str(args.root/'results.csv'), 'rows': len(rows),
                'beam': args.beam, 'rerank_depth': args.rerank_depth, 'threads': 8,
                'repeats': 3, 'components': [label for _,label,_ in PARTS],
                'note': 'Native component accounting, not exact RSS attribution; cold FP32 excluded; no unmeasured FP32 bar.'}
    (dest/(name+'.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    print('[done]', dest/(name+'.svg'))
    print('[done]', dest/(name+'-share.svg'))
    print('[done]', dest/(name+'.csv'))


if __name__ == '__main__':
    main()
