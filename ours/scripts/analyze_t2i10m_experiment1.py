#!/usr/bin/env python3
"""Audit and plot a completed T2I-10M traversal experiment, without rerunning it.

Only the standard library is required. Existing reports/measurements are never
modified. Comparisons select observed operating points, not interpolated QPS.
Optional smooth SVGs interpolate a Pareto frontier for display only.
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

METHODS = ['opq-pq', 'qaq', 'extended-rabitq', 'saq', 'leanvec-ood']
NAMES = {'opq-pq': 'OPQ-PQ', 'qaq': 'QAQ', 'extended-rabitq': 'Extended RaBitQ',
         'saq': 'SAQ', 'leanvec-ood': 'LeanVec-OOD'}
COLORS = dict(zip(METHODS, ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#db2777']))
METRICS = [('qps', 'QPS'), ('mean_service_latency_ms', 'Mean service latency (ms)'),
           ('p50_ms', 'P50 service latency (ms)'), ('p95_ms', 'P95 service latency (ms)'),
           ('p99_ms', 'P99 service latency (ms)')]


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def truth(value):
    return str(value).lower() == 'true'


def audit(root, raw, rows):
    errors = []
    modes = collections.Counter()
    graphs = collections.defaultdict(set)
    grouped = collections.defaultdict(list)
    def operating_key(row):
        return tuple(row[field] for field in
                     ('experiment', 'primary', 'variant', 'side_bits', 'beam', 'rerank_depth'))
    for row in raw:
        ours = row['variant'] == 'remora'
        summary_path = Path(row['directory']) / (row['point_id'] + '.summary.json')
        native = json.loads(summary_path.read_text())
        modes[(row['variant'], native.get('ji_traversal_enabled'),
               native.get('terminal_ji', {}).get('enabled'))] += 1
        graphs[row['primary']].add(row['graph_id'])
        grouped[operating_key(row)].append(row)
        checks = {
            'status': row['status'] == 'OK',
            'variant': row['variant'] in ('primary', 'remora'),
            'traversal': native.get('ji_traversal_enabled') is ours,
            'terminal_disabled': native.get('terminal_ji', {}).get('enabled') is False,
            'side_bits': int(row['side_bits']) == (64 if ours else 0),
            'raw_query': not ours or native.get('ji_query_space') == 'raw',
            'threads': native['threads'] == 8,
            'queries': native['queries'] == 10000,
            'rerank_cap': 0 <= int(row['rerank_depth']) <= min(200, int(row['beam'])),
            'direct_rerank': not int(row['rerank_depth']) or native['rerank_storage'] == 'direct',
            'raw_vectors_not_loaded': native['base_vectors_loaded'] is False,
            'memory_cap': truth(row['budget_pass']),
        }
        errors.extend({'point_id': row['point_id'], 'check': key}
                      for key, ok in checks.items() if not ok)
    for method, identifiers in graphs.items():
        if len(identifiers) != 1:
            errors.append({'primary': method, 'check': 'same_graph'})
    for row in rows:
        repeats = grouped[operating_key(row)]
        if len(repeats) != int(row['repeats']) or len(repeats) < 3:
            errors.append({'point_id': row['point_id'], 'check': 'repeats'})
        if max(float(r['recall_at_10']) for r in repeats) - min(float(r['recall_at_10']) for r in repeats) > 1e-8:
            errors.append({'point_id': row['point_id'], 'check': 'deterministic_recall'})
        if abs(statistics.median(float(r['qps']) for r in repeats) - float(row['qps'])) > 1e-6:
            errors.append({'point_id': row['point_id'], 'check': 'median_qps'})
    if errors:
        raise ValueError('Audit failed: ' + json.dumps(errors[:20]))
    return {'raw_measurements': len(raw), 'parameter_points': len(rows), 'errors': [],
            'native_modes': [{'variant': k[0], 'traversal': k[1], 'terminal': k[2], 'runs': v}
                             for k, v in modes.items()],
            'same_graph_per_primary': True, 'repeat_counts': dict(collections.Counter(r['repeats'] for r in rows)),
            'qps_cv_max': max(float(r['qps_cv']) for r in rows),
            'qps_cv_over_5_percent': sum(float(r['qps_cv']) > .05 for r in rows)}


def select(rows, method, variant, mode, target):
    feasible = [r for r in rows if r['primary'] == method and r['variant'] == variant
                and bool(int(r['rerank_depth'])) == (mode == 'rerank')
                and float(r['recall_at_10']) >= target]
    return max(feasible, key=lambda r: (float(r['qps']), -int(r['beam']), -int(r['rerank_depth']))) if feasible else None


def equal_recall(rows):
    result = []
    for mode, targets in [('no-rerank', [.6, .65, .7, .75, .8, .85, .9, .95, .99]),
                          ('rerank', [.8, .85, .9, .92, .95, .97, .99])]:
        for method in METHODS:
            for target in targets:
                record = {'primary': method, 'mode': mode, 'target_recall': target,
                          'selection': 'fastest measured median-QPS point with Recall >= target; no interpolation'}
                selected = {}
                for variant in ('primary', 'remora'):
                    r = select(rows, method, variant, mode, target)
                    selected[variant] = r
                    record[variant + '_achieved'] = r is not None
                    if r:
                        for field in ['beam', 'rerank_depth', 'recall_at_10', 'qps', 'mean_service_latency_ms',
                                      'p50_ms', 'p95_ms', 'p99_ms', 'visited', 'rerank_io_ms', 'rerank_physical_bytes', 'qps_cv']:
                            record[variant + '_' + field] = r[field]
                p, o = selected['primary'], selected['remora']
                if p and o:
                    record['qps_gain_percent'] = (float(o['qps']) / float(p['qps']) - 1) * 100
                    record['mean_latency_reduction_percent'] = (1 - float(o['mean_service_latency_ms']) / float(p['mean_service_latency_ms'])) * 100
                    record['visited_reduction_percent'] = (1 - float(o['visited']) / float(p['visited'])) * 100
                    if int(p['rerank_depth']):
                        record['rerank_depth_reduction_percent'] = (1 - int(o['rerank_depth']) / int(p['rerank_depth'])) * 100
                result.append(record)
    return result


def same_beam(rows):
    lookup = {(r['primary'], r['variant'], int(r['beam']), int(r['rerank_depth'])): r for r in rows}
    result = []
    for method in METHODS:
        for beam in sorted({int(r['beam']) for r in rows if int(r['rerank_depth']) == 0}):
            p = lookup[(method, 'primary', beam, 0)]
            o = lookup[(method, 'remora', beam, 0)]
            record = {'primary': method, 'beam': beam,
                      'primary_recall': p['recall_at_10'], 'remora_recall': o['recall_at_10'],
                      'recall_gain_pp': 100 * (float(o['recall_at_10']) - float(p['recall_at_10'])),
                      'primary_qps': p['qps'], 'remora_qps': o['qps'],
                      'qps_change_percent': 100 * (float(o['qps']) / float(p['qps']) - 1),
                      'primary_visited': p['visited'], 'remora_visited': o['visited'],
                      'visited_reduction_percent': 100 * (1 - float(o['visited']) / float(p['visited']))}
            ep = lookup.get((method, 'primary', beam, beam))
            eo = lookup.get((method, 'remora', beam, beam))
            if ep and eo:
                coverage_gain = float(eo['recall_at_10']) - float(ep['recall_at_10'])
                rank_gain = float(ep['recall_at_10']) - float(p['recall_at_10']) - float(eo['recall_at_10']) + float(o['recall_at_10'])
                record.update(primary_full_pool_exact_recall=ep['recall_at_10'],
                              remora_full_pool_exact_recall=eo['recall_at_10'],
                              retained_candidate_gain_pp=100 * coverage_gain,
                              ranking_gain_pp=100 * rank_gain,
                              ranking_share_of_gain=10000 * rank_gain / record['recall_gain_pp'])
            result.append(record)
    return result


def pareto(rows, metric):
    minimize = metric != 'qps'
    by_x = {}
    for row in rows:
        x = float(row['recall_at_10']) * 100
        y = float(row[metric])
        old = by_x.get(x)
        if old is None or (y < float(old[metric]) if minimize else y > float(old[metric])):
            by_x[x] = row
    retained = []
    best = math.inf if minimize else -math.inf
    for x, row in sorted(by_x.items(), reverse=True):
        y = float(row[metric])
        if y < best if minimize else y > best:
            retained.append(row)
            best = y
    return list(reversed(retained))


def monotone_slopes(points):
    """PCHIP slopes; a display interpolator, never an operating-point estimate."""
    n = len(points)
    if n < 2:
        return [0.] * n
    h = [points[i + 1][0] - points[i][0] for i in range(n - 1)]
    d = [(points[i + 1][1] - points[i][1]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    slopes = [0.] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    def endpoint(h0, h1, d0, d1):
        value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if value * d0 <= 0:
            return 0.
        if d0 * d1 < 0 and abs(value) > 3 * abs(d0):
            return 3 * d0
        return value
    slopes[0] = endpoint(h[0], h[1], d[0], d[1])
    slopes[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    return slopes


def draw(path, series, metric, mode, smooth=False):
    xmin, xmax = (40., 85.) if mode == 'no-rerank' else (80., 100.)
    left, right, top, bottom = 88., 660., 72., 462.
    visible = [p for _, _, _, pp in series for p in pp if xmin <= p[0] <= xmax]
    maximum = max((y for _, y in visible), default=1.) * 1.06
    step = 10 ** math.floor(math.log10(maximum / 5))
    ymax = math.ceil(maximum / step) * step
    xx = lambda x: left + (right - left) * (x - xmin) / (xmax - xmin)
    yy = lambda y: bottom - (bottom - top) * y / ymax
    title = 'Without FP32 rerank' if mode == 'no-rerank' else 'With finite NVMe FP32 rerank'
    label = dict(METRICS)[metric]
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="940" height="560" viewBox="0 0 940 560">',
            '<rect width="940" height="560" fill="white"/>',
            '<g font-family="Arial,Helvetica,sans-serif" fill="#222">',
            '<text x="374" y="27" text-anchor="middle" font-size="18" font-weight="600">T2I-10M — ' + title + '</text>',
            '<text x="374" y="49" text-anchor="middle" font-size="12">8 CPU threads · 64-bit REMORA traversal · shared host</text>']
    xticks = list(range(40, 86, 5)) if mode == 'no-rerank' else [80, 85, 90, 95, 100]
    for x in xticks:
        body.extend([f'<path d="M{xx(x):.3f},{top}V{bottom}" stroke="#e5e7eb" fill="none"/>',
                     f'<text x="{xx(x):.3f}" y="487" text-anchor="middle" font-size="12">{x}</text>'])
    for i in range(6):
        y = ymax * i / 5
        body.extend([f'<path d="M{left},{yy(y):.3f}H{right}" stroke="#e5e7eb" fill="none"/>',
                     f'<text x="78" y="{yy(y)+4:.3f}" text-anchor="end" font-size="12">{y:,.0f}</text>' if metric == 'qps' else
                     f'<text x="78" y="{yy(y)+4:.3f}" text-anchor="end" font-size="12">{y:.2f}</text>'])
    body.extend([f'<path d="M{left},{top}V{bottom}H{right}" stroke="#111" fill="none"/>',
                 '<text x="374" y="519" text-anchor="middle" font-size="15">Recall@10 (%)</text>',
                 f'<text transform="translate(22,267) rotate(-90)" text-anchor="middle" font-size="15">{escape(label)}</text>'])
    for index, (name, color, dashed, pp) in enumerate(series):
        pp = [p for p in pp if xmin <= p[0] <= xmax]
        if not pp:
            continue
        dash = ' stroke-dasharray="8 5"' if dashed else ''
        coords = [(xx(x), yy(y)) for x, y in pp]
        if smooth and len(pp) > 1:
            slopes = monotone_slopes(pp)
            command = f'M{coords[0][0]:.3f},{coords[0][1]:.3f}'
            for i in range(len(pp) - 1):
                x0, y0 = pp[i]
                x1, y1 = pp[i + 1]
                h = x1 - x0
                command += (f' C{xx(x0+h/3):.3f},{yy(y0+h*slopes[i]/3):.3f}'
                            f' {xx(x1-h/3):.3f},{yy(y1-h*slopes[i+1]/3):.3f}'
                            f' {xx(x1):.3f},{yy(y1):.3f}')
            curve = f'<path d="{command}"'
        else:
            coord_text = ' '.join(f'{x:.3f},{y:.3f}' for x, y in coords)
            curve = f'<polyline points="{coord_text}"'
        body.append(curve + f' fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"{dash}/>')
        legend_y = 102 + 27 * index
        body.extend([f'<path d="M681,{legend_y}h30" stroke="{color}" stroke-width="2.2"{dash}/>',
                     f'<text x="719" y="{legend_y+4}" font-size="12">{escape(name)}</text>'])
    note = ('Shape-preserving interpolation for display only; comparisons use measured points.' if smooth else
            'Measured Pareto frontier; 3-run medians. No extrapolation or synthetic measurements.')
    body.append('<text x="88" y="546" font-size="11" fill="#555">' + note + '</text>')
    body.append('</g></svg>')
    path.write_text('\n'.join(body) + '\n')
    ET.parse(path)


def plots(dest, rows):
    frontier_records = []
    for mode in ['no-rerank', 'rerank']:
        for metric, _ in METRICS:
            series = []
            for method in METHODS:
                for variant in ['primary', 'remora']:
                    local = [r for r in rows if r['primary'] == method and r['variant'] == variant
                             and bool(int(r['rerank_depth'])) == (mode == 'rerank')]
                    selected = pareto(local, metric)
                    for row in selected:
                        frontier_records.append(dict(metric=metric, mode=mode, **row))
                    points = [(100 * float(r['recall_at_10']), float(r[metric])) for r in selected]
                    series.append((NAMES[method] + (' Primary' if variant == 'primary' else ' + REMORA'),
                                   COLORS[method], variant == 'primary', points))
            filename = ('latency' if metric == 'mean_service_latency_ms' else metric) + '-recall-' + mode
            draw(dest / (filename + '.svg'), series, metric, mode)
            smooth_dir = dest / 'smooth-display'
            smooth_dir.mkdir(exist_ok=True)
            draw(smooth_dir / (filename + '.svg'), series, metric, mode, smooth=True)
            if metric in ('qps', 'mean_service_latency_ms'):
                panel_dir = dest / 'per-primary'
                panel_dir.mkdir(exist_ok=True)
                for index, method in enumerate(METHODS):
                    draw(panel_dir / (method + '-' + filename + '.svg'), series[index*2:index*2+2], metric, mode)
    write_csv(dest / 'frontier-measured-points.csv', frontier_records)


def memory(rows):
    output = []
    for method in METHODS:
        for variant in ['primary', 'remora']:
            local = [r for r in rows if r['primary'] == method and r['variant'] == variant]
            r = min(local, key=lambda r: (int(r['beam']), int(r['rerank_depth'])))
            record = {'primary': method, 'variant': variant, 'accounting_reference_beam': r['beam'],
                      'accounting_reference_rerank_depth': r['rerank_depth']}
            for field in ['graph_memory_bytes', 'primary_code_bytes', 'refiner_code_bytes',
                          'codebooks_transform_bytes', 'runtime_buffer_bytes', 'accounted_memory_bytes']:
                record[field + '_GB'] = float(r[field]) / 1e9
            record['min_peak_rss_GB'] = min(float(x['peak_rss_bytes']) for x in local) / 1e9
            record['max_peak_rss_GB'] = max(float(x['peak_rss_bytes']) for x in local) / 1e9
            record['fp32_hot_GB'] = 0
            record['fp32_cold_GB'] = 8.192004096
            output.append(record)
    return output


def report(dest, root, rows, matched, beam_rows, memories, checked):
    def f(value, digits=2):
        return f'{float(value):,.{digits}f}'
    lines = ['# T2I-10M 实验一结果分析', '',
             '本报告只使用本轮 REMORA traversal 数据，不混入旧 terminal-only 结果。', '',
             '## 1. 核验与测试口径', '',
             f'- {checked["parameter_points"]} 个参数点、{checked["raw_measurements"]} 次运行；每点 3 次，零失败。',
             '- 全部 REMORA 原始记录：traversal 开启、terminal-only 关闭、raw query space。',
             '- 每种 primary 的两个分支复用同一张图、同一套 primary；REMORA 增加 64 bits/node。',
             '- 10,000 个测试查询；8 CPU 线程，绑定 CPU 16–23；共享宿主机，不是独占机器。',
             '- beam 范围 10–2000；有限精排 L ≤ min(beam, 200)，取各自搜索队列前 L 个。',
             '- 精排 FP32 通过 NVMe O_DIRECT 读取，绕过 OS page cache；未禁用 SSD 设备缓存。',
             '- QPS 为全批吞吐；P50/P95/P99 为每查询 service latency，不包括服务端排队，不等于 1/QPS。',
             f'- 最大重复运行 QPS CV：{100*checked["qps_cv_max"]:.2f}%；不替代逐查询统计置信区间。',
             '- 本轮未运行 FP32 graph-search 性能分支，图中不添加来自其他设置的 FP32 曲线。', '',
             '## 2. 无 FP32 精排：固定 beam=100', '',
             '| Primary | Primary Recall | REMORA Recall | 增益 (pp) | Primary QPS | REMORA QPS | QPS 变化 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in beam_rows:
        if r['beam'] == 100:
            lines.append(f'| {NAMES[r["primary"]]} | {f(100*float(r["primary_recall"]),3)}% | {f(100*float(r["remora_recall"]),3)}% | +{f(r["recall_gain_pp"],3)} | {f(r["primary_qps"],0)} | {f(r["remora_qps"],0)} | {f(r["qps_change_percent"])}% |')
    lines += ['', '同 beam 提高 Recall，但 scoring 成本使吞吐下降。应另外报告 matched-Recall，不能把同 beam 的质量优势写成无条件加速。', '',
              '### 本次扫描达到的最高无精排 Recall', '',
              '| Primary | Primary | REMORA | 增益 (pp) |', '|---|---:|---:|---:|']
    for method in METHODS:
        mx = {}
        for variant in ['primary', 'remora']:
            mx[variant] = max(float(r['recall_at_10']) for r in rows if r['primary']==method and r['variant']==variant and int(r['rerank_depth'])==0)
        lines.append(f'| {NAMES[method]} | {100*mx["primary"]:.3f}% | {100*mx["remora"]:.3f}% | +{100*(mx["remora"]-mx["primary"]):.3f} |')
    lines += ['', '以上最大值均在 beam=2000。五种 REMORA 无精排都未达到 85%/90%；不能宣称已替代高 Recall 的 FP32 精排。这是测试范围内结果，不是理论编码上限。', '',
              '## 3. 有限 NVMe 精排：达到指定 Recall 的最快已测点', '',
              '以下属于测试扫描的离散 frontier 比较，实际 Recall 可能超过阈值且不完全相等；不作插值估计，不等价于验证集预先选择的部署点。']
    for target in [.9,.95,.99]:
        lines += ['', f'### Recall ≥ {100*target:.0f}%', '',
                  '| Primary | Primary beam/L | REMORA beam/L | 实际 Recall P / O | QPS P / O | QPS 提升 | Mean latency P / O (ms) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in matched:
            if r['mode'] != 'rerank' or r['target_recall'] != target:
                continue
            def pair(prefix, fields):
                return '/'.join(str(r[prefix+'_'+field]) for field in fields) if r[prefix+'_achieved'] else '未达到'
            p, o = r['primary_achieved'], r['remora_achieved']
            recall = ' / '.join(f'{100*float(r[v+"_recall_at_10"]):.3f}%' if r[v+'_achieved'] else '—' for v in ['primary','remora'])
            qps = ' / '.join(f(r[v+'_qps'],0) if r[v+'_achieved'] else '—' for v in ['primary','remora'])
            lat = ' / '.join(f(r[v+'_mean_service_latency_ms'],3) if r[v+'_achieved'] else '—' for v in ['primary','remora'])
            gain = '+'+f(r['qps_gain_percent'])+'%' if p and o else '不可计算'
            lines.append(f'| {NAMES[r["primary"]]} | {pair("primary",["beam","rerank_depth"])} | {pair("remora",["beam","rerank_depth"])} | {recall} | {qps} | {gain} | {lat} |')
    lines += ['', '99%：Extended Primary 最高 98.944%，SAQ Primary 最高 97.605%；在 L≤200、beam≤2000 下未达到 99%，不能赋予速度比或外推曲线。', '',
              '90%/95% 下五种 primary 均有更高吞吐；收益结合更好的队列排序、可减少的精排 L 和部分 traversal 工作减少，不能全部归因于 visited。', '',
              '## 4. 固定 beam 下的收益分解', '',
              '使用同一分支候选池完整 FP32 精排（L=beam）作为 retained candidate coverage 的观测。以下为 beam=100：', '',
              '| Primary | 总无精排增益 (pp) | Full-pool exact 增益 (pp) | 排序损失减少 (pp) | 排序占比 |',
              '|---|---:|---:|---:|---:|']
    for r in beam_rows:
        if r['beam']==100:
            lines.append(f'| {NAMES[r["primary"]]} | {r["recall_gain_pp"]:.3f} | {r["retained_candidate_gain_pp"]:.3f} | {r["ranking_gain_pp"]:.3f} | {r["ranking_share_of_gain"]:.1f}% |')
    lines += ['', 'OPQ/QAQ 的 candidate coverage 改善很小，主要改善保留候选的排序；SAQ 的 coverage 增益更大。Full-pool exact 的差值是候选覆盖结果，不能单独证明 trajectory 所有局部决策更正确。', '',
              '## 5. 内存与存储', '',
              '| Primary | Primary 热内存核算 (GB) | REMORA 热内存核算 (GB) | Primary 最大 RSS (GB) | REMORA 最大 RSS (GB) |',
              '|---|---:|---:|---:|---:|']
    for method in METHODS:
        mm={r['variant']:r for r in memories if r['primary']==method}
        lines.append(f'| {NAMES[method]} | {mm["primary"]["accounted_memory_bytes_GB"]:.3f} | {mm["remora"]["accounted_memory_bytes_GB"]:.3f} | {mm["primary"]["max_peak_rss_GB"]:.3f} | {mm["remora"]["max_peak_rss_GB"]:.3f} |')
    lines += ['', '- 核算热内存表使用相同最小 beam 配置；完整 scan 的 RSS 最大值另列，不混用口径。',
              '- 10M×64 bits 额外载荷 80 MB，共享 REMORA 模型约 0.367 MB，另有线程工作缓冲。',
              '- 图在 Primary/REMORA 间不变；各方法的 primary 物理大小不同，不能称跨方法 equal-memory。',
              '- 两者均仍保留 8.192 GB 冷 FP32 文件，有限精排没有消除它的存储成本；热内存里没有加载全库 FP32。',
              '- 256 GB 是单进程地址空间限额加 RSS 检查，非独占 cgroup 物理内存预算；本轮实际 RSS 低于 4 GB。', '',
              '## 6. 绘图口径与尚未完成的证据', '',
              '- Primary 虚线、REMORA 实线；同 primary 同色，FP32 曲线不混入。',
              '- 主 SVG 用实测 Pareto 点直线连接。smooth-display/ 是形状保持插值，仅美化展示，不用于表格或速度比。',
              '- 每个 latency 分位数单独取其 Pareto frontier，因此其 operating points 不必与 QPS 曲线一致。',
              '- 无精排 x 轴 40%–85%，有限精排 80%–100%；曲线只延伸到实测端点。',
              '- 不绘制暗色背景散点；全部实测点与各 frontier 选择保存在 CSV，便于复核。',
              '- 867 个点是已扫参数，不是连续空间全局最优；不同分支自适应补点数不同。',
              '- inversion/teacher-forced routing 尚未由本次实验一测得；不能将 Recall 改善冒充实测 inversion 下降。',
              '- QAQ/LeanVec 为本地论文复现；与原作者运行时性能应区别。',
              '- nominal primary：OPQ/QAQ 800 bits；Extended/SAQ/LeanVec 768 bits，物理 bytes 也不完全相等。', '',
              '## 7. 结论', '',
              'REMORA-64 在所有五种 primary 上提升无 FP 精排 Recall，也改善有限 NVMe 精排的 Recall–QPS frontier；在 90%/95% 工作点，节省精排读取和搜索工作足以覆盖 correction 开销。当前仍未实现 90% 的压缩码直接输出，主要剩余缺口是候选内细排序精度。', '',
              f'数据根目录：`{root}`。所有分析文件位于本报告所在目录，原始实验文件保持不变。', '']
    (dest/'analysis.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root/'experiment1/summary.json').read_text())
    if summary['status'] != 'COMPLETE' or summary['scoring_placement'] != 'traversal':
        raise ValueError('A completed REMORA traversal experiment is required')
    raw = [r for r in read_csv(root/'results.csv') if int(r['experiment'])==1]
    rows = [r for r in read_csv(root/'aggregate.csv') if int(r['experiment'])==1 and r['status']=='OK' and truth(r['timing_complete'])]
    checked = audit(root, raw, rows)
    matched = equal_recall(rows)
    beam_rows = same_beam(rows)
    memories = memory(rows)
    dest = root/'experiment1/analysis'
    dest.mkdir(exist_ok=True)
    write_csv(dest/'equal-recall-comparison.csv', matched)
    write_csv(dest/'same-beam-comparison.csv', beam_rows)
    write_csv(dest/'memory-summary.csv', memories)
    (dest/'audit.json').write_text(json.dumps(checked, indent=2)+'\n')
    plots(dest, rows)
    report(dest, root, rows, matched, beam_rows, memories, checked)
    print(f'[audit OK] {len(raw)} measurements / {len(rows)} operating points')
    print(f'[done] {dest}')


if __name__ == '__main__':
    main()
