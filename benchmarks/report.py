"""读 benchmark_report.json 渲染 markdown 报告（含 feature-level 表）。

对应 HANDOFF2.md §10 / §15 / §18：
  - 总体 + 每组 suite 的场景级指标
  - 按对象属性分组的 feature-level 召回/精确/IoU/类别/填充/颜色
  - 与 GT floor 对比看模型真实上限
  - 失败样本汇总
  - 给出"模型哪里不会"的诊断结论

用法：
  python benchmarks/report.py --report runs/benchmark_report.json
  python benchmarks/report.py --report runs/benchmark_report.json --out runs/benchmark_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 维度 → 显示名 + 缺失值桶名
FEATURE_AXES = [
    ("shape", "Shape", "_other"),
    ("fill", "Fill type", "_other"),
    ("alpha", "Semi-transparent", "_none"),
    ("stroke", "Has stroke", "_none"),
    ("hole", "Has hole", "_none"),
    ("occluded", "Occluded", "_none"),
    ("area", "Area bucket", "_none"),
]


def _bucket(recs, axis):
    out = {}
    for r in recs:
        k = r["suite_feature"].get(axis, "_none")
        out.setdefault(k, []).append(r)
    return out


def _agg(rs, key, default=0.0):
    if not rs:
        return default
    vs = [r.get(key, 0.0) for r in rs]
    return sum(vs) / len(vs)


def _fmt(x, digits=3):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def feature_table(objects, suite_filter=None):
    """按 suite_feature 各维度聚合，输出 (axis, table_rows) 列表。"""
    if suite_filter:
        objs = [o for o in objects if o["suite"] in suite_filter]
    else:
        objs = objects
    if not objs:
        return []
    out = []
    for axis, label, missing in FEATURE_AXES:
        groups = _bucket(objs, axis)
        rows = []
        for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            n = len(rs)
            matched = sum(1 for r in rs if r["matched"])
            rows.append({
                "value": k if k != missing else "(none)",
                "n_gt": n,
                "matched": matched,
                "recall": matched / n,
                "shape_ok": _agg([r for r in rs if r["matched"]], "shape_ok"),
                "fill_ok": _agg([r for r in rs if r["matched"]], "fill_ok"),
                "stroke_ok": _agg([r for r in rs if r["matched"]], "stroke_ok"),
                "iou_mean": _agg([r for r in rs if r["matched"]], "iou"),
                "color_l1": _agg([r for r in rs if r["matched"]], "color_l1"),
            })
        out.append((axis, label, rows))
    return out


def render_suite_table(report):
    s = report["suites"]
    lines = [
        "| Suite | mae | ssim | floor | recall | prec | exact | n_objs | t_med(ms) | t_max(ms) |",
        "|-------|-----|------|-------|--------|------|-------|--------|-----------|-----------|",
    ]
    for name in sorted(s.keys()):
        d = s[name]
        lines.append(
            f"| {name:<18} | {_fmt(d['mae_mean'])} | {_fmt(d['ssim_mean'])} | "
            f"{_fmt(d['floor_mean'])} | {_fmt(d['recall'], 2)} | {_fmt(d['precision'], 2)} | "
            f"{_fmt(d['obj_exact'], 2)} | {_fmt(d['n_objs_mean'], 1)} | "
            f"{_fmt(d['total_ms_median'], 0)} | {_fmt(d['total_ms_max'], 0)} |"
        )
    return "\n".join(lines)


def render_feature_table(rows):
    if not rows:
        return "_无对象级数据_"
    lines = [
        "| value | n_gt | matched | recall | shape_ok | fill_ok | stroke_ok | iou | color_l1 |",
        "|-------|------|---------|--------|----------|---------|-----------|-----|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['value']:<14} | {r['n_gt']:>4} | {r['matched']:>4} | "
            f"{_fmt(r['recall'], 2)} | {_fmt(r['shape_ok'], 2)} | "
            f"{_fmt(r['fill_ok'], 2)} | {_fmt(r['stroke_ok'], 2)} | "
            f"{_fmt(r['iou_mean'], 3)} | {_fmt(r['color_l1'], 3)} |"
        )
    return "\n".join(lines)


def diagnose(report, features):
    """基于指标给出 1-3 条最显著的"哪里不会"诊断。"""
    obj_by_match = [r for r in report["objects"] if r["matched"]]
    lines = []
    n_obj = len(report["objects"])
    n_match = len(obj_by_match)
    lines.append(f"- 总匹配率：{n_match}/{n_obj} = "
                 f"{(n_match / max(n_obj, 1)):.2f}（位置预测的召回上限）")
    if n_match > 0:
        avg_iou = sum(r["iou"] for r in obj_by_match) / n_match
        avg_cd = sum(r["center_dist"] for r in obj_by_match) / n_match
        avg_clr = sum(r["color_l1"] for r in obj_by_match) / n_match
        lines.append(f"- 匹配对平均 IoU={avg_iou:.3f}，"
                     f"中心归一化距离={avg_cd:.3f}，"
                     f"颜色 L1={avg_clr:.3f}")
    for axis, label, rows in features:
        if not rows:
            continue
        worst = min(rows, key=lambda r: r["recall"])
        if worst["recall"] < 0.5 and worst["n_gt"] >= 5:
            lines.append(f"- **{label} = `{worst['value']}`** 召回仅 "
                         f"{_fmt(worst['recall'], 2)}（n={worst['n_gt']}），"
                         f"是主要弱项")
    n_unsup = len(report["meta"].get("unsupported_features", []))
    if n_unsup:
        lines.append(f"- 未覆盖能力（生成器暂不支持，对应 HANDOFF2 Level 4）："
                     f"{', '.join(report['meta']['unsupported_features'])}")
    n_fail = len(report["failures"])
    if n_fail:
        by_suite = {}
        for f in report["failures"]:
            by_suite[f["suite"]] = by_suite.get(f["suite"], 0) + 1
        top = sorted(by_suite.items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"- 失败样本（mae>{report['meta'].get('fail_mae', 0.12)}）"
                     f"共 {n_fail} 个，集中在："
                     f"{', '.join(f'{k}({v})' for k, v in top)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Benchmark report -> markdown")
    ap.add_argument("--report", type=str, default="runs/benchmark_report.json")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    r = json.load(open(args.report, encoding="utf-8"))
    meta = r["meta"]
    md = []
    md.append(f"# Benchmark Report — `{os.path.basename(meta['ckpt'])}`")
    md.append(f"\n> step={meta.get('step')} size={meta['size']} "
              f"num/suite={meta['num_per_suite']} base_seed={meta['base_seed']} "
              f"iou_match={meta['iou_match']}\n")

    md.append("## 1. 场景级 — 每组能力集\n")
    md.append(render_suite_table(r))
    md.append("\n\n> `recall/prec` 按对象级 IoU≥{0} 匹配统计，"
              "`exact`=预测对象数==GT，"
              "`floor`=GT 场景自身渲染误差（模型理论下限），"
              "`t_med`/`t_max`=CPU 推理单张总耗时中位/最大值。\n".format(meta["iou_match"]))

    md.append("\n## 2. 总体 Feature-level Diagnosis\n")
    feats_all = feature_table(r["objects"])
    for axis, label, rows in feats_all:
        md.append(f"\n### {label}\n")
        md.append(render_feature_table(rows))

    md.append("\n\n## 3. 主要诊断\n")
    md.append(diagnose(r, feats_all))

    md.append("\n\n## 4. 失败样本\n")
    if r["failures"]:
        lines = ["| suite | seed | mae |", "|-------|------|-----|"]
        for f in r["failures"][:20]:
            lines.append(f"| {f['suite']} | {f['seed']} | {_fmt(f['mae'])} |")
        if len(r["failures"]) > 20:
            lines.append(f"\n_…另有 {len(r['failures'])-20} 个，"
                         f"图像见 `{os.path.dirname(meta.get('cases_dir', 'benchmarks/cases'))}/`_")
        md.append("\n".join(lines))
    else:
        md.append("_无失败样本（所有场景 mae 低于阈值）_")

    md.append("\n\n## 5. 局限\n")
    md.append("- Generator 当前不生成 `shadow / blur / glow`（HANDOFF2 §15 Level 4），"
              "故不覆盖；需扩展 Generator 才可评测。\n"
              "- 对象级匹配使用 bbox IoU≥{0}，GT 与 pred 的 bbox 都在归一化 [0,1] 空间，"
              "匹配逻辑与坐标无关。\n"
              "- 本报告基于静态 checkpoint；任何模型改动后应重跑对比。\n".format(meta["iou_match"]))

    out_text = "\n".join(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"[saved] {args.out}  ({len(out_text)} chars)")
    else:
        print(out_text)


if __name__ == "__main__":
    main()