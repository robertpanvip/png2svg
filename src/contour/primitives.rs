//! 直边 / 真圆弧 分段（segment_primitives），以及整圆/整椭圆的 SVG 子路径生成。
//!
//! segment_primitives 把一条简化轮廓切成「直线段(L)」与「真圆弧(A)」交替的子路径：
//!   - 对角/斜线（栅格化后是台阶）被合并成单条直线，消除毛刺；
//!   - 圆角/倒角（部分圆弧）被识别成 SVG 弧命令，而不是碎成多边形或整条贝塞尔。
//! 自由曲线（花瓣/云朵，曲率处处变化）不应进入此分支，由上层改走贝塞尔。

use crate::contour::fit::fit_and_validate_arc;
use crate::contour::simplify::{point_line_dist, simplify_polygon};

pub(crate) fn circle_subpath(cx: f32, cy: f32, r: f32) -> String {
    format!(
        "M {} {} A {} {} 0 0 1 {} {} A {} {} 0 0 1 {} {} Z",
        cx - r, cy, r, r, cx + r, cy, r, r, cx - r, cy
    )
}

/// 旋转椭圆子路径：用 SVG 弧命令（带 x-axis-rotation）还原旋转椭圆。
pub(crate) fn rotated_ellipse_subpath(cx: f32, cy: f32, rx: f32, ry: f32, theta_rad: f32) -> String {
    let th = theta_rad as f64;
    let ct = th.cos();
    let st = th.sin();
    let x0 = cx as f64 + (rx as f64) * ct;
    let y0 = cy as f64 + (rx as f64) * st;
    let x1 = cx as f64 - (rx as f64) * ct;
    let y1 = cy as f64 - (rx as f64) * st;
    let deg = theta_rad.to_degrees();
    format!(
        "M {:.2} {:.2} A {} {} {:.2} 0 1 {:.2} {:.2} A {} {} {:.2} 0 1 {:.2} {:.2} Z",
        x0, y0, rx, ry, deg, x1, y1, rx, ry, deg, x0, y0
    )
}

/// 把闭环切成 直线(L) / 真圆弧(A) 子路径（基于「运行」的贪心分段：弧优先）。
pub fn segment_primitives(loop_: &[(f32, f32)], eps: f32) -> String {
    let line_eps = if eps < 1.0 { 1.0 } else { eps };
    let arc_tol_abs = (eps * 3.0).max(1.5);
    let pts = simplify_polygon(loop_, line_eps);
    let n = pts.len();
    if n < 3 {
        if pts.len() >= 2 {
            let mut s = format!("M {} {}", pts[0].0, pts[0].1);
            for q in &pts[1..] {
                s.push_str(&format!(" L {} {}", q.0, q.1));
            }
            s.push_str(" Z");
            return s;
        }
        return String::new();
    }

    // 把每个简化顶点映射回原始轮廓索引
    let loop_len = loop_.len();
    let mut idx = vec![0usize; n];
    let mut cursor = 0usize;
    for k in 0..n {
        let mut found = None;
        for i in cursor..loop_len {
            if loop_[i] == pts[k] {
                found = Some(i);
                break;
            }
        }
        if found.is_none() {
            for i in 0..cursor {
                if loop_[i] == pts[k] {
                    found = Some(i);
                    break;
                }
            }
        }
        let fi = found.unwrap_or(cursor);
        idx[k] = fi;
        cursor = fi + 1;
    }

    // 辅助：顶点 a..=b 之间原始轮廓点到弦的最大垂直距离（拱高）
    let chord_max_dev = |a: usize, b: usize| -> f32 {
        let ca = pts[a % n];
        let cb = pts[b % n];
        let ia = idx[a % n];
        let ib = idx[b % n];
        let mut maxd = 0.0f32;
        let mut i = ia;
        loop {
            if i != ia && i != ib {
                let d = point_line_dist(loop_[i], ca, cb);
                if d > maxd {
                    maxd = d;
                }
            }
            if i == ib {
                break;
            }
            i = (i + 1) % loop_len;
            if i == ia {
                break;
            }
        }
        maxd
    };
    // 辅助：收集顶点 a..=b 之间的原始轮廓点
    let collect_run = |a: usize, b: usize| -> Vec<(f32, f32)> {
        let ia = idx[a % n];
        let ib = idx[b % n];
        let mut run = Vec::new();
        let mut i = ia;
        loop {
            run.push(loop_[i]);
            if i == ib {
                break;
            }
            i = (i + 1) % loop_len;
            if i == ia {
                break;
            }
        }
        run
    };

    // 起点：选第一条直边起步
    let mut start_e = 0usize;
    for e in 0..n {
        if chord_max_dev(e, (e + 1) % n) <= line_eps {
            start_e = e;
            break;
        }
    }

    // 每个顶点的有符号转角（弧度）
    let mut turn = vec![0.0f64; n];
    for k in 0..n {
        let a = pts[(k + n - 1) % n];
        let b = pts[k];
        let c = pts[(k + 1) % n];
        let v1 = (b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64);
        let v2 = (c.0 as f64 - b.0 as f64, c.1 as f64 - b.1 as f64);
        let cross = v1.0 * v2.1 - v1.1 * v2.0;
        let dot = v1.0 * v2.0 + v1.1 * v2.1;
        turn[k] = cross.atan2(dot);
    }
    const T_HI: f64 = 1.25; // ~72° 尖角门限
    const T_TURN_STRAIGHT: f64 = 0.2; // ~11°

    // ---------- Pass 1: 检测直线段 / 圆弧，收集为图元列表 ----------
    // 先把整条轮廓切成「直线(L)」与「真圆弧(A)」交替的图元，延迟到 Pass 3 统一发射。
    // 这样 Pass 2 才能两端同步校正弧端点而不破坏路径连通性。
    #[derive(Clone, Copy)]
    enum Item {
        Line(usize),
        Arc {
            i: usize,
            e_end: usize,
            ccx: f64,
            ccy: f64,
            r: f64,
            total: f64,
        },
    }
    let mut items: Vec<Item> = Vec::new();
    let mut i = start_e;
    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > n + 2 {
            break;
        }
        // 1) 弧运行（弧优先）
        let nxt0 = (i + 1) % n;
        if nxt0 != i && nxt0 != start_e {
            let arc_tol = arc_tol_abs as f64;
            let mut best_arc: Option<(f64, f64, f64, usize, f64, f32)> = None;
            let mut e = nxt0;
            let mut safety = 0usize;
            loop {
                let pnext = (e + 1) % n;
                if pnext == i || pnext == start_e {
                    break;
                }
                if turn[pnext].abs() > T_HI {
                    break;
                }
                safety += 1;
                if safety > 220 {
                    break;
                }
                let prun = collect_run(i, pnext);
                if let Some((ccx, ccy, rr, total, maxd)) =
                    fit_and_validate_arc(&prun, arc_tol, line_eps)
                {
                    best_arc = Some((ccx, ccy, rr, pnext, total, maxd));
                    e = pnext;
                    continue;
                }
                if best_arc.is_some() {
                    break;
                }
                if safety >= 20 {
                    break;
                }
                e = pnext;
            }
            if let Some((ccx, ccy, r, e_end, total, maxd)) = best_arc {
                let bow = chord_max_dev(i, e_end);
                if bow > line_eps && maxd < bow * 0.7 && total.abs() > 0.6 {
                    items.push(Item::Arc {
                        i,
                        e_end,
                        ccx,
                        ccy,
                        r,
                        total,
                    });
                    i = e_end;
                    if i == start_e {
                        break;
                    }
                    continue;
                }
            }
        }
        // 2) 直边运行
        let mut j = i;
        loop {
            let nxt = (j + 1) % n;
            if nxt == i || nxt == start_e {
                break;
            }
            if chord_max_dev(j, nxt) <= line_eps && turn[nxt].abs() < T_TURN_STRAIGHT {
                j = nxt;
            } else {
                break;
            }
        }
        if j == i {
            j = (i + 1) % n;
        }
        items.push(Item::Line(j));
        i = j;
        if i == start_e {
            break;
        }
    }

    // ---------- Pass 2: 弧端点切线校正（两端同步，保持路径连通） ----------
    // 把每个圆弧的起/止顶点投影到拟合圆上：若相邻有足够长的直边，则取「该直边与圆相切」的
    // 切点（圆弧与直边严格相切，且弦长<=直径，几何永有效）；否则退化为径向投影。
    // 关键：相邻直线段与圆弧共享同一顶点索引，校正 cp[k] 同时服务于前一段的终点与后一段
    // 的起点，因此整条路径天然连续，不会出现单端修正时的不对称/断裂。
    let mut cp: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let project_to_circle = |p: (f64, f64), cx: f64, cy: f64, r: f64| -> (f64, f64) {
        let dx = p.0 - cx;
        let dy = p.1 - cy;
        let d = (dx * dx + dy * dy).sqrt().max(1e-9);
        (cx + r * dx / d, cy + r * dy / d)
    };
    // 过点 p、方向 (dx,dy) 的直线与圆 (cx,cy,r) 的切点（取离 p 更近的那个候选）。
    let tangent_point = |p: (f64, f64), dx: f64, dy: f64, cx: f64, cy: f64, r: f64| -> (f64, f64) {
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let ux = dx / len;
        let uy = dy / len;
        let c1 = (cx + r * (-uy), cy + r * ux);
        let c2 = (cx + r * uy, cy + r * (-ux));
        let d1 = (c1.0 - p.0).powi(2) + (c1.1 - p.1).powi(2);
        let d2 = (c2.0 - p.0).powi(2) + (c2.1 - p.1).powi(2);
        if d1 < d2 {
            c1
        } else {
            c2
        }
    };
    for item in &items {
        if let Item::Arc {
            i,
            e_end,
            ccx,
            ccy,
            r,
            ..
        } = *item
        {
            // 入射方向：沿「简化顶点 i 之前的直边」向后走到尽头，取整段长边方向
            // （单段相邻顶点可能极短，如起点附近的闭合段，会误判为径向投影）。
            let mut a = i;
            loop {
                let prev = (a + n - 1) % n;
                if prev == i {
                    break;
                }
                if chord_max_dev(prev, a) <= line_eps && turn[prev].abs() < T_TURN_STRAIGHT {
                    a = prev;
                } else {
                    break;
                }
            }
            let cur = pts[i];
            let din = (cur.0 as f64 - pts[a].0 as f64, cur.1 as f64 - pts[a].1 as f64);
            if (din.0 * din.0 + din.1 * din.1).sqrt() > 3.0 {
                cp[i] = tangent_point((cur.0 as f64, cur.1 as f64), din.0, din.1, ccx, ccy, r);
            } else {
                cp[i] = project_to_circle((cur.0 as f64, cur.1 as f64), ccx, ccy, r);
            }
            // 出射方向：沿「简化顶点 e_end 之后的直边」向前走到尽头。
            let mut b = e_end;
            loop {
                let nxt = (b + 1) % n;
                if nxt == e_end {
                    break;
                }
                if chord_max_dev(b, nxt) <= line_eps && turn[nxt].abs() < T_TURN_STRAIGHT {
                    b = nxt;
                } else {
                    break;
                }
            }
            let cur2 = pts[e_end];
            let dout = (pts[b].0 as f64 - cur2.0 as f64, pts[b].1 as f64 - cur2.1 as f64);
            if (dout.0 * dout.0 + dout.1 * dout.1).sqrt() > 3.0 {
                cp[e_end] =
                    tangent_point((cur2.0 as f64, cur2.1 as f64), dout.0, dout.1, ccx, ccy, r);
            } else {
                cp[e_end] = project_to_circle((cur2.0 as f64, cur2.1 as f64), ccx, ccy, r);
            }
        }
    }

    // ---------- Pass 3: 发射（所有顶点用校正后的 cp） ----------
    let mut s = format!("M {:.2} {:.2}", cp[start_e].0, cp[start_e].1);
    for item in &items {
        match *item {
            Item::Line(j) => {
                let p = cp[j];
                s.push_str(&format!(" L {:.2} {:.2}", p.0, p.1));
            }
            Item::Arc {
                i,
                e_end,
                ccx,
                ccy,
                r,
                total,
            } => {
                let p0 = cp[i];
                let p1 = cp[e_end];
                // 由「拟合圆心」反推 (large, sweep)，确保 SVG 弧的圆心落在正确一侧：
                // sub-180° 弧若仅按 total 符号取 sweep，圆心会错配到弦的另一侧 → 畸形；
                // 半圆(h=0)时两侧重合，want_plus 取等→sweep=0，与旧版半圆行为一致。
                let large = if total.abs() > std::f64::consts::PI {
                    1
                } else {
                    0
                };
                let mx = (p0.0 + p1.0) * 0.5;
                let my = (p0.1 + p1.1) * 0.5;
                let ddx = p1.0 - p0.0;
                let ddy = p1.1 - p0.1;
                let clen = (ddx * ddx + ddy * ddy).sqrt();
                let hh = (r * r - (clen * 0.5).powi(2)).max(0.0).sqrt();
                let nx = -ddy / clen;
                let ny = ddx / clen;
                let cpc = (mx + hh * nx, my + hh * ny);
                let cpm = (mx - hh * nx, my - hh * ny);
                let want_plus = (cpc.0 - ccx).powi(2) + (cpc.1 - ccy).powi(2)
                    < (cpm.0 - ccx).powi(2) + (cpm.1 - ccy).powi(2);
                let sweep = if want_plus {
                    if large == 1 {
                        0
                    } else {
                        1
                    }
                } else if large == 1 {
                    1
                } else {
                    0
                };
                s.push_str(&format!(
                    " A {:.2} {:.2} 0 {} {} {:.2} {:.2}",
                    r, r, large, sweep, p1.0, p1.1
                ));
            }
        }
    }
    s.push_str(" Z");
    s
}
