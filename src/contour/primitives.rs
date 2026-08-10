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
                if d > maxd { maxd = d; }
            }
            if i == ib { break; }
            i = (i + 1) % loop_len;
            if i == ia { break; }
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
            if i == ib { break; }
            i = (i + 1) % loop_len;
            if i == ia { break; }
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

    // 生成路径
    let mut s = format!("M {} {}", pts[start_e].0, pts[start_e].1);
    let mut i = start_e;
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > n + 2 { break; }
        // 1) 弧运行（弧优先）
        let nxt0 = (i + 1) % n;
        let mut arc_emitted = false;
        if nxt0 != i && nxt0 != start_e {
            let arc_tol = arc_tol_abs as f64;
            let mut best_arc: Option<(f64, f64, f64, usize, f64, f32)> = None;
            let mut e = nxt0;
            let mut safety = 0usize;
            loop {
                let pnext = (e + 1) % n;
                if pnext == i || pnext == start_e { break; }
                if turn[pnext].abs() > T_HI { break; }
                safety += 1;
                if safety > 220 { break; }
                let prun = collect_run(i, pnext);
                if let Some((ccx, ccy, rr, total, maxd)) =
                    fit_and_validate_arc(&prun, arc_tol, line_eps)
                {
                    best_arc = Some((ccx, ccy, rr, pnext, total, maxd));
                    e = pnext;
                    continue;
                }
                if best_arc.is_some() { break; }
                if safety >= 20 { break; }
                e = pnext;
            }
            if let Some((ccx, ccy, r, e_end, total, maxd)) = best_arc {
                let bow = chord_max_dev(i, e_end);
                if bow > line_eps && maxd < bow * 0.7 && total.abs() > 0.6 {
                    let end = pts[e_end];
                    let large = if total.abs() > std::f64::consts::PI { 1 } else { 0 };
                    let p0 = pts[i];
                    let p1 = end;
                    let mx = (p0.0 as f64 + p1.0 as f64) * 0.5;
                    let my = (p0.1 as f64 + p1.1 as f64) * 0.5;
                    let ddx = p1.0 as f64 - p0.0 as f64;
                    let ddy = p1.1 as f64 - p0.1 as f64;
                    let clen = (ddx * ddx + ddy * ddy).sqrt();
                    let hh = (r * r - (clen * 0.5).powi(2)).max(0.0).sqrt();
                    let nx = -ddy / clen;
                    let ny = ddx / clen;
                    let cp = (mx + hh * nx, my + hh * ny);
                    let cm = (mx - hh * nx, my - hh * ny);
                    let want_plus = (cp.0 - ccx).powi(2) + (cp.1 - ccy).powi(2)
                        < (cm.0 - ccx).powi(2) + (cm.1 - ccy).powi(2);
                    let sweep = if want_plus {
                        if large == 1 { 0 } else { 1 }
                    } else {
                        if large == 1 { 1 } else { 0 }
                    };
                    s.push_str(&format!(
                        " A {:.2} {:.2} 0 {} {} {:.2} {:.2}",
                        r, r, large, sweep, end.0, end.1
                    ));
                    i = e_end;
                    if i == start_e { break; }
                    arc_emitted = true;
                }
            }
        }
        if arc_emitted { continue; }
        // 2) 直边运行
        let mut j = i;
        loop {
            let nxt = (j + 1) % n;
            if nxt == i || nxt == start_e { break; }
            if chord_max_dev(j, nxt) <= line_eps && turn[nxt].abs() < T_TURN_STRAIGHT {
                j = nxt;
            } else { break; }
        }
        if j == i { j = (i + 1) % n; }
        let end = pts[j];
        s.push_str(&format!(" L {} {}", end.0, end.1));
        i = j;
        if i == start_e { break; }
    }
    s.push_str(" Z");
    s
}
