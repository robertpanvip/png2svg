//! 曲率驱动的三次贝塞尔重建。
//!
//! 思路：沿轮廓算每点的「切线」（中心差分单位向量）与「有符号曲率」（顶点处转角 / 段长），
//! 据此把闭环在曲率局部极值 / 拐点 / 折角处切成若干段（放置结点）；然后用这些结点作为
//! 控制点做闭合 Catmull-Rom，段内曲率单调、CR 用相邻结点估计切线，得到与曲率特征点数量
//! 相当的少数几条干净三次贝塞尔（而非逐点插值或过度细分）。直线多边形不会进入此分支
//! （coarse ≤ 12，已走 L 多边形），因此不会被磨圆。

use crate::contour::simplify::simplify_polygon;

/// 闭合 Catmull-Rom 样条 → 三次贝塞尔路径（tension=1 标准系数）。
/// 经过所有控制点，在尖角处可能轻微过冲；因此只对曲线型轮廓使用（见 curvature_bezier_path）。
/// 现主要作为曲率重建在噪声轮廓下的兜底。
fn catmull_rom_path(pts: &[(f32, f32)]) -> String {
    let n = pts.len();
    if n < 3 {
        return String::new();
    }
    let mut s = format!("M {} {}", pts[0].0, pts[0].1);
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        let c1x = p1.0 + (p2.0 - p0.0) / 6.0;
        let c1y = p1.1 + (p2.1 - p0.1) / 6.0;
        let c2x = p2.0 - (p3.0 - p1.0) / 6.0;
        let c2y = p2.1 - (p3.1 - p1.1) / 6.0;
        s.push_str(&format!(
            " C {} {} {} {} {} {}",
            c1x, c1y, c2x, c2y, p2.0, p2.1
        ));
    }
    s.push_str(" Z");
    s
}

/// 用「轮廓点 + 切线 + 曲率分析」重建闭合曲线的三次贝塞尔路径。
/// 步骤：算每点中心差分切线与有符号曲率；平滑曲率抑制像素台阶噪声；在平滑后曲率的
/// 局部极值 / 拐点 / 折角处放置结点（要求最小角间距、上限 24 个，避免噪声产生密集结点）；
/// 结点作为控制点做闭合 Catmull-Rom，得到与曲率特征数量相当的少数干净贝塞尔。若曲线
/// 仍极曲折（结点数超限）则退化为较粗 RDP 简化 + Catmull-Rom。直线多边形不进入此分支
/// （coarse ≤ 12，已走 L 多边形），因此不会被磨圆。
pub fn curvature_bezier_path(pts: &[(f32, f32)]) -> String {
    let n = pts.len();
    if n < 3 {
        return String::new();
    }
    let p: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    // 切线（中心差分，单位向量）
    let mut tan = vec![(0.0, 0.0); n];
    for i in 0..n {
        let a = p[(i + n - 1) % n];
        let b = p[(i + 1) % n];
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt();
        if l > 1e-9 {
            tan[i] = (dx / l, dy / l);
        }
    }
    // 有符号曲率（每顶点转角 / 段长）
    let mut kappa = vec![0.0; n];
    for i in 0..n {
        let tp = tan[(i + n - 1) % n];
        let tc = tan[i];
        let cross = tp.0 * tc.1 - tp.1 * tc.0;
        let dot = tp.0 * tc.0 + tp.1 * tc.1;
        let phi = cross.atan2(dot);
        let a = p[(i + n - 1) % n];
        let b = p[i];
        let seg = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        kappa[i] = if seg > 1e-9 { phi / seg } else { 0.0 };
    }
    // 平滑曲率：像素台阶会让原始曲率剧烈抖动，用滑动平均抑制噪声，只保留真实特征。
    let kw = 3usize;
    let ks: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            let mut c = 0;
            for d in -(kw as i32)..=(kw as i32) {
                let idx = (i as i32 + d + n as i32) as usize % n;
                s += kappa[idx];
                c += 1;
            }
            s / c as f64
        })
        .collect();
    // 放置结点：平滑后曲率的“显著”极值（局部极大或极小）/ 拐点 / 折角。
    // 显著极值 = 与左右邻点的曲率差都超过一定相对比例，这样既能抓住椭圆的长轴端点
    // （曲率极大），也能抓住短轴端点（曲率极小），而不被像素噪声骗出大量结点。
    let corner = 0.6; // ~34°
    let mut knots: Vec<usize> = vec![0];
    for i in 0..n {
        let kp = ks[(i + n - 1) % n].abs();
        let kc = ks[i].abs();
        let kn = ks[(i + 1) % n].abs();
        let drop_l = (kp - kc).abs();
        let drop_r = (kn - kc).abs();
        let local_ref = kp.max(kn).max(1e-6);
        let is_extremum = i != 0 && drop_l > 0.25 * local_ref && drop_r > 0.25 * local_ref;
        let is_inflection = ks[(i + n - 1) % n].abs() > 1e-6
            && ks[i].abs() > 1e-6
            && ks[(i + n - 1) % n] * ks[i] < 0.0;
        let tp = tan[(i + n - 1) % n];
        let tc = tan[i];
        let phi = (tp.0 * tc.1 - tp.1 * tc.0).atan2(tp.0 * tc.0 + tp.1 * tc.1);
        let is_corner = phi.abs() > corner;
        if is_extremum || is_inflection || is_corner {
            knots.push(i);
        }
    }
    if knots.len() == 1 {
        // 没有任何曲率特征点（接近圆等曲率恒定的曲线）：按累计转角每 ~90° 放一个结点
        let mut cum = 0.0;
        let target = std::f64::consts::FRAC_PI_2; // 90°
        for i in 1..n {
            let tp = tan[(i + n - 1) % n];
            let tc = tan[i];
            let cross = tp.0 * tc.1 - tp.1 * tc.0;
            let dot = tp.0 * tc.0 + tp.1 * tc.1;
            cum += cross.atan2(dot);
            if cum >= target {
                knots.push(i);
                cum -= target;
            }
        }
    }
    knots.sort_unstable();
    knots.dedup();
    // 最小间距，避免相邻噪声结点
    let min_gap = (n / 24).max(2);
    let mut filtered: Vec<usize> = Vec::new();
    for &k in &knots {
        if let Some(&last) = filtered.last() {
            let gap = (k + n - last) % n;
            if gap < min_gap {
                continue;
            }
        }
        filtered.push(k);
    }
    // 结点数仍过多（极曲折曲线）→ 退化为较粗 RDP 简化 + Catmull-Rom，保证贝塞尔条数可控
    if filtered.len() > 24 {
        let (mut minx, mut miny, mut maxx, mut maxy) =
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in &p {
            if x < minx {
                minx = x;
            }
            if y < miny {
                miny = y;
            }
            if x > maxx {
                maxx = x;
            }
            if y > maxy {
                maxy = y;
            }
        }
        let diag = ((maxx - minx).powi(2) + (maxy - miny).powi(2)).sqrt();
        let ctrl_eps = (diag * 0.02).max(2.0) as f32;
        let ctrl = simplify_polygon(pts, ctrl_eps);
        if ctrl.len() >= 3 {
            return catmull_rom_path(&ctrl);
        }
        let mut s = format!("M {} {}", ctrl[0].0, ctrl[0].1);
        for q in &ctrl[1..] {
            s.push_str(&format!(" L {} {}", q.0, q.1));
        }
        s.push_str(" Z");
        return s;
    }
    // 用结点作为控制点做闭合 Catmull-Rom：每段一条三次贝塞尔，段数与曲率特征点
    // 数量相当（而非逐点或过度细分）。结点已落在曲率极值 / 拐点 / 折角处，段内曲率
    // 单调，CR 用相邻结点估计切线，拟合良好且平滑。直线多边形不会进入此分支
    // （coarse ≤ 12，已走 L 多边形）。
    let m = filtered.len();
    if m < 3 {
        // 结点过少（极端情况）→ 退化为逐点 Catmull-Rom，保证不退化
        return catmull_rom_path(pts);
    }
    let kp: Vec<(f64, f64)> = filtered.iter().map(|&k| p[k]).collect();
    let mut s = format!("M {:.2} {:.2}", kp[0].0, kp[0].1);
    for i in 0..m {
        let p0 = kp[(i + m - 1) % m];
        let p1 = kp[i];
        let p2 = kp[(i + 1) % m];
        let p3 = kp[(i + 2) % m];
        let c1 = (p1.0 + (p2.0 - p0.0) / 6.0, p1.1 + (p2.1 - p0.1) / 6.0);
        let c2 = (p2.0 - (p3.0 - p1.0) / 6.0, p2.1 - (p3.1 - p1.1) / 6.0);
        s.push_str(&format!(
            " C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
            c1.0, c1.1, c2.0, c2.1, p2.0, p2.1
        ));
    }
    s.push_str(" Z");
    s
}
