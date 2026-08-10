//! 图元拟合（圆 / 椭圆）与单圆弧校验。
//!
//! 提供两类能力：
//!   - 整形状拟合：fit_circle（Pratt 归一化，紧残差判据，用于识别真圆）、
//!     fit_rotated_ellipse（PCA 主轴 + 轴对齐最小二乘，识别真椭圆）；
//!   - 局部弧校验：fit_and_validate_arc —— 对一段简化轮廓做"单一圆弧"判定，
//!     内部用 fit_circle_ls（Pratt 归一化最小二乘）拟合参考圆并校验全部点落在圆上。
//!
//! 所有坐标直接采用像素坐标（icon 约 200px 量级），对 Pratt 归一化友好。

/// 解 3×3 线性方程组（高斯消元，列主元）。返回 None 表示奇异。
pub(crate) fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = m;
    let mut c = b;
    for col in 0..3 {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..3 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            c.swap(col, piv);
        }
        let d = a[col][col];
        for r in col + 1..3 {
            let f = a[r][col] / d;
            for k in col..3 {
                a[r][k] -= f * a[col][k];
            }
            c[r] -= f * c[col];
        }
    }
    let mut x = [0f64; 3];
    for i in (0..3).rev() {
        let mut s = c[i];
        for j in i + 1..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

fn solve2(m: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-12 {
        return None;
    }
    Some([
        (m[1][1] * b[0] - m[0][1] * b[1]) / det,
        (m[0][0] * b[1] - m[1][0] * b[0]) / det,
    ])
}

/// 最小二乘圆拟合（Pratt 归一化，降低半径偏差）。返回 (cx, cy, r, rmse)。
pub(crate) fn fit_circle(pts: &[(f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    if pts.len() < 5 {
        return None;
    }
    let mut ata = [[0f64; 3]; 3];
    let mut atb = [0f64; 3];
    for &(x, y) in pts {
        let x = x as f64;
        let y = y as f64;
        let z = x * x + y * y;
        let zn = z.sqrt().max(1e-9);
        let row = [x / zn, y / zn, 1.0 / zn];
        let rhs = (-z) / zn;
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] += row[i] * rhs;
        }
    }
    let sol = solve3(ata, atb)?;
    let (aa, bb, cc) = (sol[0], sol[1], sol[2]);
    let cx = -aa / 2.0;
    let cy = -bb / 2.0;
    let r2 = cx * cx + cy * cy - cc;
    if r2 <= 0.0 || !r2.is_finite() {
        return None;
    }
    let r = r2.sqrt();
    if r <= 0.0 {
        return None;
    }
    let n = pts.len() as f64;
    let mut se = 0.0;
    for &(x, y) in pts {
        let d = ((x as f64 - cx).hypot(y as f64 - cy) - r).abs();
        se += d * d;
    }
    let rmse = (se / n).sqrt();
    Some((cx as f32, cy as f32, r as f32, rmse as f32))
}

/// 给定圆心，最小二乘轴对齐椭圆拟合 (xi-cx)^2/rx^2 + (yi-cy)^2/ry^2 = 1。
/// 令 u=1/rx^2, v=1/ry^2，方程变为 u*A + v*B = 1（对 u,v 线性）。返回 (cx, cy, rx, ry, rmse)。
fn fit_axis_ellipse(
    pts: &[(f32, f32)],
    cx: f64,
    cy: f64,
) -> Option<(f32, f32, f32, f32, f32)> {
    if pts.len() < 5 {
        return None;
    }
    let mut ata = [[0f64; 2]; 2];
    let mut atb = [0f64; 2];
    for &(x, y) in pts {
        let dx = x as f64 - cx;
        let dy = y as f64 - cy;
        let a = dx * dx;
        let b = dy * dy;
        ata[0][0] += a * a;
        ata[0][1] += a * b;
        ata[1][0] += a * b;
        ata[1][1] += b * b;
        atb[0] += a;
        atb[1] += b;
    }
    let sol = solve2(ata, atb)?;
    let (u, v) = (sol[0], sol[1]);
    if u <= 0.0 || v <= 0.0 {
        return None;
    }
    let rx = 1.0 / u.sqrt();
    let ry = 1.0 / v.sqrt();
    if !rx.is_finite() || !ry.is_finite() {
        return None;
    }
    let n = pts.len() as f64;
    let mut se = 0.0;
    for &(x, y) in pts {
        let dx = x as f64 - cx;
        let dy = y as f64 - cy;
        let val = (dx * dx / (rx * rx) + dy * dy / (ry * ry)).sqrt();
        se += (val - 1.0).powi(2);
    }
    let rmse = (se / n).sqrt();
    Some((cx as f32, cy as f32, rx as f32, ry as f32, rmse as f32))
}

/// 旋转椭圆拟合（涵盖轴对齐情形）：先 PCA 估计主轴方向 theta，旋转到主轴坐标系后做
/// 轴对齐椭圆最小二乘拟合，返回 (cx, cy, rx, ry, theta_rad, rmse)。theta 为局部 x 轴
/// 到主轴(长轴)的逆时针角（弧度）。残差沿用 fit_axis_ellipse 的归一化定义，与
/// ell_tol 保持同一量纲。对完整椭圆（边界点对称）PCA 主轴与椭圆轴精确对齐，theta 无偏。
pub(crate) fn fit_rotated_ellipse(pts: &[(f32, f32)]) -> Option<(f32, f32, f32, f32, f32, f32)> {
    let n = pts.len();
    if n < 6 {
        return None;
    }
    let (mx, my) = centroid(pts);
    // 协方差矩阵 -> 主轴角 theta
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in pts {
        let dx = x as f64 - mx;
        let dy = y as f64 - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let n64 = n as f64;
    let cxx = sxx / n64;
    let cyy = syy / n64;
    let cxy = sxy / n64;
    // tan(2*theta) = 2*cxy/(cxx-cyy)
    let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let (ct, st) = (theta.cos(), theta.sin());
    // 旋转到主轴坐标系（中心移到质心/原点）
    let mut rot: Vec<(f32, f32)> = Vec::with_capacity(n);
    for &(x, y) in pts {
        let dx = x as f64 - mx;
        let dy = y as f64 - my;
        let u = dx * ct + dy * st;
        let v = -dx * st + dy * ct;
        rot.push((u as f32, v as f32));
    }
    // 轴对齐椭圆拟合（中心固定在原点）；残差即旋转椭圆的拟合质量
    let (_, _, rx, ry, eres) = fit_axis_ellipse(&rot, 0.0, 0.0)?;
    Some((mx as f32, my as f32, rx, ry, theta as f32, eres))
}

fn centroid(pts: &[(f32, f32)]) -> (f64, f64) {
    let (mut sx, mut sy) = (0.0, 0.0);
    for &(x, y) in pts {
        sx += x as f64;
        sy += y as f64;
    }
    let n = pts.len() as f64;
    (sx / n, sy / n)
}

/// Pratt 归一化最小二乘圆拟合：解 x^2+y^2 + A*x + B*y + C = 0 的正规方程，
/// 每行除以 sqrt(x^2+y^2+1) 归一化（Pratt 方法），消除 Kåsa 普通最小二乘对远离原点
/// 点赋予过大权重的偏差。对 <180° 弧段（圆角/胶囊半圆）半径估计比 Kåsa 更准，
/// 尤其当弧的几何中心不在坐标原点附近时。
/// 返回 (cx, cy, r)。半径超出 [1, 400] 或退化返回 None。
pub(crate) fn fit_circle_ls(run: &[(f32, f32)]) -> Option<(f64, f64, f64)> {
    let n = run.len();
    if n < 6 {
        return None;
    }
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for p in run {
        let x = p.0 as f64;
        let y = p.1 as f64;
        let z = x * x + y * y;
        let w = 1.0 / (z + 1.0).sqrt(); // Pratt normalization weight
        let row = [x * w, y * w, w];
        let rhs_v = (-z) * w;
        for i in 0..3 {
            atb[i] += row[i] * rhs_v;
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let sol = solve3(ata, atb)?;
    let a = sol[0];
    let b = sol[1];
    let c = sol[2];
    let cx = -a / 2.0;
    let cy = -b / 2.0;
    let r2 = cx * cx + cy * cy - c;
    if r2 <= 0.0 {
        return None;
    }
    let r = r2.sqrt();
    if !r.is_finite() || r < 1.0 || r > 400.0 {
        return None;
    }
    Some((cx, cy, r))
}

#[inline]
fn angle_of(p: (f32, f32), cx: f64, cy: f64) -> f64 {
    (p.1 as f64 - cy).atan2(p.0 as f64 - cx)
}

/// 对一组按环形顺序排列的轮廓点做「单一圆弧」判定：用全部点做 Pratt 最小二乘圆拟合，
/// 再校验全部点落在容差 `tol` 内、绕心角度单调、总转角 < ~350°。
/// 成功返回 (cx, cy, r, 总扫角, 最大偏差)。
///
/// 关键：必须在「跨多段」的窗口上拟合，而非仅首段——细采样的小圆角首段近乎直线，三点定圆会得到
/// 巨大（浅）半径而被 `rr>200` 过滤；只有窗口足够长（>=8 点，约 20°+ 弧）才能稳定恢复真实半径。
/// 直边因三点近似共线 -> `rr` 远超 200 自然被拒，不必额外判"直"。
pub(crate) fn fit_and_validate_arc(
    run: &[(f32, f32)],
    tol: f64,
    _line_eps: f32,
) -> Option<(f64, f64, f64, f64, f32)> {
    if run.len() < 8 {
        return None; // window too short
    }
    // Pratt LS circle fit (full window), robust to raster noise.
    let (ccx, ccy, rr) = fit_circle_ls(run)?;
    if rr < 1.0 || rr > 300.0 {
        return None;
    }
    let mut prev = angle_of(run[0], ccx, ccy);
    let mut dir = 0i32;
    let mut total = 0.0f64;
    let mut maxd = 0.0f64;
    for q in run {
        let dist = ((q.0 as f64 - ccx).hypot(q.1 as f64 - ccy) - rr).abs();
        if dist > maxd {
            maxd = dist;
        }
        if dist > tol {
            return None;
        }
        let ang = angle_of(*q, ccx, ccy);
        let mut da = ang - prev;
        while da > std::f64::consts::PI {
            da -= 2.0 * std::f64::consts::PI;
        }
        while da < -std::f64::consts::PI {
            da += 2.0 * std::f64::consts::PI;
        }
        if dir == 0 && da > 1e-9 {
            dir = 1;
        } else if dir == 0 && da < -1e-9 {
            dir = -1;
        } else if (da > 1e-9 && dir < 0) || (da < -1e-9 && dir > 0) {
            return None;
        }
        total += da;
        prev = ang;
    }
    // Allow >180 deg arcs (up to ~350 deg) for teardrop-like union shapes
    // where the circular body spans >180 deg. Full circles (360 deg) are handled
    // by the upper-level fit_circle branch; leave margin to avoid degenerate full-circle.
    if total.abs() > 2.0 * std::f64::consts::PI - 0.2 {
        return None;
    }
    Some((ccx, ccy, rr, total, maxd as f32))
}
