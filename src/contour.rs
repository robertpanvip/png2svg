//! Step 1：轮廓提取（黑白）。
//!
//! 核心算法是 Moore 邻域边界跟踪：沿 8 邻域顺时针扫描，始终让“前景”保持在
//! 右侧，从而描出闭合轮廓。配合：
//!   - 4 连通分量提取（分离互不相连的形状）；
//!   - 背景可达性洪泛（从图像边界出发），用来识别“孔洞”（被前景完全包围的背景）；
//!   - RDP 多边形简化（去掉冗余点，缩小 SVG 体积）。
//!
//! 输出的每个形状都带“外边界 + 内边界（孔洞）”，用 fill-rule=evenodd 形成真洞。

/// 8 邻域，顺时针：E, SE, S, SW, W, NW, N, NE
const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// 4 邻域坐标迭代。
pub fn four_neighbors(x: u32, y: u32, w: u32, h: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(4);
    if x > 0 {
        out.push((x - 1, y));
    }
    if x + 1 < w {
        out.push((x + 1, y));
    }
    if y > 0 {
        out.push((x, y - 1));
    }
    if y + 1 < h {
        out.push((x, y + 1));
    }
    out
}

fn dir_index(dx: i32, dy: i32) -> usize {
    for (i, &(ox, oy)) in DIRS.iter().enumerate() {
        if ox == dx && oy == dy {
            return i;
        }
    }
    4 // 默认朝西（理想起始方向）
}

#[inline]
fn get(mask: &[bool], w: u32, h: u32, x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        false
    } else {
        mask[(y as u32 * w + x as u32) as usize]
    }
}

/// Moore 邻域跟踪，描出单个前景连通分量的【外边界】。
/// 要求 start 是边界像素（其西侧为背景），否则也能产出闭合环，但起点可能不规整。
fn trace_component(mask: &[bool], w: u32, h: u32, start: (u32, u32)) -> Vec<(f32, f32)> {
    let (sx, sy) = (start.0 as i32, start.1 as i32);
    let mut current = (sx, sy);
    let mut prev = (sx - 1, sy); // 假设从西侧进入（起点最左，西为空白）
    let mut boundary = vec![(sx as f32, sy as f32)];
    let max_iter = (w * h) as i32 * 8 + 256;
    let mut guard = 0;

    loop {
        guard += 1;
        if guard > max_iter {
            break;
        }
        let d = dir_index(prev.0 - current.0, prev.1 - current.1);
        let mut found: Option<usize> = None;
        for k in 1..=8 {
            let nd = ((d as i32 + k) % 8) as usize;
            let nx = current.0 + DIRS[nd].0;
            let ny = current.1 + DIRS[nd].1;
            if get(mask, w, h, nx, ny) {
                found = Some(nd);
                break;
            }
        }
        let nd = match found {
            Some(v) => v,
            None => break,
        };
        // 回溯点 = 扫描序列中 nd 之前的那一个（顺时针）
        let back_idx = (nd + 8 - 1) % 8;
        let bx = current.0 + DIRS[back_idx].0;
        let by = current.1 + DIRS[back_idx].1;
        let np = (current.0 + DIRS[nd].0, current.1 + DIRS[nd].1);
        prev = (bx, by);
        current = np;
        if current == (sx, sy) {
            break;
        }
        boundary.push((current.0 as f32, current.1 as f32));
    }
    boundary
}

/// 从图像边界出发，标记所有“能到达边界的背景像素”（即图形外的背景）。
fn compute_outside_background(mask: &[bool], w: u32, h: u32) -> Vec<bool> {
    let n = mask.len();
    let mut outside = vec![false; n];
    let mut stack = Vec::new();
    let push_if = |stack: &mut Vec<(u32, u32)>, outside: &mut Vec<bool>, x: u32, y: u32| {
        let i = (y * w + x) as usize;
        if !mask[i] && !outside[i] {
            outside[i] = true;
            stack.push((x, y));
        }
    };
    for x in 0..w {
        push_if(&mut stack, &mut outside, x, 0);
        push_if(&mut stack, &mut outside, x, h - 1);
    }
    for y in 1..h.saturating_sub(1) {
        push_if(&mut stack, &mut outside, 0, y);
        push_if(&mut stack, &mut outside, w - 1, y);
    }
    while let Some((x, y)) = stack.pop() {
        for (nx, ny) in four_neighbors(x, y, w, h) {
            push_if(&mut stack, &mut outside, nx, ny);
        }
    }
    outside
}

/// 取一个掩码的所有闭合轮廓：外边界 + 所有孔洞（内边界）。
/// 返回的每个 Vec 是“一个形状”的若干环（第一环外边界，其余为孔洞）。
pub fn trace_with_holes(mask: &[bool], w: u32, h: u32) -> Vec<Vec<(f32, f32)>> {
    // 外边界起点：行优先第一个前景像素（其北/西必为背景）
    let outer_start = (|| {
        for y in 0..h {
            for x in 0..w {
                if mask[(y * w + x) as usize] {
                    return Some((x, y));
                }
            }
        }
        None
    })();
    let outer_start = match outer_start {
        Some(s) => s,
        None => return Vec::new(),
    };
    let outer = trace_component(mask, w, h, outer_start);

    let outside = compute_outside_background(mask, w, h);
    let mut visited_hole = vec![false; mask.len()];
    let mut loops: Vec<Vec<(f32, f32)>> = vec![outer];

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if mask[i] || outside[i] || visited_hole[i] {
                continue;
            }
            // 找到一个孔洞连通分量
            let mut tmp = vec![false; mask.len()];
            let mut stack = vec![(x, y)];
            let mut comp = Vec::new();
            while let Some((cx, cy)) = stack.pop() {
                let ci = (cy * w + cx) as usize;
                if visited_hole[ci] || tmp[ci] || mask[ci] || outside[ci] {
                    continue;
                }
                visited_hole[ci] = true;
                tmp[ci] = true;
                comp.push((cx, cy));
                for (nx, ny) in four_neighbors(cx, cy, w, h) {
                    let ni = (ny * w + nx) as usize;
                    if !mask[ni] && !outside[ni] && !visited_hole[ni] {
                        stack.push((nx, ny));
                    }
                }
            }
            // 选一个与前景相邻的像素作为起始点
            let hstart = comp
                .iter()
                .find(|&&(cx, cy)| {
                    four_neighbors(cx, cy, w, h)
                        .iter()
                        .any(|&(nx, ny)| mask[(ny * w + nx) as usize])
                })
                .copied()
                .unwrap_or(comp[0]);
            let mut hmask = vec![false; mask.len()];
            for &(cx, cy) in &comp {
                hmask[(cy * w + cx) as usize] = true;
            }
            let hole = trace_component(&hmask, w, h, hstart);
            loops.push(hole);
        }
    }
    loops
}

/// 提取所有 4 连通前景分量，每个返回独立掩码。
pub fn components(mask: &[bool], w: u32, h: u32) -> Vec<Vec<bool>> {
    let n = mask.len();
    let mut visited = vec![false; n];
    let mut comps = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if !mask[i] || visited[i] {
                continue;
            }
            let mut cmask = vec![false; n];
            let mut stack = vec![(x, y)];
            while let Some((cx, cy)) = stack.pop() {
                let ci = (cy * w + cx) as usize;
                if visited[ci] || !mask[ci] {
                    continue;
                }
                visited[ci] = true;
                cmask[ci] = true;
                for (nx, ny) in four_neighbors(cx, cy, w, h) {
                    let ni = (ny * w + nx) as usize;
                    if !visited[ni] && mask[ni] {
                        stack.push((nx, ny));
                    }
                }
            }
            comps.push(cmask);
        }
    }
    comps
}

// ---------------- 多边形简化 ----------------

fn point_line_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (x0, y0) = p;
    let (x1, y1) = a;
    let (x2, y2) = b;
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 {
        ((x0 - x1).powi(2) + (y0 - y1).powi(2)).sqrt()
    } else {
        let t = ((x0 - x1) * dx + (y0 - y1) * dy) / len2;
        let t = t.clamp(0.0, 1.0);
        let cx = x1 + t * dx;
        let cy = y1 + t * dy;
        ((x0 - cx).powi(2) + (y0 - cy).powi(2)).sqrt()
    }
}

fn rdp(points: &[(f32, f32)], eps: f32, out: &mut Vec<(f32, f32)>) {
    if points.len() < 3 {
        out.extend_from_slice(points);
        return;
    }
    let a = points[0];
    let b = points[points.len() - 1];
    let mut max_d = 0.0;
    let mut idx = 0;
    for i in 1..points.len() - 1 {
        let d = point_line_dist(points[i], a, b);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > eps {
        let mut left = Vec::new();
        rdp(&points[0..=idx], eps, &mut left);
        let mut right = Vec::new();
        rdp(&points[idx..], eps, &mut right);
        left.pop(); // 避免重复中点
        out.extend(left);
        out.extend(right);
    } else {
        out.push(a);
        out.push(b);
    }
}

/// 去掉重复点、共线点，再做 RDP 简化。
pub fn simplify_polygon(points: &[(f32, f32)], eps: f32) -> Vec<(f32, f32)> {
    // 去重
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(points.len());
    for &p in points {
        if pts.last() != Some(&p) {
            pts.push(p);
        }
    }
    if pts.len() < 3 {
        return pts;
    }
    // 去共线
    let mut red = Vec::with_capacity(pts.len());
    for i in 0..pts.len() {
        let prev = pts[(i + pts.len() - 1) % pts.len()];
        let cur = pts[i];
        let next = pts[(i + 1) % pts.len()];
        let ax = cur.0 - prev.0;
        let ay = cur.1 - prev.1;
        let bx = next.0 - cur.0;
        let by = next.1 - cur.1;
        let cross = ax * by - ay * bx;
        let area = (ax * ax + ay * ay) * (bx * bx + by * by);
        if area < 1e-9 || cross * cross > 1e-6 * area {
            red.push(cur);
        }
    }
    if red.len() < 3 {
        return red;
    }
    let mut out = Vec::new();
    rdp(&red, eps, &mut out);
    out
}

/// 把若干闭环拼成一个 SVG path（用 fill-rule=evenodd 时，第二环起为孔洞）。
/// 每个闭环按【圆 → 旋转椭圆 → 贝塞尔/多边形】的优先级识别：圆/椭圆用弧命令
/// （真圆/真椭圆，旋转椭圆带 x-axis-rotation）；都不是才回退——若允许平滑且点数较多
/// （曲线型轮廓），用「切线 + 曲率分析」重建为少数几条干净三次贝塞尔（C 命令）；
/// 否则保留多边形（L 命令）。弧/曲线/多边形可共存于同一 path，故孔洞(evenodd)仍成立。
pub fn loops_to_path(
    loops: &[Vec<(f32, f32)>],
    eps: f32,
    circ_tol: f32,
    ell_tol: f32,
    smooth: bool,
) -> String {
    let mut s = String::new();
    for lp in loops {
        // 跳过 1~2px 的孤立噪点/碎环（抗锯齿偶尔产生的单像素碎片），避免输出 0 面积 sliver。
        if lp.len() >= 1 {
            let mut minx = f32::MAX;
            let mut miny = f32::MAX;
            let mut maxx = f32::MIN;
            let mut maxy = f32::MIN;
            for &(x, y) in lp {
                if x < minx { minx = x; }
                if y < miny { miny = y; }
                if x > maxx { maxx = x; }
                if y > maxy { maxy = y; }
            }
            if (maxx - minx) < 2.0 && (maxy - miny) < 2.0 {
                continue;
            }
        }
        let sub = loop_to_subpath(lp, eps, circ_tol, ell_tol, smooth);
        if sub.is_empty() {
            continue;
        }
        s.push_str(&sub);
    }
    s
}

// ---------------- 图元拟合（圆 / 椭圆） ----------------

fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
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
fn fit_circle(pts: &[(f32, f32)]) -> Option<(f32, f32, f32, f32)> {
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

/// 给定圆心，最小二乘轴对齐椭圆拟合 (xi-cx)²/rx² + (yi-cy)²/ry² = 1。
/// 令 u=1/rx², v=1/ry²，方程变为 u·A + v·B = 1（对 u,v 线性）。返回 (cx, cy, rx, ry, rmse)。
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

/// 旋转椭圆拟合（涵盖轴对齐情形）：先 PCA 估计主轴方向 θ，旋转到主轴坐标系后做
/// 轴对齐椭圆最小二乘拟合，返回 (cx, cy, rx, ry, theta_rad, rmse)。theta 为局部 x 轴
/// 到主轴(长轴)的逆时针角（弧度）。残差沿用 fit_axis_ellipse 的归一化定义，与
/// ell_tol 保持同一量纲。对完整椭圆（边界点对称）PCA 主轴与椭圆轴精确对齐，θ 无偏。
fn fit_rotated_ellipse(pts: &[(f32, f32)]) -> Option<(f32, f32, f32, f32, f32, f32)> {
    let n = pts.len();
    if n < 6 {
        return None;
    }
    let (mx, my) = centroid(pts);
    // 协方差矩阵 → 主轴角 θ
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
    // tan(2θ) = 2·cxy/(cxx - cyy)；各向同性(如方块)时 cxy≈0 且 cxx≈cyy → θ 退化但 rx≈ry
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

fn circle_subpath(cx: f32, cy: f32, r: f32) -> String {
    format!(
        "M {} {} A {} {} 0 0 1 {} {} A {} {} 0 0 1 {} {} Z",
        cx - r, cy, r, r, cx + r, cy, r, r, cx - r, cy
    )
}

/// 旋转椭圆子路径：用 SVG 弧命令（带 x-axis-rotation）还原旋转椭圆。
/// theta 为局部坐标系 x 轴到主轴(长轴)的逆时针角（弧度）。两端点取主轴端点
/// （局部 φ=0 与 φ=180°），两段各 180° 合成完整椭圆；因两端点是椭圆的对径点，
/// large-arc-flag 取 0/1 结果相同。
fn rotated_ellipse_subpath(cx: f32, cy: f32, rx: f32, ry: f32, theta_rad: f32) -> String {
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

// ---------------- 直线 / 圆弧 分段 ----------------
//
// 把一条简化轮廓切成「直线段(L)」与「真圆弧(A)」交替的子路径。这样：
//   - 对角/斜线（栅格化后是台阶）被合并成单条直线，消除毛刺；
//   - 圆角/倒角（部分圆弧）被识别成 SVG 弧命令，而不是碎成多边形或整条贝塞尔。
// 自由曲线（花瓣/云朵，曲率处处变化）不应进入此分支，由上层改走贝塞尔。

/// 过三点的最小二乘圆拟合（解析解）。返回 ((cx, cy), r)。
fn fit_circle_3pt(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Option<((f64, f64), f64)> {
    let (x1, y1) = (a.0 as f64, a.1 as f64);
    let (x2, y2) = (b.0 as f64, b.1 as f64);
    let (x3, y3) = (c.0 as f64, c.1 as f64);
    let d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if d.abs() < 1e-9 {
        return None;
    }
    let ux = ((x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2))
        / d;
    let uy = ((x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1))
        / d;
    let r = ((x1 - ux).powi(2) + (y1 - uy).powi(2)).sqrt();
    if !r.is_finite() || r <= 0.0 {
        return None;
    }
    Some(((ux, uy), r))
}

#[inline]
fn angle_of(p: (f32, f32), cx: f64, cy: f64) -> f64 {
    (p.1 as f64 - cy).atan2(p.0 as f64 - cx)
}

/// 对一组按环形顺序排列的轮廓点做「单一圆弧」判定：用 apex 附近窗口三点定圆，再校验
/// 全部点落在容差 `tol` 内、绕心角度单调、总转角 ≤ π。成功返回 (cx, cy, r, 总扫角, 最大偏差)。
///
/// 关键：必须在「跨多段」的窗口上拟合，而非仅首段——细采样的小圆角首段近乎直线，三点定圆会得到
/// 巨大（浅）半径而被 `rr>200` 过滤；只有窗口足够长（≥8 点，约 20°+ 弧）才能稳定恢复真实半径。
/// 直边因三点近似共线 → `rr` 远超 200 自然被拒，不必额外判“直”。
fn fit_and_validate_arc(
    run: &[(f32, f32)],
    tol: f64,
    _line_eps: f32,
) -> Option<(f64, f64, f64, f64, f32)> {
    if run.len() < 8 {
        return None; // 窗口太短，三点定圆不稳定，延后到更长窗口再判
    }
    let (pa, pb) = (run[0], run[run.len() - 1]);
    let mut apex = 0usize;
    let mut bmax = -1.0f32;
    for q in 1..run.len() - 1 {
        let d = point_line_dist(run[q], pa, pb);
        if d > bmax {
            bmax = d;
            apex = q;
        }
    }
    // 窗口半宽自适应：小圆角(短运行)自动缩窗口留在弧段内，避免把整角+直边一并拟合导致失败；
    // 大圆则放大到 ~25° 跨度以稳定恢复半径（细采样大圆上过短的窗口近乎直线→半径巨大被拒）。
    let maxw = std::cmp::min(apex, run.len() - 1 - apex);
    let target = std::cmp::max(3usize, (run.len() / 4).min(15));
    let w = std::cmp::min(maxw, target);
    let ((ccx, ccy), rr) = fit_circle_3pt(run[apex - w], run[apex], run[apex + w])?;
    if rr < 1.0 || rr > 200.0 {
        return None; // 半径过大（近直边）或过小，拒掉
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
    if total.abs() > std::f64::consts::PI {
        return None;
    }
    let _ = bmax; // 仅用于 apex 选择，不再作为“直/弧”判据（短窗口 bmax 恒小会误杀真弧）
    Some((ccx, ccy, rr, total, maxd as f32))
}

/// 把闭环切成 直线(L) / 真圆弧(A) 子路径（基于「运行」的贪心分段：弧优先）。
///
/// 设计要点（修复两处旧 bug）：
/// 1. 斜线毛刺：RDP 简化已把斜线台阶压成单条直边（偏差 < eps），本函数对直边运行输出一条 L，
///    把台阶合并，故斜线不再有锯齿状毛刺。
/// 2. 倒角/圆角识别成圆：RDP 保证“每条简化边”本身偏差 < eps，因此逐边判“弯”会失效（每段
///    看起来都直 → 整圈退化为全 L 多边形，倒角出不来）。故改为**跨越多条边的运行级判定**：
///    - 先尝试“弧运行”：从当前顶点起逐步扩大弦跨，只要整段原始点都落在同一圆上（圆拟合良好）
///      就继续延伸；一旦碰到直边（点离开该圆）即停。运行末再校验“圆弧偏差 ≪ 该运行弦的拱高”，
///      成立则输出一段真圆弧 A（否则视为非圆，退化为直/折线）。
///    - 弧不行则尝试“直边运行”：从当前顶点起扩大累计弦，只要全部原始点到该弦偏差 ≤ line_eps
///      就继续；用于吸收台阶/轻微折角。
/// 起点选在一条直边，避免从弧中间起步导致回绕重复处理。圆/椭圆已在上层拦截；星形等尖角多边形
/// 因每段是直线、圆拟合返回共线(None)而走直边运行，保持尖角不磨圆；自由曲线由上层改走贝塞尔。
fn segment_primitives(loop_: &[(f32, f32)], eps: f32) -> String {
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

    // 把每个简化顶点映射回原始轮廓索引（简化点来自原始轮廓，可按值精确查找）。
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

    // 辅助：顶点 a..=b 之间（环形）的原始轮廓点到弦 pts[a]→pts[b] 的最大垂直距离（拱高）。
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
    // 辅助：收集顶点 a..=b 之间的原始轮廓点（含两端弦点）。
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

    // 起点：选第一条直边起步，避免从弧中间开始导致回绕重复处理。
    let mut start_e = 0usize;
    for e in 0..n {
        if chord_max_dev(e, (e + 1) % n) <= line_eps {
            start_e = e;
            break;
        }
    }

    // 每个顶点的有符号转角（弧度），用于判别“尖角”与“平滑弧”。
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
    // 弧运行的端顶点转角上限：超过即视为“尖角”（如方块/星形/平行四边形的 90°+ 折角），
    // 不应识别成圆弧；下限避免把近直线也当弧。
    const T_HI: f64 = 1.25; // ~72°
    // 直边运行在“端顶点转角”超过此值时停止，避免把圆角/倒角的第一小段直边吞掉，
    // 否则留给弧运行的剩余段太小（bow < line_eps）而无法识别成 A。
    const T_TURN_STRAIGHT: f64 = 0.2; // ~11°

    // 生成路径
    let mut s = format!("M {} {}", pts[start_e].0, pts[start_e].1);
    let mut i = start_e;
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > n + 2 {
            break;
        }
        // 1) 先尝试弧运行（弧优先）。关键：参考圆只在运行首段拟合一次并固定，后续所有点都
        //    与该【固定】圆比较（紧绝对容差，不随半径放大）。这样一旦运行越过倒角进入直边，
        //    直边点与参考圆偏差超限即停，杜绝旧版“每步重拟合→大圆吸附整圈”的失控。直边本身
        //    因首段共线而无法建立参考圆，自然止步 → 改走直边运行。
        // 1) 弧运行（弧优先）：从 i 向前扫描，找到「单一圆弧」段。
        //    关键修正：参考圆必须在跨多段（含角点附近整段弧）的窗口上拟合，而非仅首段。
        //    细采样的小圆角首段近乎直线，三点定圆会得到巨大（浅）半径而被过滤，导致整圈倒角
        //    被直边运行吞成台阶状 L。故向前逐步扩大端点，对整段做圆拟合+校验，一旦通过即接受
        //    并继续延伸至最长；越过角点进入直边时校验失败即停在弧尾。
        let nxt0 = (i + 1) % n;
        let mut arc_emitted = false;
        if nxt0 != i && nxt0 != start_e {
            let arc_tol = arc_tol_abs as f64;
            // (cx, cy, r, 终点顶点索引, 总扫角, 最大偏差)
            let mut best_arc: Option<(f64, f64, f64, usize, f64, f32)> = None;
            let mut e = nxt0;
            let mut safety = 0usize;
            loop {
                let pnext = (e + 1) % n;
                if pnext == i || pnext == start_e {
                    break;
                }
                // 尖角（方块/星/平行四边形 90°+ 折角）不是弧，停。
                if turn[pnext].abs() > T_HI {
                    break;
                }
                safety += 1;
                if safety > 220 {
                    break; // 安全上限（覆盖大半圆等长弧）
                }
                let prun = collect_run(i, pnext);
                if let Some((ccx, ccy, rr, total, maxd)) =
                    fit_and_validate_arc(&prun, arc_tol, line_eps)
                {
                    // 整段可拟合为干净圆弧：接受，并继续延伸寻找更长弧。
                    best_arc = Some((ccx, ccy, rr, pnext, total, maxd));
                    e = pnext;
                    continue;
                }
                // 校验失败：可能还在爬升段（尚未抵达弧），或已越过弧进入直边。
                if best_arc.is_some() {
                    break; // 已找到弧，停在弧尾
                }
                // 仍在寻找：若已延伸很多段仍无法成弧，判定此处不是弧。
                if safety >= 20 {
                    break;
                }
                e = pnext;
            }
            if let Some((ccx, ccy, r, e_end, total, maxd)) = best_arc {
                let bow = chord_max_dev(i, e_end);
                // 末校验：确为「曲线」而非「微抖直线」，且总扫角够大（≥~34°）。
                if bow > line_eps && maxd < bow * 0.7 && total.abs() > 0.6 {
                    let end = pts[e_end];
                    // 由「拟合圆心」反推 (large, sweep)，确保 SVG 弧的圆心落在正确一侧：
                    // sub-180° 弧（圆角 / 整圆被切成 1/4 弧）若仅按 total 符号取 sweep，
                    // 圆心会错配到弦的另一侧，多段弧圆心互不重合 → 畸形。半圆(h=0)时两侧重合，
                    // want_plus 取等→sweep=0，与旧版半圆行为一致。
                    let large = if total.abs() > std::f64::consts::PI {
                        1
                    } else {
                        0
                    };
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
                    // SVG：center = M - h·n 当 large==sweep；= M + h·n 当 large!=sweep。
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
                    if i == start_e {
                        break;
                    }
                    arc_emitted = true;
                }
            }
        }
        if arc_emitted {
            continue;
        }
        // 2) 弧不行 → 直边运行：从 i 起扩大累计弦，只要全部原始点偏差 ≤ line_eps 就继续（吸收台阶/轻微折角）。
        let mut j = i;
        loop {
            let nxt = (j + 1) % n;
            if nxt == i || nxt == start_e {
                break;
            }
            if chord_max_dev(i, nxt) <= line_eps && turn[nxt].abs() < T_TURN_STRAIGHT {
                j = nxt;
            } else {
                break;
            }
        }
        if j == i {
            // 没有任何可延伸的直边（退化情形）：强制前进一步，避免卡死
            j = (i + 1) % n;
        }
        let end = pts[j];
        s.push_str(&format!(" L {} {}", end.0, end.1));
        i = j;
        if i == start_e {
            break;
        }
    }
    s.push_str(" Z");
    s
}

/// 统计闭环中“足够长的直边”数量：把简化顶点映射回原始轮廓，检查每条简化边所对应的
/// 原始轮廓点是否都落在该边弦线 `line_eps` 内、且边长 ≥ `min_len`。用于判断轮廓是否为
/// “直线+圆弧”型（方块/星/圆角矩形/八边形）——这类应走 `segment_primitives` 识别倒角，
/// 而非整条贝塞尔曲线。
fn count_straight_edges(
    loop_: &[(f32, f32)],
    simp: &[(f32, f32)],
    line_eps: f32,
    min_len: f32,
) -> usize {
    let n = simp.len();
    if n < 3 {
        return 0;
    }
    let loop_len = loop_.len();
    let mut idx = vec![0usize; n];
    let mut cursor = 0usize;
    for k in 0..n {
        let mut found = None;
        for i in cursor..loop_len {
            if loop_[i] == simp[k] {
                found = Some(i);
                break;
            }
        }
        if found.is_none() {
            for i in 0..cursor {
                if loop_[i] == simp[k] {
                    found = Some(i);
                    break;
                }
            }
        }
        idx[k] = found.unwrap_or(cursor);
        cursor = idx[k] + 1;
    }
    let mut count = 0usize;
    for k in 0..n {
        let a = simp[k];
        let b = simp[(k + 1) % n];
        let ia = idx[k];
        let ib = idx[(k + 1) % n];
        let mut maxd = 0.0f32;
        let mut i = ia;
        loop {
            if i != ia && i != ib {
                let d = point_line_dist(loop_[i], a, b);
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
        let len = ((b.0 - a.0).hypot(b.1 - a.1)) as f32;
        if maxd <= line_eps && len >= min_len {
            count += 1;
        }
    }
    count
}

/// 统计闭环中“明显偏离最佳拟合形状”的边数量（即“扁平边 / 平面边”）。
/// 用于区分【真圆 / 真椭圆】与【多边形 / 圆角矩形】：
///  - 真圆：所有原始点都紧贴拟合圆（最大偏差 ≈ 拟合残差），无扁平边 -> 返回 0。
///  - 八边形 / 方块 / 圆角矩形：存在长直边，这些边整体偏离拟合圆（向内凹），偏差远大于残差 -> 返回 ≥2。
/// `dist(p)` 返回点 p 到理想形状的有符号距离（圆: |p-c|-r；椭圆: 旋转缩放至单位圆后 |·|-1）。
/// `thr` 为“视为扁平边”的偏差阈值，通常取 max(残差*K, 1.0px)。
/// 注意用“几何偏差”而非“边长”判定：圆与曲线 blob 的 RDP 简化边都偏短且近乎直，
/// 仅靠边长/直度无法区分；但多边形/矩形的直边会整体偏离最佳拟合圆，由此可可靠识别。
fn count_flat_sides(
    loop_: &[(f32, f32)],
    simp: &[(f32, f32)],
    dist: impl Fn((f32, f32)) -> f64,
    thr: f64,
) -> usize {
    let n = simp.len();
    if n < 3 {
        return 0;
    }
    let loop_len = loop_.len();
    let mut idx = vec![0usize; n];
    let mut cursor = 0usize;
    for k in 0..n {
        let mut found = None;
        for i in cursor..loop_len {
            if loop_[i] == simp[k] {
                found = Some(i);
                break;
            }
        }
        if found.is_none() {
            for i in 0..cursor {
                if loop_[i] == simp[k] {
                    found = Some(i);
                    break;
                }
            }
        }
        idx[k] = found.unwrap_or(cursor);
        cursor = idx[k] + 1;
    }
    let mut count = 0usize;
    for k in 0..n {
        let ia = idx[k];
        let ib = idx[(k + 1) % n];
        let mut maxd = 0.0f64;
        let mut i = ia;
        loop {
            if i != ia && i != ib {
                let d = dist(loop_[i]).abs();
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
        if maxd > thr {
            count += 1;
        }
    }
    count
}

/// 把一个闭环渲染成 SVG 子路径：圆 → 弧；旋转椭圆（含轴对齐）→ 弧；否则回退。
/// 回退时：若 smooth 且点数较多（轮廓处处弯曲，如花瓣/云朵），用「切线 + 曲率分析」
/// 重建为少数干净三次贝塞尔（C 命令）；否则把轮廓切成「直线(L) + 真圆弧(A)」——这能
/// 消除斜线台阶毛刺、并把圆角/倒角识别成圆弧。直线多边形（方块/星形）保持多边形、不磨角。
fn loop_to_subpath(
    loop_: &[(f32, f32)],
    eps: f32,
    circ_tol: f32,
    ell_tol: f32,
    smooth: bool,
) -> String {
    let line_eps = if eps < 1.0 { 1.0 } else { eps };
    let simp = simplify_polygon(loop_, eps);
    if simp.len() < 3 {
        return String::new();
    }
    // 粗简化轮廓：用于“稳健折角计数”。细简化(eps≈0.5px)对 200px 图标几乎不简化，
    // 抗锯齿台阶被全部保留，导致每个顶点都像“折角”；粗简化(eps≥3px)抹平台阶、
    // 只保留几何特征点，真圆/椭圆/平滑 blob 折角≈0，多边形保留真实折角。
    let simp_c = simplify_polygon(loop_, eps.max(3.0));
    // 角点计数（0.3rad≈17°）：用于贝塞尔门限，区分“平滑 blob”与“多边形”。
    let m = simp_c.len();
    let mut corners = 0usize;
    for k in 0..m {
        let a = simp_c[(k + m - 1) % m];
        let b = simp_c[k];
        let c = simp_c[(k + 1) % m];
        let v1 = (b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64);
        let v2 = (c.0 as f64 - b.0 as f64, c.1 as f64 - b.1 as f64);
        let cross = v1.0 * v2.1 - v1.1 * v2.0;
        let dot = v1.0 * v2.0 + v1.1 * v2.1;
        if cross.atan2(dot).abs() > 0.3 {
            corners += 1;
        }
    }
    // 噪声稳健折角计数（0.5rad≈29°）：粗简化后真多边形几何折角均 >0.5rad，平滑
    // 圆/椭圆即便有残余台阶也 <0.5rad。与圆/椭圆拟合残差联合，可靠区分
    // “真圆(res/r≈0.004, 折角≈0)”与“八边形(res/r≈0.024, 折角=8)”。
    let mut corners_robust = 0usize;
    for k in 0..m {
        let a = simp_c[(k + m - 1) % m];
        let b = simp_c[k];
        let c = simp_c[(k + 1) % m];
        let v1 = (b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64);
        let v2 = (c.0 as f64 - b.0 as f64, c.1 as f64 - b.1 as f64);
        let cross = v1.0 * v2.1 - v1.1 * v2.0;
        let dot = v1.0 * v2.0 + v1.1 * v2.1;
        if cross.atan2(dot).abs() > 0.5 {
            corners_robust += 1;
        }
    }
    // 圆/椭圆判定：核心判据是“紧密拟合”(res/r < circ_tol / ell_tol)，噪声稳健折角计数
    // 仅作为第二道保险挡掉多边形。八边形 res/r≈0.024 接近圆但仍能凭 8 个大折角被挡掉；
    // 真圆 res/r≈0.004~0.011、伪折角 <0.5rad → 通过，输出 2 段干净弧而非被碎成多段。
    if loop_.len() >= 5 {
        // 先试圆。判定：紧密拟合(res/r < circ_tol) 且不是清晰多边形。
        // 清晰的几何多边形由“噪声稳健折角计数”挡掉；但对 res/r 极小(<0.013)的轮廓，
        // 它已是铁证的圆/近圆（八边形 res/r≈0.024 远高于此），即便抗锯齿噪声把折角
        // 计数顶到 ≥3 也直接判圆，避免小圆/内孔被碎成多段弧。
        if let Some((cx, cy, r, res)) = fit_circle(loop_) {
            let rr = (res as f64) / (r as f64);
            if r > 0.5 && rr < circ_tol as f64 && (corners_robust < 3 || rr < 0.013) {
                return circle_subpath(cx, cy, r);
            }
        }
        // 再试椭圆（含旋转）：PCA 估计主轴方向后做轴对齐椭圆拟合，命中则输出真椭圆弧
        // （带 x-axis-rotation）。
        if let Some((ecx, ecy, erx, ery, etheta, eres)) = fit_rotated_ellipse(loop_) {
            if erx > 0.5
                && ery > 0.5
                && eres < ell_tol
                && (erx / ery) < 12.0
                && (ery / erx) < 12.0
                && corners_robust < 3
            {
                return rotated_ellipse_subpath(ecx, ecy, erx, ery, etheta);
            }
        }
    }
    // 贝塞尔曲线检测：仅对“无折角、处处弯曲”的自由曲线（花瓣/云朵）生效。
    // 多边形/圆角矩形有折角(corners>=3) -> 跳过，交给 segment_primitives
    // （直边保持为 L、倒角识别为 A；多边形保持全 L）。
    if smooth && simp.len() >= 8 && corners < 3 {
        let coarse = simplify_polygon(loop_, eps.max(3.0));
        if coarse.len() > 12 {
            return curvature_bezier_path(&simp);
        }
    }
    segment_primitives(loop_, eps)
}

/// 闭合 Catmull-Rom 样条 → 三次贝塞尔路径（tension=1 标准系数）。
/// 经过所有控制点，在尖角处可能轻微过冲；因此只对曲线型轮廓使用（见 loop_to_subpath）。
/// 现主要作为曲率重建 `curvature_bezier_path` 在噪声轮廓下的兜底。
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

// ---------------- 曲率驱动的三次贝塞尔重建 ----------------
//
// 思路：沿轮廓算每点的「切线」（中心差分单位向量）与「有符号曲率」（顶点处转角 / 段长），
// 据此把闭环在曲率局部极值 / 拐点 / 折角处切成若干段（放置结点）；然后用这些结点作为
// 控制点做闭合 Catmull-Rom，段内曲率单调、CR 用相邻结点估计切线，得到与曲率特征点数量
// 相当的少数几条干净三次贝塞尔（而非逐点插值或过度细分）。直线多边形不会进入此分支
// （coarse ≤ 12，已走 L 多边形），因此不会被磨圆。

/// 用「轮廓点 + 切线 + 曲率分析」重建闭合曲线的三次贝塞尔路径。
/// 步骤：算每点中心差分切线与有符号曲率；平滑曲率抑制像素台阶噪声；在平滑后曲率的
/// 局部极值 / 拐点 / 折角处放置结点（要求最小角间距、上限 24 个，避免噪声产生密集结点）；
/// 结点作为控制点做闭合 Catmull-Rom，得到与曲率特征数量相当的少数干净贝塞尔。若曲线
/// 仍极曲折（结点数超限）则退化为较粗 RDP 简化 + Catmull-Rom。直线多边形不进入此分支
/// （coarse ≤ 12，已走 L 多边形），因此不会被磨圆。
fn curvature_bezier_path(pts: &[(f32, f32)]) -> String {
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
