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
/// 每个闭环会先尝试拟合成【圆】或【轴对齐椭圆】，命中则用弧命令（真圆/真椭圆）；
/// 否则回退：若允许平滑且点数较多（曲线型轮廓），用 Catmull-Rom 样条生成平滑
/// 密集曲线（C 命令）；否则保留多边形（L 命令）。弧/曲线/多边形可共存于同一 path，
/// 故孔洞(evenodd)仍成立。
pub fn loops_to_path(
    loops: &[Vec<(f32, f32)>],
    eps: f32,
    circ_tol: f32,
    ell_tol: f32,
    smooth: bool,
) -> String {
    let mut s = String::new();
    for lp in loops {
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

fn ellipse_subpath(cx: f32, cy: f32, rx: f32, ry: f32) -> String {
    format!(
        "M {} {} A {} {} 0 0 1 {} {} A {} {} 0 0 1 {} {} Z",
        cx - rx, cy, rx, ry, cx + rx, cy, rx, ry, cx - rx, cy
    )
}

/// 把一个闭环渲染成 SVG 子路径：圆 → 弧；轴对齐椭圆 → 弧；否则回退。
/// 回退时：若 smooth 且点数较多，则用 Catmull-Rom 平滑曲线（高密度、平滑）；
/// 否则保留多边形。
fn loop_to_subpath(
    loop_: &[(f32, f32)],
    eps: f32,
    circ_tol: f32,
    ell_tol: f32,
    smooth: bool,
) -> String {
    if loop_.len() >= 5 {
        // 先试圆
        if let Some((cx, cy, r, res)) = fit_circle(loop_) {
            if r > 0.5 && res / r < circ_tol {
                return circle_subpath(cx, cy, r);
            }
        }
        // 再试椭圆：用质心作为椭圆中心（细长椭圆的圆拟合中心会偏移，不可信）
        let (mx, my) = centroid(loop_);
        if let Some((_, _, rx, ry, eres)) = fit_axis_ellipse(loop_, mx, my) {
            if rx > 0.5
                && ry > 0.5
                && eres < ell_tol
                && (rx / ry) < 12.0
                && (ry / rx) < 12.0
            {
                return ellipse_subpath(mx as f32, my as f32, rx, ry);
            }
        }
    }
    // 回退：先 RDP 简化（保留细节用于最终曲线）
    let simp = simplify_polygon(loop_, eps);
    if simp.len() < 3 {
        return String::new();
    }
    // 贝塞尔曲线检测：粗简化（压平像素台阶）后，若轮廓仍需很多段才能表示（>12），
    // 说明它是处处弯曲的曲线（圆/椭圆/花瓣/云朵等），输出密集三次贝塞尔；
    // 直线多边形（方块/星形/正多边形）粗化后只剩少数边（≤12），保持多边形、不磨角。
    if smooth && simp.len() >= 8 {
        // 贝塞尔曲线检测：粗简化（压平像素台阶）后，若轮廓仍需很多段才能表示（>12），
        // 说明它是处处弯曲的曲线（圆/椭圆/花瓣/云朵等），输出密集三次贝塞尔；
        // 直线多边形（方块/星形/正多边形）粗化后只剩少数边（≤12），保持多边形、不磨角。
        let coarse = simplify_polygon(loop_, eps.max(3.0));
        if coarse.len() > 12 {
            return catmull_rom_path(&simp);
        }
    }
    let mut s = format!("M {} {}", simp[0].0, simp[0].1);
    for p in &simp[1..] {
        s.push_str(&format!(" L {} {}", p.0, p.1));
    }
    s.push_str(" Z");
    s
}

/// 闭合 Catmull-Rom 样条 → 三次贝塞尔路径（tension=1 标准系数）。
/// 经过所有控制点，在尖角处可能轻微过冲；因此只对曲线型轮廓使用（见 loop_to_subpath）。
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
