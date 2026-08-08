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
pub fn loops_to_path(loops: &[Vec<(f32, f32)>], eps: f32) -> String {
    let mut s = String::new();
    for lp in loops {
        let simp = simplify_polygon(lp, eps);
        if simp.len() < 3 {
            continue;
        }
        s.push_str(&format!("M {} {}", simp[0].0, simp[0].1));
        for p in &simp[1..] {
            s.push_str(&format!(" L {} {}", p.0, p.1));
        }
        s.push_str(" Z");
    }
    s
}
