//! 多边形简化（RDP）。把抗锯齿后的密集轮廓点压成少量顶点，缩小 SVG 体积、
//! 抹平台阶噪点，为后续图元拟合（圆/椭圆/弧）提供干净输入。

/// 点到线段 (a,b) 的垂直距离。
pub(crate) fn point_line_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
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
