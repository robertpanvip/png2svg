//! Step 3 的数据准备：对每个区域拟合颜色平面，判定纯色 / 渐变。
//!
//! 对每个通道独立拟合平面：color = a + b·x + c·y
//! 用去中心化坐标的最小二乘解 2×2 线性方程组：
//!     [Σx²   Σxy][b]   [Σx·dc]
//!     [Σxy   Σy²][c] = [Σy·dc]
//! 其中 dc = color - mean(color)。
//!
//! 判定：
//!   - 平均解释方差 R² 高，且“整片区域的颜色变化量”足够大 -> 【渐变】；
//!   - 否则 -> 【纯色】（取均值色）。
//! 用“整片颜色变化量 = 梯度幅值 × 投影跨度”而非“每像素梯度幅值”，
//! 才能正确识别跨度大但每像素变化平缓的真实渐变。

pub struct Fit {
    pub is_gradient: bool,
    /// 区域均值色（纯色时使用）
    pub mean: (u8, u8, u8),
    /// 渐变起止点（像素坐标，userSpaceOnUse 用）
    pub start_pt: (f32, f32),
    pub end_pt: (f32, f32),
    /// 渐变起止颜色
    pub start_color: (u8, u8, u8),
    pub end_color: (u8, u8, u8),
}

pub struct FitParams {
    /// 判定为渐变所需的最小“整片颜色变化量”（亮度近似）
    pub min_var: f32,
    /// 判定为渐变所需的最小平均解释方差 R²
    pub r2_thresh: f32,
}

impl Default for FitParams {
    fn default() -> Self {
        FitParams {
            min_var: 8.0,
            r2_thresh: 0.6,
        }
    }
}

pub fn fit(samples: &[(f32, f32, u8, u8, u8, u8)], p: &FitParams) -> Fit {
    let n = samples.len() as f32;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mr = 0.0;
    let mut mg = 0.0;
    let mut mb = 0.0;
    for &(x, y, r, g, b, _a) in samples {
        mx += x;
        my += y;
        mr += r as f32;
        mg += g as f32;
        mb += b as f32;
    }
    mx /= n;
    my /= n;
    mr /= n;
    mg /= n;
    mb /= n;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    // 各通道：Σ dx·dc, Σ dy·dc, Σ dc²
    let mut srx = 0.0;
    let mut sry = 0.0;
    let mut tr = 0.0;
    let mut sgx = 0.0;
    let mut sgy = 0.0;
    let mut tg = 0.0;
    let mut sbx = 0.0;
    let mut sby = 0.0;
    let mut tb = 0.0;

    for &(x, y, r, g, b, _a) in samples {
        let dx = x - mx;
        let dy = y - my;
        let dr = r as f32 - mr;
        let dg = g as f32 - mg;
        let db = b as f32 - mb;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        srx += dx * dr;
        sry += dy * dr;
        tr += dr * dr;
        sgx += dx * dg;
        sgy += dy * dg;
        tg += dg * dg;
        sbx += dx * db;
        sby += dy * db;
        tb += db * db;
    }

    let det = sxx * syy - sxy * sxy;
    let (br, bc, gr, gc, bb, bc2) = if det > 1e-9 {
        let br = (srx * syy - sry * sxy) / det;
        let bc = (sxx * sry - sxy * srx) / det;
        let gr = (sgx * syy - sgy * sxy) / det;
        let gc = (sxx * sgy - sxy * sgx) / det;
        let bb = (sbx * syy - sby * sxy) / det;
        let bc2 = (sxx * sby - sxy * sbx) / det;
        (br, bc, gr, gc, bb, bc2)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    // 解释方差 R² = 1 - 残差/总方差，残差 = 总方差 - 拟合解释量
    fn r2_of(b: f32, c: f32, sxx: f32, syy: f32, sxy: f32, sr: f32, sc: f32, total: f32) -> f32 {
        if total <= 1e-9 {
            return 1.0;
        }
        let explained = b * b * sxx + 2.0 * b * c * sxy + c * c * syy;
        (1.0 - (total - explained) / total).clamp(0.0, 1.0)
    }

    let rr2 = r2_of(br, bc, sxx, syy, sxy, srx, sry, tr);
    let rg2 = r2_of(gr, gc, sxx, syy, sxy, sgx, sgy, tg);
    let rb2 = r2_of(bb, bc2, sxx, syy, sxy, sbx, sby, tb);
    let mean_r2 = (rr2 + rg2 + rb2) / 3.0;

    // 亮度方向的梯度（用 Rec.601 权重），用来定义渐变主轴
    let lux = 0.299 * br + 0.587 * gr + 0.114 * bb;
    let luy = 0.299 * bc + 0.587 * gc + 0.114 * bc2;
    let lumag = (lux * lux + luy * luy).sqrt();

    let (ux, uy) = if lumag > 1e-9 {
        (lux / lumag, luy / lumag)
    } else {
        (1.0, 0.0)
    };
    // 沿渐变主轴求投影范围，得到“整片区域的颜色变化量”
    let mut min_p = f32::MAX;
    let mut max_p = f32::MIN;
    let mut start_color = (mr as u8, mg as u8, mb as u8);
    let mut end_color = (mr as u8, mg as u8, mb as u8);
    for &(x, y, r, g, b, _a) in samples {
        let dx = x - mx;
        let dy = y - my;
        let proj = ux * dx + uy * dy;
        if proj < min_p {
            min_p = proj;
            start_color = (r, g, b);
        }
        if proj > max_p {
            max_p = proj;
            end_color = (r, g, b);
        }
    }
    let proj_range = (max_p - min_p).max(0.0);
    // 颜色变化量 = 梯度幅值 × 投影跨度（亮度近似）
    let color_var = lumag * proj_range;

    // 判定：R² 高 且 整片颜色有明显变化 -> 渐变；否则纯色
    let is_gradient = mean_r2 > p.r2_thresh && color_var > p.min_var;

    let start_pt = (mx + ux * min_p, my + uy * min_p);
    let end_pt = (mx + ux * max_p, my + uy * max_p);

    let mean = (mr as u8, mg as u8, mb as u8);
    Fit {
        is_gradient,
        mean,
        start_pt,
        end_pt,
        start_color,
        end_color,
    }
}
