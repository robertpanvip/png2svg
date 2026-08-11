//! Step 3 的数据准备：对每个区域拟合颜色场，判定纯色 / 渐变，并细分渐变类型。
//!
//! 对每个通道独立拟合颜色场，支持三种模型：
//!   * 平面（线性渐变）：   color = a + b·x + c·y
//!   * 径向（径向渐变）：   color = a' + k·r，   r = 到质心的距离
//!   * 双线性（网格渐变）： color = a + b·x + c·y + d·x·y（四角插值 / Coons 网格）
//!
//! 用最小二乘分别解三种模型，比较各自解释方差 R²：
//!   - 径向 R² 高（≈1）且不低于平面 → 【径向渐变】；
//!   - 双线性 R² 高、且明显优于平面（xy 项起作用，单一方向无法表示）→ 【网格渐变】；
//!   - 否则 → 【线性渐变】（平面投影即可完整表示）。
//! 用“整片颜色变化量 = 梯度幅值 × 投影跨度”而不是“每像素梯度幅值”判定是否为渐变，
//! 才能正确识别跨度大但每像素变化平缓的真实渐变。

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GradKind {
    Linear,
    Radial,
    Mesh,
}

/// 网格（双线性）渐变系数：region 包围盒 + 每通道 color = a + b·x + c·y + d·x·y。
pub struct MeshCoef {
    pub minx: f32,
    pub miny: f32,
    pub maxx: f32,
    pub maxy: f32,
    /// 每通道系数 [a, b, c, d]，用 f64 保持精度（x 可达数百，b 很小）
    pub r: [f64; 4],
    pub g: [f64; 4],
    pub b: [f64; 4],
}

impl MeshCoef {
    /// 在 (x, y) 处用双线性函数求颜色（RGB 各通道独立）。
    pub fn color_at(&self, x: f64, y: f64) -> (u8, u8, u8) {
        fn f(v: f64) -> u8 {
            v.clamp(0.0, 255.0).round() as u8
        }
        let cr = self.r[0] + self.r[1] * x + self.r[2] * y + self.r[3] * x * y;
        let cg = self.g[0] + self.g[1] * x + self.g[2] * y + self.g[3] * x * y;
        let cb = self.b[0] + self.b[1] * x + self.b[2] * y + self.b[3] * x * y;
        (f(cr), f(cg), f(cb))
    }
}

pub struct Fit {
    pub is_gradient: bool,
    /// 渐变细分类型（仅当 is_gradient=true 时有意义）
    pub grad_kind: GradKind,
    /// 区域均值色（纯色时使用）
    pub mean: (u8, u8, u8),
    /// 线性渐变起止点（像素坐标，userSpaceOnUse 用）
    pub start_pt: (f32, f32),
    pub end_pt: (f32, f32),
    /// 线性渐变起止颜色
    pub start_color: (u8, u8, u8),
    pub end_color: (u8, u8, u8),
    /// 径向渐变：中心与半径（像素坐标）
    pub center: (f32, f32),
    pub radius: f32,
    /// 径向渐变起止颜色（中心色 / 边缘色）
    pub rad_center: (u8, u8, u8),
    pub rad_rim: (u8, u8, u8),
    /// 网格渐变：双线性系数（仅 Mesh 时有值）
    pub mesh: Option<MeshCoef>,
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

/// 解 3×3 线性方程组 A·x = v（高斯消元 + 部分主元），失败（奇异）返回 None。
fn solve3(m: [[f64; 3]; 3], v: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = m;
    let mut b = v;
    for col in 0..3 {
        // 选主元
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..3 {
            let val = a[row][col].abs();
            if val > best {
                best = val;
                piv = row;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            a.swap(piv, col);
            b.swap(piv, col);
        }
        // 消元
        let d = a[col][col];
        for row in (col + 1)..3 {
            let f = a[row][col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    // 回代
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for k in (i + 1)..3 {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

pub fn fit(samples: &[(f32, f32, u8, u8, u8, u8)], p: &FitParams) -> Fit {
    let n = samples.len() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mr = 0.0;
    let mut mg = 0.0;
    let mut mb = 0.0;
    for &(x, y, r, g, b, _a) in samples {
        mx += x as f64;
        my += y as f64;
        mr += r as f64;
        mg += g as f64;
        mb += b as f64;
    }
    mx /= n;
    my /= n;
    mr /= n;
    mg /= n;
    mb /= n;

    // 平面（线性）与双线性所需的二阶/三阶矩与 RHS
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut sx2y = 0.0; // Σ x'² y'
    let mut sxy2 = 0.0; // Σ x' y'²
    let mut sx2y2 = 0.0; // Σ x'² y'²

    // 每通道：平面 RHS / 总方差
    let mut srx = 0.0;
    let mut sry = 0.0;
    let mut tr = 0.0;
    let mut sgx = 0.0;
    let mut sgy = 0.0;
    let mut tg = 0.0;
    let mut sbx = 0.0;
    let mut sby = 0.0;
    let mut tb = 0.0;
    // 每通道：双线性 RHS（Σ x'·dc, Σ y'·dc, Σ x'y'·dc）
    let mut srx2y_r = 0.0;
    let mut sry2_r = 0.0;
    let mut srxy_r = 0.0;
    let mut srx2y_g = 0.0;
    let mut sry2_g = 0.0;
    let mut srxy_g = 0.0;
    let mut srx2y_b = 0.0;
    let mut sry2_b = 0.0;
    let mut srxy_b = 0.0;

    // 包围盒（网格渲染用）+ 各通道颜色极值（方向无关的“整片颜色变化量”门限）
    let mut minx = f64::MAX;
    let mut miny = f64::MAX;
    let mut maxx = f64::MIN;
    let mut maxy = f64::MIN;
    let mut rmin = 255.0f64;
    let mut rmax = 0.0f64;
    let mut gmin = 255.0f64;
    let mut gmax = 0.0f64;
    let mut bmin = 255.0f64;
    let mut bmax = 0.0f64;

    for &(x, y, r, g, b, _a) in samples {
        let xp = x as f64 - mx;
        let yp = y as f64 - my;
        let dr = r as f64 - mr;
        let dg = g as f64 - mg;
        let db = b as f64 - mb;
        sxx += xp * xp;
        syy += yp * yp;
        sxy += xp * yp;
        sx2y += xp * xp * yp;
        sxy2 += xp * yp * yp;
        sx2y2 += xp * xp * yp * yp;

        srx += xp * dr;
        sry += yp * dr;
        tr += dr * dr;
        sgx += xp * dg;
        sgy += yp * dg;
        tg += dg * dg;
        sbx += xp * db;
        sby += yp * db;
        tb += db * db;

        srx2y_r += xp * xp * yp * dr;
        sry2_r += xp * yp * yp * dr;
        srxy_r += xp * yp * dr;
        srx2y_g += xp * xp * yp * dg;
        sry2_g += xp * yp * yp * dg;
        srxy_g += xp * yp * dg;
        srx2y_b += xp * xp * yp * db;
        sry2_b += xp * yp * yp * db;
        srxy_b += xp * yp * db;

        if (x as f64) < minx {
            minx = x as f64;
        }
        if (y as f64) < miny {
            miny = y as f64;
        }
        if (x as f64) > maxx {
            maxx = x as f64;
        }
        if (y as f64) > maxy {
            maxy = y as f64;
        }
        if (r as f64) < rmin {
            rmin = r as f64;
        }
        if (r as f64) > rmax {
            rmax = r as f64;
        }
        if (g as f64) < gmin {
            gmin = g as f64;
        }
        if (g as f64) > gmax {
            gmax = g as f64;
        }
        if (b as f64) < bmin {
            bmin = b as f64;
        }
        if (b as f64) > bmax {
            bmax = b as f64;
        }
    }

    // ---------- 平面（线性）拟合 ----------
    let det = sxx * syy - sxy * sxy;
    let (br, bc, gr, gc, bb, bc2) = if det > 1e-12 {
        (
            (srx * syy - sry * sxy) / det,
            (sxx * sry - sxy * srx) / det,
            (sgx * syy - sgy * sxy) / det,
            (sxx * sgy - sxy * sgx) / det,
            (sbx * syy - sby * sxy) / det,
            (sxx * sby - sxy * sbx) / det,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    fn r2_of(b: f64, c: f64, sxx: f64, syy: f64, sxy: f64, total: f64) -> f64 {
        if total <= 1e-12 {
            return 1.0;
        }
        let explained = b * b * sxx + 2.0 * b * c * sxy + c * c * syy;
        (1.0 - (total - explained) / total).clamp(0.0, 1.0)
    }

    let rr2 = r2_of(br, bc, sxx, syy, sxy, tr);
    let rg2 = r2_of(gr, gc, sxx, syy, sxy, tg);
    let rb2 = r2_of(bb, bc2, sxx, syy, sxy, tb);
    let linear_r2 = (rr2 + rg2 + rb2) / 3.0;

    // 亮度方向的梯度（Rec.601），定义线性渐变主轴
    let lux = 0.299 * br + 0.587 * gr + 0.114 * bb;
    let luy = 0.299 * bc + 0.587 * gc + 0.114 * bc2;
    let lumag = (lux * lux + luy * luy).sqrt();
    let (ux, uy) = if lumag > 1e-9 {
        (lux / lumag, luy / lumag)
    } else {
        (1.0, 0.0)
    };
    // 沿主轴求投影范围 → 整片颜色变化量
    let mut min_p = f64::MAX;
    let mut max_p = f64::MIN;
    let mut start_color = (mr as u8, mg as u8, mb as u8);
    let mut end_color = (mr as u8, mg as u8, mb as u8);
    for &(x, y, r, g, b, _a) in samples {
        let dx = x as f64 - mx;
        let dy = y as f64 - my;
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

    // ---------- 双线性（网格）拟合 ----------
    let m_bi = [
        [sxx, sxy, sx2y],
        [sxy, syy, sxy2],
        [sx2y, sxy2, sx2y2],
    ];
    let (bi_r, bi_g, bi_b) = (
        solve3(m_bi, [srx, sry, srxy_r]),
        solve3(m_bi, [sgx, sgy, srxy_g]),
        solve3(m_bi, [sbx, sby, srxy_b]),
    );
    let mut bilinear_r2 = 0.0;
    let mut mesh: Option<MeshCoef> = None;
    if let (Some(bi_r), Some(bi_g), Some(bi_b)) = (bi_r, bi_g, bi_b) {
        // 解释方差：Σ(a+b x'+c y'+d x'y')² = b²Σx'² + c²Σy'² + d²Σx'²y'²
        //   + 2bc Σx'y' + 2bd Σx'²y' + 2cd Σx'y'²
        fn r2_bi(
            co: [f64; 3],
            sxx: f64,
            syy: f64,
            sxy: f64,
            sx2y: f64,
            sxy2: f64,
            sx2y2: f64,
            total: f64,
        ) -> f64 {
            if total <= 1e-12 {
                return 1.0;
            }
            let (b, c, d) = (co[0], co[1], co[2]);
            let explained = b * b * sxx
                + c * c * syy
                + d * d * sx2y2
                + 2.0 * b * c * sxy
                + 2.0 * b * d * sx2y
                + 2.0 * c * d * sxy2;
            (1.0 - (total - explained) / total).clamp(0.0, 1.0)
        }
        let rr2b = r2_bi(bi_r, sxx, syy, sxy, sx2y, sxy2, sx2y2, tr);
        let rg2b = r2_bi(bi_g, sxx, syy, sxy, sx2y, sxy2, sx2y2, tg);
        let rb2b = r2_bi(bi_b, sxx, syy, sxy, sx2y, sxy2, sx2y2, tb);
        bilinear_r2 = (rr2b + rg2b + rb2b) / 3.0;
        // 双线性拟合在“中心化坐标 x'=x-mx, y'=y-my”下进行，得到
        //   color = a + b·x' + c·y' + d·x'·y'（a = 均值）。
        // 但 color_at 接收的是绝对坐标，故在此把系数改写成绝对形式
        //   color = A + B·x + C·y + D·x·y
        // 这样 color_at 直接代入绝对 (x,y) 即得正确颜色。
        fn to_abs(a_c: f64, bi: [f64; 3], cx: f64, cy: f64) -> [f64; 4] {
            let (b, c, d) = (bi[0], bi[1], bi[2]);
            let a = a_c - b * cx - c * cy + d * cx * cy;
            let bb = b - d * cy;
            let cc = c - d * cx;
            [a, bb, cc, d]
        }
        let (cx, cy) = (mx, my);
        mesh = Some(MeshCoef {
            minx: minx as f32,
            miny: miny as f32,
            maxx: maxx as f32,
            maxy: maxy as f32,
            r: to_abs(mr, bi_r, cx, cy),
            g: to_abs(mg, bi_g, cx, cy),
            b: to_abs(mb, bi_b, cx, cy),
        });
    }

    // 线性渐变主轴端点（供径向候选中心使用）
    let start_pt = ((mx + ux * min_p) as f32, (my + uy * min_p) as f32);
    let end_pt = ((mx + ux * max_p) as f32, (my + uy * max_p) as f32);

    // ---------- 径向拟合（中心在多个候选点中择优） ----------
    // 真实径向渐变的圆心不一定在质心（如 t_radial2 圆心在左上角 0.35,0.35）。若固定
    // 中心=质心，径向 R² 远低于线性 → 误判为线性。改为在候选中心中挑选使径向 R² 最高者：
    // 质心、包围盒中心、线性两端点、最亮/最暗像素（径向停止点常在极值色处）、包围盒 6×6 网格。
    let mut radial_r2 = 0.0;
    let mut rad_center = (mr as u8, mg as u8, mb as u8);
    let mut rad_rim = (mr as u8, mg as u8, mb as u8);
    let mut rad_center_pt = (mx as f32, my as f32);
    let mut radius = 0.0f64;

    // 最亮 / 最暗像素位置
    let mut best_lum = f64::MIN;
    let mut worst_lum = f64::MAX;
    let mut bright_pt = (mx, my);
    let mut dark_pt = (mx, my);
    for &(x, y, r, g, b, _a) in samples {
        let lum = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
        if lum > best_lum {
            best_lum = lum;
            bright_pt = (x as f64, y as f64);
        }
        if lum < worst_lum {
            worst_lum = lum;
            dark_pt = (x as f64, y as f64);
        }
    }

    let mut cand: Vec<(f64, f64)> = vec![
        (mx, my),
        ((minx + maxx) * 0.5, (miny + maxy) * 0.5),
        (start_pt.0 as f64, start_pt.1 as f64),
        (end_pt.0 as f64, end_pt.1 as f64),
        bright_pt,
        dark_pt,
    ];
    // 包围盒 6×6 网格
    for gi in 0..6 {
        for gj in 0..6 {
            let cx = minx + (maxx - minx) * (gi as f64) / 5.0;
            let cy = miny + (maxy - miny) * (gj as f64) / 5.0;
            cand.push((cx, cy));
        }
    }
    cand.sort_by(|a, b| {
        (a.0, a.1)
            .partial_cmp(&(b.0, b.1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    cand.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);

    // 单中心径向拟合：color = a' + k·r（r = 到候选中心距离）
    fn fit_radial_ch(
        n: f64,
        sr: f64,
        sr2: f64,
        det_r: f64,
        r1: f64,
        rr: f64,
        total: f64,
    ) -> (f64, f64, f64) {
        let ap = (sr2 * r1 - sr * rr) / det_r;
        let k = (n * rr - sr * r1) / det_r;
        let ss_reg = ap * ap * n + 2.0 * ap * k * sr + k * k * sr2;
        let ss_res = (total - ss_reg).max(0.0);
        let r2 = if total <= 1e-12 {
            1.0
        } else {
            (1.0 - ss_res / total).clamp(0.0, 1.0)
        };
        (ap, k, r2)
    }

    for (ccx, ccy) in cand {
        let mut sr = 0.0f64;
        let mut sr2 = 0.0f64;
        let mut sr1_r = 0.0f64;
        let mut srr_r = 0.0f64;
        let mut sr1_g = 0.0f64;
        let mut srr_g = 0.0f64;
        let mut sr1_b = 0.0f64;
        let mut srr_b = 0.0f64;
        let mut rmax = 0.0f64;
        for &(x, y, r, g, b, _a) in samples {
            let rp = ((x as f64 - ccx).powi(2) + (y as f64 - ccy).powi(2)).sqrt();
            sr += rp;
            sr2 += rp * rp;
            let dr = r as f64 - mr;
            let dg = g as f64 - mg;
            let db = b as f64 - mb;
            sr1_r += dr;
            srr_r += rp * dr;
            sr1_g += dg;
            srr_g += rp * dg;
            sr1_b += db;
            srr_b += rp * db;
            if rp > rmax {
                rmax = rp;
            }
        }
        let det_r = n * sr2 - sr * sr;
        if det_r.abs() <= 1e-9 {
            continue;
        }
        let (ar, kr, rr2rad) = fit_radial_ch(n, sr, sr2, det_r, sr1_r, srr_r, tr);
        let (ag, kg, rg2rad) = fit_radial_ch(n, sr, sr2, det_r, sr1_g, srr_g, tg);
        let (ab, kb, rb2rad) = fit_radial_ch(n, sr, sr2, det_r, sr1_b, srr_b, tb);
        let cand_r2 = (rr2rad + rg2rad + rb2rad) / 3.0;
        if cand_r2 > radial_r2 {
            radial_r2 = cand_r2;
            rad_center_pt = (ccx as f32, ccy as f32);
            radius = rmax;
            let cc = (
                (mr + ar).clamp(0.0, 255.0),
                (mg + ag).clamp(0.0, 255.0),
                (mb + ab).clamp(0.0, 255.0),
            );
            let rc = (
                (mr + ar + kr * rmax).clamp(0.0, 255.0),
                (mg + ag + kg * rmax).clamp(0.0, 255.0),
                (mb + ab + kb * rmax).clamp(0.0, 255.0),
            );
            rad_center = (cc.0 as u8, cc.1 as u8, cc.2 as u8);
            rad_rim = (rc.0 as u8, rc.1 as u8, rc.2 as u8);
        }
    }

    // ---------- 类型判定 ----------
    let best_r2 = linear_r2.max(radial_r2).max(bilinear_r2);
    // 方向无关的颜色变化量（径向/对称渐变在单一线性投影上变化量≈0，必须用通道极差）
    let color_range = (rmax - rmin).max(gmax - gmin).max(bmax - bmin);
    let is_gradient = best_r2 > p.r2_thresh as f64 && color_range > p.min_var as f64;

    let grad_kind = if !is_gradient {
        GradKind::Linear
    } else if radial_r2 >= 0.85 && radial_r2 >= linear_r2 - 0.02 {
        GradKind::Radial
    } else if bilinear_r2 >= 0.95 && (bilinear_r2 - linear_r2) >= 0.02 {
        GradKind::Mesh
    } else {
        GradKind::Linear
    };

    Fit {
        is_gradient,
        grad_kind,
        mean: (mr as u8, mg as u8, mb as u8),
        start_pt,
        end_pt,
        start_color,
        end_color,
        center: rad_center_pt,
        radius: radius as f32,
        rad_center,
        rad_rim,
        mesh,
    }
}
