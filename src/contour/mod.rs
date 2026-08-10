//! 轮廓提取与矢量化（把 PNG 前景掩码转成 SVG path）。
//!
//! 子模块划分：
//!   - trace.rs       Moore 邻域边界跟踪、孔洞识别、连通分量提取
//!   - simplify.rs    RDP 多边形简化
//!   - fit.rs         圆 / 椭圆拟合、弧校验（Kåsa/Pratt 最小二乘）
//!   - primitives.rs  直边 / 真圆弧分段（segment_primitives）
//!   - bezier.rs      曲率驱动的三次贝塞尔曲线重建
//!
//! 每个形状输出「外边界 + 内边界(孔洞)」，用 fill-rule=evenodd 形成真洞。

mod bezier;
mod fit;
mod primitives;
mod simplify;
mod trace;

pub use trace::four_neighbors;
pub use trace::trace_with_holes;

use bezier::curvature_bezier_path;
use fit::{fit_circle, fit_rotated_ellipse};
use primitives::{circle_subpath, rotated_ellipse_subpath, segment_primitives};
use simplify::simplify_polygon;

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
        // （带 x-axis-rotation）。门限同圆：核心看紧密拟合(eres<ell_tol)，corners_robust<3
        // 作第二保险挡多边形；但 eres<0.015 是铁证椭圆（八边形 eres≈0.024 远高于此），
        // 即便 AA 噪声把粗简化折角顶到 ≥3 也直接判椭圆，避免椭圆被碎成多段弧。
        if let Some((ecx, ecy, erx, ery, etheta, eres)) = fit_rotated_ellipse(loop_) {
            if erx > 0.5
                && ery > 0.5
                && eres < ell_tol
                && (erx / ery) < 12.0
                && (ery / erx) < 12.0
                && (corners_robust < 3 || eres < 0.015)
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
