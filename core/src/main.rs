//! png2svg —— 面向 icon 的 PNG→SVG 转换工具。
//!
//! 设计（轮廓由“成熟黑白轮廓识别”得到，渐变/阴影沿用原有平滑拟合）：
//!   1. 先用洪泛分割（segment）把图像分成区域，仅用来识别【渐变 / 阴影】两类：
//!        * 渐变（Gradient）—— 颜色随位置平滑变化，细分为线性 / 径向 / 网格；
//!        * 阴影（Shadow）  —— 半透明（平均 alpha 偏低），降低不透明度画在底层。
//!      这两类的像素会被标记为 claimed，不参与纯色步骤。
//!   2. 纯色采用“成熟黑白轮廓识别”：把前景里未被渐变/阴影占用的像素，按颜色
//!      范围（color_tol）聚成若干调色板颜色；每种颜色生成一张黑白掩码
//!      （该颜色=黑、其余=白），描出轮廓后填该色，把所有颜色的结果叠加，
//!      即得到完整、干净的彩色轮廓。这样同一颜色的相邻碎块会自动合并成
//!      一个轮廓，避免逐区域描边产生的碎片与幽灵圆。
//!
//! 用法：
//!   png2svg input.png -o out.svg
//!   png2svg gen-test test_icon.png      # 生成含渐变环/纯色块/阴影的测试图标

mod contour;
mod gradient;
mod raster;
mod segment;
mod svg;

use gradient::{fit, FitParams};
use raster::{color_dist, Raster};
use std::process::exit;

/// alpha 维度在综合距离中的权重（让半透明阴影与实色明显分离）。
const ALPHA_W: f32 = 1.0;
/// 判定为阴影的最小像素面积，过滤掉边缘抗锯齿产生的细小半透明噪点。
const MIN_SHADOW_AREA: usize = 6;

const KIND_SHADOW: u8 = 0;
const KIND_GRADIENT: u8 = 1;
const KIND_SOLID: u8 = 2;

struct Opts {
    input: String,
    output: Option<String>,
    tolerance: f32,
    alpha_thresh: f32,
    bg_thresh: f32,
    invert: bool,
    shadow_alpha: f32,
    solid_eps: f32,
    r2_thresh: f32,
    simplify: f32,
    circ_tol: f32,
    ell_tol: f32,
    smooth: bool,
    /// 纯色按颜色范围聚类的容差（即“颜色范围”的粗细）
    color_tol: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        exit(1);
    }
    if args[1] == "gen-test" {
        let out = args.get(2).cloned().unwrap_or_else(|| "test_icon.png".into());
        gen_test(&out);
        return;
    }

    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("参数错误：{e}");
            exit(1);
        }
    };

    let raster = match Raster::load(&opts.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            exit(1);
        }
    };
    let (w, h) = (raster.width, raster.height);
    let fg = build_foreground(&raster, opts.alpha_thresh, opts.bg_thresh, opts.invert);

    let mut builder = svg::SvgBuilder::new(w, h);

    // Step A：分割 + 识别。渐变 / 阴影区域沿用原有的平滑拟合渲染
    // （“以前的渐变 这些保留”），并把它们的像素标记为 claimed，避免下一步
    // 被当作纯色重复渲染。纯色区域此处不参与渲染。
    let regions = segment::segment(&raster, &fg, opts.tolerance, ALPHA_W);
    let params = FitParams {
        min_var: opts.solid_eps,
        r2_thresh: opts.r2_thresh,
    };
    let n = (w * h) as usize;
    let mut claimed = vec![false; n];

    // 阴影画在最底层、渐变在中间层；纯色在 Step B 单独处理。
    let mut shadows: Vec<(String, gradient::Fit, f32)> = Vec::new();
    let mut gradients: Vec<(String, gradient::Fit)> = Vec::new();

    for reg in &regions {
        let avg_a = reg.samples.iter().map(|&(_, _, _, _, _, a)| a as f32).sum::<f32>()
            / reg.samples.len() as f32;
        let f = fit(&reg.samples, &params);
        let kind = if avg_a < opts.shadow_alpha && reg.samples.len() >= MIN_SHADOW_AREA {
            KIND_SHADOW
        } else if f.is_gradient {
            KIND_GRADIENT
        } else {
            KIND_SOLID
        };

        // 区域包围盒（细长条效验，丢弃近直线边缘的抗锯齿像素，消除幽灵圆）
        let (mut bx0, mut by0, mut bx1, mut by1) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y, _, _, _, _) in &reg.samples {
            if x < bx0 {
                bx0 = x;
            }
            if y < by0 {
                by0 = y;
            }
            if x > bx1 {
                bx1 = x;
            }
            if y > by1 {
                by1 = y;
            }
        }
        let bw = (bx1 - bx0).max(0.0);
        let bh = (by1 - by0).max(0.0);
        let aspect = if bw.min(bh) > 0.0 {
            bw.max(bh) / bw.min(bh)
        } else {
            f32::MAX
        };
        if kind == KIND_SHADOW && aspect > 5.0 {
            continue;
        }

        // 纯色区域不参与此处渲染，交给 Step B 的“按颜色量化”统一处理。
        if kind == KIND_SOLID {
            continue;
        }

        let loops = contour::trace_with_holes(&reg.mask, w, h);
        let d = contour::loops_to_path(
            &loops,
            opts.simplify,
            opts.circ_tol,
            opts.ell_tol,
            opts.smooth,
        );
        if d.is_empty() {
            continue;
        }
        if kind == KIND_SHADOW {
            shadows.push((d, f, (avg_a / 255.0).clamp(0.0, 1.0)));
        } else {
            gradients.push((d, f));
        }
        // 标记为 claimed，纯色步骤不再触碰这些像素
        for i in 0..n {
            if reg.mask[i] {
                claimed[i] = true;
            }
        }
    }

    // Step B：纯色 —— 成熟的“黑白轮廓识别”。
    // 把前景里未被渐变/阴影占用的像素，按颜色范围（color_tol）聚成若干调色板颜色；
    // 每种颜色生成一张黑白掩码（该颜色=黑、其余=白），描出轮廓后填该色，
    // 把所有颜色的结果叠加，即得到完整、干净的彩色轮廓。
    let solid_buckets = quantize_solids(&raster, &fg, &claimed, opts.color_tol);
    if std::env::var("PNG2SVG_DEBUG").is_ok() {
        eprintln!(
            "[dbg] solid buckets = {} （color_tol={}）",
            solid_buckets.len(),
            opts.color_tol
        );
        for (i, b) in solid_buckets.iter().enumerate() {
            let mut minx = usize::MAX;
            let mut miny = usize::MAX;
            let mut maxx = 0usize;
            let mut maxy = 0usize;
            for (idx, &v) in b.mask.iter().enumerate() {
                if v {
                    let x = idx % (w as usize);
                    let y = idx / (w as usize);
                    if x < minx { minx = x; }
                    if y < miny { miny = y; }
                    if x > maxx { maxx = x; }
                    if y > maxy { maxy = y; }
                }
            }
            eprintln!(
                "[dbg]   bucket#{} color=rgb({},{},{}) pixels={} bbox=({},{}),({},{})",
                i,
                b.color.0,
                b.color.1,
                b.color.2,
                b.mask.iter().filter(|&&v| v).count(),
                minx,
                miny,
                maxx,
                maxy
            );
        }
    }

    // Step C：按层级渲染：阴影（底层）→ 渐变 → 纯色（顶层）。
    for (d, f, op) in &shadows {
        if f.is_gradient {
            render_gradient(&mut builder, d, f, *op);
        } else {
            builder.add_solid_path(d, f.mean, *op);
        }
    }
    for (d, f) in &gradients {
        render_gradient(&mut builder, d, f, 1.0);
    }
    for bucket in &solid_buckets {
        let loops = contour::trace_with_holes(&bucket.mask, w, h);
        let d = contour::loops_to_path(
            &loops,
            opts.simplify,
            opts.circ_tol,
            opts.ell_tol,
            opts.smooth,
        );
        if !d.is_empty() {
            builder.add_solid_path(&d, bucket.color, 1.0);
        }
    }

    let doc = builder.to_string();
    match &opts.output {
        Some(path) => match std::fs::write(path, &doc) {
            Ok(_) => println!(
                "已写出：{path}（{} 字节，{} 渐变 + {} 阴影 + {} 纯色）",
                doc.len(),
                gradients.len(),
                shadows.len(),
                solid_buckets.len()
            ),
            Err(e) => {
                eprintln!("写入失败：{e}");
                exit(1);
            }
        },
        None => println!("{doc}"),
    }
}

/// 构建前景掩码（即“哪些像素属于图标，而非外部空白”）。
/// - 有透明通道：alpha >= 阈值 即前景（阈值调低以纳入半透明阴影）；
/// - 无透明通道：以四条边框平均色为背景，颜色距离 > 阈值 即前景（--invert 翻转）。
fn build_foreground(raster: &Raster, alpha_thresh: f32, bg_thresh: f32, invert: bool) -> Vec<bool> {
    if raster.has_alpha() {
        raster
            .pixels
            .iter()
            .map(|&(_, _, _, a)| (a as f32) >= alpha_thresh)
            .collect()
    } else {
        let bg = raster.border_average();
        raster
            .pixels
            .iter()
            .map(|&(r, g, b, _)| {
                let fg = color_dist((r, g, b), bg) > bg_thresh;
                if invert {
                    !fg
                } else {
                    fg
                }
            })
            .collect()
    }
}

/// 按渐变类型把一条轮廓路径渲染成渐变填充（线性 / 径向 / 网格）。
pub fn render_gradient(builder: &mut svg::SvgBuilder, d: &str, f: &gradient::Fit, op: f32) {
    match f.grad_kind {
        gradient::GradKind::Linear => {
            builder.add_gradient_path(d, f.start_pt, f.end_pt, f.start_color, f.end_color, op)
        }
        gradient::GradKind::Radial => builder.add_radial_gradient_path(
            d,
            f.center,
            f.radius,
            f.rad_center,
            f.rad_rim,
            op,
        ),
        gradient::GradKind::Mesh => {
            if let Some(ref mesh) = f.mesh {
                builder.add_mesh_path(d, mesh, op);
            } else {
                // 不应发生：mesh 系数缺失时回退为线性
                builder.add_gradient_path(d, f.start_pt, f.end_pt, f.start_color, f.end_color, op);
            }
        }
    }
}

/// 一个纯色调色板桶：代表图像里“一种颜色范围”，以及它在整图中的黑白掩码。
pub struct SolidBucket {
    /// 桶的代表色（运行均值四舍五入）
    pub color: (u8, u8, u8),
    /// 该颜色像素的掩码（true = 属于此桶），即“该颜色=黑、其余=白”的黑白图。
    pub mask: Vec<bool>,
}

/// 对前景中未被 claimed（即非渐变 / 非阴影）的像素，按颜色范围聚成若干调色板颜色。
///
/// 每种颜色对应一张黑白掩码（该颜色像素=黑、其余=白）；后续对每张掩码描轮廓、
/// 填该色并叠加，即得到由各颜色轮廓叠加而成的完整彩色轮廓。相邻同色碎块会因
/// 同属一个桶而自动合并成一个轮廓。
fn quantize_solids(
    raster: &Raster,
    fg: &[bool],
    claimed: &[bool],
    tol: f32,
) -> Vec<SolidBucket> {
    let n = fg.len();
    // 每个桶：运行均值（f64）+ 掩码 + 计数
    let mut reps: Vec<(f64, f64, f64)> = Vec::new();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();

    for i in 0..n {
        if !fg[i] || claimed[i] {
            continue;
        }
        let (r, g, b, _) = raster.pixels[i];
        // 找最近的已有桶（颜色距离 <= tol）
        let mut best: Option<usize> = None;
        let mut best_d = f32::MAX;
        for (bi, rep) in reps.iter().enumerate() {
            let d = raster::color_dist((r, g, b), (rep.0 as u8, rep.1 as u8, rep.2 as u8));
            if d <= tol && d < best_d {
                best_d = d;
                best = Some(bi);
            }
        }
        match best {
            Some(bi) => {
                let c = (counts[bi] + 1) as f64;
                // 在线更新运行均值（作为后续像素的距离基准，也用于最终填色）
                reps[bi].0 += (r as f64 - reps[bi].0) / c;
                reps[bi].1 += (g as f64 - reps[bi].1) / c;
                reps[bi].2 += (b as f64 - reps[bi].2) / c;
                counts[bi] += 1;
                masks[bi][i] = true;
            }
            None => {
                let mut m = vec![false; n];
                m[i] = true;
                reps.push((r as f64, g as f64, b as f64));
                masks.push(m);
                counts.push(1);
            }
        }
    }

    reps
        .into_iter()
        .zip(masks)
        .map(|(rep, mask)| SolidBucket {
            color: (rep.0 as u8, rep.1 as u8, rep.2 as u8),
            mask,
        })
        .collect()
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        input: String::new(),
        output: None,
        tolerance: 28.0,
        alpha_thresh: 8.0,
        bg_thresh: 40.0,
        invert: false,
        shadow_alpha: 230.0,
        solid_eps: 8.0,
        r2_thresh: 0.6,
        simplify: 0.5,
        circ_tol: 0.025,
        ell_tol: 0.06,
        smooth: true,
        color_tol: 32.0,
    };
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "-o" | "--output" => {
                opts.output = args.get(i + 1).cloned();
                i += 2;
            }
            "--tolerance" => {
                opts.tolerance = next_f32(args, i)?;
                i += 2;
            }
            "--alpha" => {
                opts.alpha_thresh = next_f32(args, i)?;
                i += 2;
            }
            "--bg-thresh" => {
                opts.bg_thresh = next_f32(args, i)?;
                i += 2;
            }
            "--shadow-alpha" => {
                opts.shadow_alpha = next_f32(args, i)?;
                i += 2;
            }
            "--invert" => {
                opts.invert = true;
                i += 1;
            }
            "--solid-eps" => {
                opts.solid_eps = next_f32(args, i)?;
                i += 2;
            }
            "--r2" => {
                opts.r2_thresh = next_f32(args, i)?;
                i += 2;
            }
            "--simplify" => {
                opts.simplify = next_f32(args, i)?;
                i += 2;
            }
            "--circ-tol" => {
                opts.circ_tol = next_f32(args, i)?;
                i += 2;
            }
            "--ell-tol" => {
                opts.ell_tol = next_f32(args, i)?;
                i += 2;
            }
            "--no-smooth" => {
                opts.smooth = false;
                i += 1;
            }
            "--color-tol" => {
                opts.color_tol = next_f32(args, i)?;
                i += 2;
            }
            s if !s.starts_with('-') => {
                if opts.input.is_empty() {
                    opts.input = s.to_string();
                }
                i += 1;
            }
            _ => {
                return Err(format!("未知参数：{a}"));
            }
        }
    }
    if opts.input.is_empty() {
        return Err("缺少输入文件".into());
    }
    Ok(opts)
}

fn next_f32(args: &[String], i: usize) -> Result<f32, String> {
    args.get(i + 1)
        .ok_or_else(|| format!("参数 {} 缺少数值", args[i]))?
        .parse::<f32>()
        .map_err(|_| format!("参数 {} 不是合法数字", args[i]))
}

fn print_usage() {
    println!(
        "png2svg —— 面向 icon 的 PNG→SVG 工具（按颜色/alpha 区分轮廓，识别纯色/渐变/阴影）\n\
用法：\n\
  png2svg <input.png> [-o out.svg] [选项]\n\
  png2svg gen-test [out.png]           生成测试图标\n\
选项：\n\
  --tolerance N      颜色+alpha 分割容差（默认 28）\n\
  --alpha N          前景 alpha 阈值 0-255（默认 8，调低可纳入更淡的阴影）\n\
  --bg-thresh N      无 alpha 时的背景距离阈值（默认 40）\n\
  --shadow-alpha N   平均 alpha 低于此值判为阴影（默认 230）\n\
  --invert           无 alpha 时翻转前景/背景\n\
  --solid-eps N      判定为渐变的最小整片颜色变化量（默认 8）\n\
  --r2 N             判定为渐变的最小 R²（默认 0.6）\n\
  --simplify N       轮廓简化精度（默认 0.5，越小点越密）\n\
  --no-smooth        关闭曲线平滑，回退为纯多边形（配合小 --simplify 可得极密多边形）\n\
  --circ-tol N       圆拟合容许误差/半径（默认 0.06，越小越严格）\n\
  --ell-tol N        椭圆拟合容许误差（默认 0.06，越小越严格）\n\
  --color-tol N      纯色按颜色范围聚类的容差（默认 32，越大合并越多颜色）"
    );
}

/// 生成一个测试图标：渐变环（带孔）+ 两个纯色方块 + 一个半透明阴影。
fn gen_test(path: &str) {
    use image::{Rgba, RgbaImage};
    let size = 256u32;
    let mut img = RgbaImage::new(size, size);
    let cx = 128.0;
    let cy = 128.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            let (pr, pg, pb, pa) = if r <= 100.0 && r >= 40.0 {
                // 渐变环：颜色随 x、y 线性变化
                let red = (60.0 + x as f32 * 0.75).clamp(0.0, 255.0) as u8;
                let green = 80u8;
                let blue = (60.0 + y as f32 * 0.75).clamp(0.0, 255.0) as u8;
                (red, green, blue, 255)
            } else if x >= 170 && x <= 220 && y >= 30 && y <= 80 {
                (40, 190, 120, 255) // 纯色绿方块
            } else if x >= 30 && x <= 70 && y >= 180 && y <= 220 {
                (230, 80, 60, 255) // 纯色红方块
            } else if ((x as f32 - 128.0).powi(2)) / (48.0_f32.powi(2))
                + ((y as f32 - 240.0).powi(2)) / (13.0_f32.powi(2))
                <= 1.0
            {
                (15, 15, 20, 75) // 半透明投影（阴影）
            } else {
                (0, 0, 0, 0)
            };
            img.put_pixel(x, y, Rgba([pr, pg, pb, pa]));
        }
    }
    if let Err(e) = img.save(path) {
        eprintln!("保存测试图失败：{e}");
        exit(1);
    }
    println!("已生成测试图标：{path}");
}
