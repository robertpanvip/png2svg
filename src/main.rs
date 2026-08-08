//! png2svg —— 面向 icon 的 PNG→SVG 转换工具。
//!
//! 设计（按“靠颜色区分轮廓”）：
//!   - 轮廓不是单独二值化得到的，而是【每个彩色区域的边界】——区域之间靠
//!     颜色 / alpha 差异自然分开，边界即轮廓。
//!   - 每个区域再被识别为三类之一：
//!        * 纯色（Solid）        —— 颜色近似恒定；
//!        * 渐变（Gradient）     —— 颜色随位置线性变化（拟合平面 R² 高）；
//!        * 阴影（Shadow）       —— 半透明（平均 alpha 偏低），渲染时降低不透明度，
//!                                  并画在底层，模拟投影。
//!   识别顺序：先判阴影（alpha），再判渐变，最后兜底纯色。
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

    // Step A：按颜色 + alpha 分割区域
    let regions = segment::segment(&raster, &fg, opts.tolerance, ALPHA_W);
    let params = FitParams {
        min_var: opts.solid_eps,
        r2_thresh: opts.r2_thresh,
    };

    // Step B：识别每一块是 纯色 / 渐变 / 阴影，并记录几何
    let mut items: Vec<(u8, String, gradient::Fit, f32)> = Vec::new();
    for reg in &regions {
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
        items.push((kind, d, f, avg_a));
    }

    // Step C：阴影画在最底层，其余按类渲染
    items.sort_by_key(|&(kind, _, _, _)| kind);
    for (kind, d, f, avg_a) in &items {
        let op = (avg_a / 255.0).clamp(0.0, 1.0);
        match *kind {
            KIND_SHADOW => {
                if f.is_gradient {
                    builder.add_gradient_path(
                        d, f.start_pt, f.end_pt, f.start_color, f.end_color, op,
                    );
                } else {
                    builder.add_solid_path(d, f.mean, op);
                }
            }
            KIND_GRADIENT => builder.add_gradient_path(
                d, f.start_pt, f.end_pt, f.start_color, f.end_color, 1.0,
            ),
            _ => builder.add_solid_path(d, f.mean, 1.0),
        }
    }

    let doc = builder.to_string();
    match &opts.output {
        Some(path) => match std::fs::write(path, &doc) {
            Ok(_) => println!("已写出：{path}（{} 字节，{} 个区域）", doc.len(), items.len()),
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
        circ_tol: 0.06,
        ell_tol: 0.06,
        smooth: true,
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
  --ell-tol N        椭圆拟合容许误差（默认 0.06，越小越严格）"
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
