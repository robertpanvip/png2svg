//! 图像加载与前景掩码。
//!
//! 设计要点（对应需求 Step 1 的“黑白化 / 取轮廓”）：
//! - 有透明通道时，直接用 alpha 阈值区分“图形”与“背景”，最鲁棒；
//! - 无透明通道（被压平的背景图）时，用图像四条边的平均色作为背景色，
//!   再用每个像素与背景的颜色距离来区分前景，避免“亮色图形在白底上”被误判。

use image::DynamicImage;

pub struct Raster {
    pub width: u32,
    pub height: u32,
    /// (r, g, b, a)，每个分量 0..=255
    pub pixels: Vec<(u8, u8, u8, u8)>,
}

impl Raster {
    pub fn load(path: &str) -> Result<Raster, String> {
        let img: DynamicImage =
            image::open(path).map_err(|e| format!("无法打开图像 {path}: {e}"))?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let p = rgba.get_pixel(x, y);
                pixels.push((p[0], p[1], p[2], p[3]));
            }
        }
        Ok(Raster {
            width: w,
            height: h,
            pixels,
        })
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// 是否存在非完全不透明的像素。
    pub fn has_alpha(&self) -> bool {
        self.pixels.iter().any(|&(_, _, _, a)| a != 255)
    }

    /// 计算四条边框像素的平均颜色，作为“背景色”估计。
    pub fn border_average(&self) -> (u8, u8, u8) {
        let (w, h) = (self.width, self.height);
        let mut r = 0u64;
        let mut g = 0u64;
        let mut b = 0u64;
        let mut n = 0u64;
        let mut add = |x: u32, y: u32| {
            let (cr, cg, cb, _) = self.pixels[self.idx(x, y)];
            r += cr as u64;
            g += cg as u64;
            b += cb as u64;
            n += 1;
        };
        for x in 0..w {
            add(x, 0);
            add(x, h - 1);
        }
        for y in 1..h.saturating_sub(1) {
            add(0, y);
            add(w - 1, y);
        }
        if n == 0 {
            return (0, 0, 0);
        }
        (
            (r / n) as u8,
            (g / n) as u8,
            (b / n) as u8,
        )
    }
}

/// 两个 RGB 颜色之间的欧氏距离（0..≈441）。
#[inline]
pub fn color_dist(a: (u8, u8, u8), b: (u8, u8, u8)) -> f32 {
    let dr = a.0 as f32 - b.0 as f32;
    let dg = a.1 as f32 - b.1 as f32;
    let db = a.2 as f32 - b.2 as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}
