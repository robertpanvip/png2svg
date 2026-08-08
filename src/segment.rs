//! 区域分割（对应“靠颜色区分轮廓 + 区分纯色/渐变/阴影”）。
//!
//! 对每个未访问的前景像素，以它为种子做 4 邻域漫水填充，扩展条件是
//! “与相邻像素的综合距离 <= 容差”。综合距离 = sqrt(RGB距离² + (α权重·α差)²)。
//! 这样：
//!   - 相邻像素颜色几乎一样        -> 合并为【纯色区域】；
//!   - 颜色沿某方向平滑变化        -> 合并为【渐变区域】；
//!   - 半透明像素（阴影）与实色    -> 因 α 差巨大而被自动切分为独立区域【阴影】。
//! 之后对每个区域单独拟合/分类（见 gradient / main）。

use crate::contour::four_neighbors;
use crate::raster::Raster;

pub struct Region {
    /// (x, y, r, g, b, a) 采样点
    pub samples: Vec<(f32, f32, u8, u8, u8, u8)>,
    /// 该区域在整图中的掩码（true = 属于本区域）
    pub mask: Vec<bool>,
}

/// 颜色 + alpha 综合距离。alpha 维度加权，使半透明阴影与实色明显分离。
#[inline]
pub fn color_alpha_dist(a: (u8, u8, u8, u8), b: (u8, u8, u8, u8), alpha_w: f32) -> f32 {
    let dr = a.0 as f32 - b.0 as f32;
    let dg = a.1 as f32 - b.1 as f32;
    let db = a.2 as f32 - b.2 as f32;
    let da = (a.3 as f32 - b.3 as f32) * alpha_w;
    (dr * dr + dg * dg + db * db + da * da).sqrt()
}

/// 对前景掩码做分割，得到若干区域。alpha_w 控制 alpha 差异在距离中的权重。
pub fn segment(raster: &Raster, fg: &[bool], tolerance: f32, alpha_w: f32) -> Vec<Region> {
    let w = raster.width;
    let h = raster.height;
    let n = fg.len();
    let mut visited = vec![false; n];
    let mut regions = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let i = raster.idx(x, y);
            if !fg[i] || visited[i] {
                continue;
            }
            // --- 漫水填充（4 邻域，相邻综合距离容差）---
            let mut stack = vec![(x, y)];
            let mut samples = Vec::new();
            let mut mask = vec![false; n];
            while let Some((cx, cy)) = stack.pop() {
                let ci = raster.idx(cx, cy);
                if visited[ci] || !fg[ci] {
                    continue;
                }
                visited[ci] = true;
                mask[ci] = true;
                let (r, g, b, a) = raster.pixels[ci];
                samples.push((cx as f32, cy as f32, r, g, b, a));
                for (nx, ny) in four_neighbors(cx, cy, w, h) {
                    let ni = raster.idx(nx, ny);
                    if visited[ni] || !fg[ni] {
                        continue;
                    }
                    let (nr, ng, nb, na) = raster.pixels[ni];
                    if color_alpha_dist((r, g, b, a), (nr, ng, nb, na), alpha_w) <= tolerance {
                        stack.push((nx, ny));
                    }
                }
            }
            regions.push(Region { samples, mask });
        }
    }
    regions
}
