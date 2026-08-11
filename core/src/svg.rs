//! SVG 文档构建：defs（渐变）+ 填充路径 + 描边轮廓。

use crate::gradient;

pub struct SvgBuilder {
    width: f32,
    height: f32,
    defs: Vec<String>,
    bodies: Vec<String>,
    grad_counter: usize,
    clip_counter: usize,
}

impl SvgBuilder {
    pub fn new(width: u32, height: u32) -> Self {
        SvgBuilder {
            width: width as f32,
            height: height as f32,
            defs: Vec::new(),
            bodies: Vec::new(),
            grad_counter: 0,
            clip_counter: 0,
        }
    }

    fn rgb(c: (u8, u8, u8)) -> String {
        format!("rgb({},{},{})", c.0, c.1, c.2)
    }

    /// 添加一个纯色填充路径（d 可含多个子路径，用 evenodd 形成孔洞）。
    /// opacity < 1.0 时输出 fill-opacity（用于半透明阴影）。
    pub fn add_solid_path(&mut self, d: &str, color: (u8, u8, u8), opacity: f32) {
        if d.is_empty() {
            return;
        }
        let op = if opacity < 1.0 {
            format!(" fill-opacity=\"{:.3}\"", opacity)
        } else {
            String::new()
        };
        self.bodies.push(format!(
            "<path d=\"{}\" fill=\"{}\" fill-rule=\"evenodd\"{}/>",
            d,
            Self::rgb(color),
            op
        ));
    }

    /// 添加一个线性渐变填充路径，返回渐变是否被创建。
    /// opacity < 1.0 时输出 fill-opacity（用于半透明渐变阴影）。
    pub fn add_gradient_path(
        &mut self,
        d: &str,
        start_pt: (f32, f32),
        end_pt: (f32, f32),
        start_color: (u8, u8, u8),
        end_color: (u8, u8, u8),
        opacity: f32,
    ) {
        if d.is_empty() {
            return;
        }
        self.grad_counter += 1;
        let id = format!("g{}", self.grad_counter);
        self.defs.push(format!(
            "<linearGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\">\
<stop offset=\"0\" stop-color=\"{}\"/>\
<stop offset=\"1\" stop-color=\"{}\"/>\
</linearGradient>",
            id,
            start_pt.0,
            start_pt.1,
            end_pt.0,
            end_pt.1,
            Self::rgb(start_color),
            Self::rgb(end_color)
        ));
        let op = if opacity < 1.0 {
            format!(" fill-opacity=\"{:.3}\"", opacity)
        } else {
            String::new()
        };
        self.bodies.push(format!(
            "<path d=\"{}\" fill=\"url(#{})\" fill-rule=\"evenodd\"{}/>",
            d, id, op
        ));
    }

    /// 添加一个径向渐变填充路径（userSpaceOnUse，中心 + 半径）。opacity<1 时降不透明度。
    pub fn add_radial_gradient_path(
        &mut self,
        d: &str,
        center: (f32, f32),
        radius: f32,
        center_color: (u8, u8, u8),
        rim_color: (u8, u8, u8),
        opacity: f32,
    ) {
        if d.is_empty() {
            return;
        }
        self.grad_counter += 1;
        let id = format!("rg{}", self.grad_counter);
        let r = radius.max(0.5);
        self.defs.push(format!(
            "<radialGradient id=\"{}\" gradientUnits=\"userSpaceOnUse\" cx=\"{}\" cy=\"{}\" r=\"{}\" fx=\"{}\" fy=\"{}\">\
<stop offset=\"0\" stop-color=\"{}\"/>\
<stop offset=\"1\" stop-color=\"{}\"/>\
</radialGradient>",
            id,
            center.0,
            center.1,
            r,
            center.0,
            center.1,
            Self::rgb(center_color),
            Self::rgb(rim_color)
        ));
        let op = if opacity < 1.0 {
            format!(" fill-opacity=\"{:.3}\"", opacity)
        } else {
            String::new()
        };
        self.bodies.push(format!(
            "<path d=\"{}\" fill=\"url(#{})\" fill-rule=\"evenodd\"{}/>",
            d, id, op
        ));
    }

    /// 添加网格（双线性）渐变填充：用 clipPath 把区域形状裁出，内部铺一层细网格，
    /// 每个格子用双线性函数在该格中心处的颜色平涂，逼近“网格渐变”。opacity<1 时整组降透明。
    pub fn add_mesh_path(&mut self, d: &str, mesh: &gradient::MeshCoef, opacity: f32) {
        if d.is_empty() {
            return;
        }
        let width = (mesh.maxx - mesh.minx).max(0.0);
        let height = (mesh.maxy - mesh.miny).max(0.0);
        if width < 1.0 || height < 1.0 {
            return;
        }
        // 网格分辨率：约 5px 一格（更贴近细渐变，减少块状 moiré），上限放宽到 80 避免元素过多
        let cols = (width / 5.0).max(4.0).min(80.0).round() as i32;
        let rows = (height / 5.0).max(4.0).min(80.0).round() as i32;
        let cw = width / cols as f32;
        let ch = height / rows as f32;

        let mut rects = String::new();
        for j in 0..rows {
            for i in 0..cols {
                let cx = mesh.minx as f64 + (i as f64 + 0.5) * cw as f64;
                let cy = mesh.miny as f64 + (j as f64 + 0.5) * ch as f64;
                let col = mesh.color_at(cx, cy);
                rects.push_str(&format!(
                    "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\"/>",
                    mesh.minx + i as f32 * cw,
                    mesh.miny + j as f32 * ch,
                    cw,
                    ch,
                    Self::rgb(col)
                ));
            }
        }

        self.clip_counter += 1;
        let cid = format!("clip{}", self.clip_counter);
        self.defs.push(format!("<clipPath id=\"{}\"><path d=\"{}\"/></clipPath>", cid, d));
        let op = if opacity < 1.0 {
            format!(" opacity=\"{:.3}\"", opacity)
        } else {
            String::new()
        };
        self.bodies.push(format!(
            "<g clip-path=\"url(#{})\"{}>{}</g>",
            cid, op, rects
        ));
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
            self.width, self.height, self.width, self.height
        ));
        if !self.defs.is_empty() {
            s.push_str("<defs>");
            for d in &self.defs {
                s.push_str(d);
            }
            s.push_str("</defs>");
        }
        for b in &self.bodies {
            s.push_str(b);
        }
        s.push_str("</svg>");
        s
    }
}
