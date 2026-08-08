//! SVG 文档构建：defs（渐变）+ 填充路径 + 描边轮廓。

pub struct SvgBuilder {
    width: f32,
    height: f32,
    defs: Vec<String>,
    bodies: Vec<String>,
    grad_counter: usize,
}

impl SvgBuilder {
    pub fn new(width: u32, height: u32) -> Self {
        SvgBuilder {
            width: width as f32,
            height: height as f32,
            defs: Vec::new(),
            bodies: Vec::new(),
            grad_counter: 0,
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
