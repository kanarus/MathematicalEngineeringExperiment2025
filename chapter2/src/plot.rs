use plotters::drawing::IntoDrawingArea;
use plotters::style::{Color, IntoFont, BLUE, RED, WHITE};
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::series::{PointSeries, DashedLineSeries};
use plotters::prelude::{SVGBackend, Circle, IntoLogRange, BindKeyPoints};

pub struct Plotter {
    pub y_desc: &'static str,
    pub data: [f64; 100],
}

impl Plotter {
    pub fn plot_into(self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = plotters::chart::ChartBuilder::on(&root)
            // .caption(format!("{} (点線は平均値)", self.caption), ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(
                (-5..105).with_key_points(vec![0, 20, 40, 60, 80, 100]),
                self.derive_y_coord(),
            )?;
        
        chart.configure_mesh()
            .x_desc("trials")
            .y_desc(self.y_desc)
            .y_label_formatter(&Self::format_y_label)
            .axis_desc_style(("sans-serif", 16).into_font())
            .label_style(("sans-serif", 16).into_font())
            .draw()?;
        
        let average = self.data.iter().copied().sum::<f64>() / (self.data.len() as f64);
        chart.draw_series(DashedLineSeries::new(
            (-5..105).map(|i| (i, average)),
            2,
            1,
            RED.into(),
        ))?;
        chart.draw_series(PointSeries::<_, _, Circle<(i32, f64), i32>, _>::new(
            (0..100).map(|i| (i, self.data[i as usize])),
            2,
            BLUE.filled(),
        ))?;
        
        root.present()?;
        
        Ok(())
    }
    
    fn derive_y_coord(&self) -> impl AsRangedCoord<Value = f64, CoordDescType: ValueFormatter<f64>> {
        let min = self.data.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = self.data.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        assert!(min > 0.0, "expected positive values for plotting in chapter 2");
        
        let (min_coef, min_exp) = {
            let (log10, log10_floor) = (min.log10(), min.log10().floor());
            (10f64.powf(log10 - log10_floor), log10_floor as i32)
        };
        let (max_coef, max_exp) = {
            let (log10, log10_floor) = (max.log10(), max.log10().floor());
            (10f64.powf(log10 - log10_floor), log10_floor as i32)
        };
        
        let key_points = if min_exp != max_exp {
            (min_exp - 1..=max_exp + 1)
                .map(|e| 10f64.powi(e))
                .collect::<Vec<_>>()
        } else {
            (min_coef.floor() as i32..=max_coef.ceil() as i32)
                .map(|c| (c as f64) * 10f64.powi(max_exp))
                .collect::<Vec<_>>()
        };
        
        (*key_points.first().unwrap()..*key_points.last().unwrap())
            .log_scale()
            .with_key_points(key_points)
    }
    
    fn format_y_label(value: &f64) -> String {
        const SUPERSCRIPT: &[char] = &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁻'];
        
        if *value >= 1.0 || *value == 0.0 {
            value.to_string()
        } else if *value < 0.0 {
            format!("-{}", Self::format_y_label(&-*value))
        } else {
            let log10 = value.log10();
            let log10_floor = log10.floor();
            let coefficient = (log10 - log10_floor > crate::EPSILON)
                .then(|| {
                    let value = 10f64.powf(log10 - log10_floor);
                    if value.fract() < crate::EPSILON {
                        format!("{}×", value.round() as i32)
                    } else {
                        format!("{:.1}×", value)
                    }
                });
            let exponent = format!("10⁻{}", (log10_floor.abs() as u32)
                .to_string()
                .chars()
                .map(|c| SUPERSCRIPT[c.to_digit(10).unwrap() as usize])
                .collect::<String>()
            );
            format!("{}{}", coefficient.unwrap_or_default(), exponent)
        }
    }
}
