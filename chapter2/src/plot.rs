use plotters::drawing::IntoDrawingArea;
use plotters::style::{IntoFont, BLUE, RED, WHITE};
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::series::{PointSeries, DashedLineSeries};
use plotters::prelude::{SVGBackend, Circle};
pub use plotters::prelude::{IntoLogRange, BindKeyPoints};

pub struct Plotter<Y: AsRangedCoord<Value = f64>> {
    caption: String,
    y_desc: String,
    y_coord: Y,
    data: Vec<[f64; 100]>,
}

pub struct PlotterInit<Y: AsRangedCoord<Value = f64>> {
    pub caption: String,
    pub y_desc: &'static str,
    pub y_coord: Y,
}

impl<Y: AsRangedCoord<Value = f64>> Plotter<Y> {
    pub fn init(init: PlotterInit<Y>) -> Self {
        Self {
            caption: init.caption,
            y_desc: init.y_desc.to_string(),
            y_coord: init.y_coord,
            data: Vec::new(),
        }
    }
}

impl<Y: AsRangedCoord<Value = f64>> Plotter<Y> {
    pub fn plot(&mut self, values: [f64; 100]) {
        self.data.push(values);
    }
    
    pub fn write_into(self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        <Y as AsRangedCoord>::CoordDescType: ValueFormatter<<Y as AsRangedCoord>::Value>,
    {
        let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = plotters::chart::ChartBuilder::on(&root)
            .caption(self.caption + " [点線は平均値]", ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                (-5..105).with_key_points(vec![0, 20, 40, 60, 80, 100]),
                self.y_coord,
            )?;
        
        chart.configure_mesh()
            .x_desc("trials")
            .y_desc(self.y_desc)
            .y_label_formatter(&Self::format_y_label)
            .axis_desc_style(("sans-serif", 16).into_font())
            .label_style(("sans-serif", 16).into_font())
            .draw()?;
        
        for values in self.data {
            let average = values.iter().copied().sum::<f64>() / (values.len() as f64);
            chart.draw_series(DashedLineSeries::new(
                (0..100).map(|i| (i, average)),
                2,
                1,
                RED.into(),
            ))?;
            chart.draw_series(PointSeries::<_, _, Circle<(i32, f64), i32>, _>::new(
                (0..100).map(|i| (i, values[i as usize])),
                2,
                BLUE
            ))?;
        }
        
        root.present()?;
        
        Ok(())
    }
    
    fn format_y_label(value: &f64) -> String {
        const SUPERSCRIPT: &[char] = &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁻'];
        
        if *value >= 1.0 || *value == 0.0 {
            value.to_string()
        } else if *value < 0.0 {
            format!("-{}", Self::format_y_label(&-*value))
        } else {
            let log10_abs = value.log10().abs();
            let coefficient = (log10_abs.fract() > 0.0)
                .then(|| format!("{:.1}×", 10f64.powf(log10_abs.fract())));
            let exponent = format!("10⁻{}", (log10_abs.floor() as usize)
                .to_string()
                .chars()
                .map(|c| SUPERSCRIPT[c.to_digit(10).unwrap() as usize])
                .collect::<String>()
            );
            format!("{}{}", coefficient.unwrap_or_default(), exponent)
        }
    }
}
