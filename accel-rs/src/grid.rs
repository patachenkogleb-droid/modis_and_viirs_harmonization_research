use anyhow::{ensure, Result};
use rayon::prelude::*;

use crate::filters::Fire;

const SEMI_MAJOR_M: f64 = 6_378_137.0;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_379_990_141_316_5;

pub fn cell(lon: f64, lat: f64, size: f64) -> Result<(i32, i32)> {
    ensure!((-180.0..=180.0).contains(&lon), "Invalid longitude");
    ensure!((-86.0..=86.0).contains(&lat), "Latitude outside EPSG:6933 domain");
    let e = ECCENTRICITY_SQUARED.sqrt();
    let phi0 = 30.0_f64.to_radians();
    let k0 = phi0.cos() / (1.0 - ECCENTRICITY_SQUARED * phi0.sin().powi(2)).sqrt();
    let sin_phi = lat.to_radians().sin();
    let u = e * sin_phi;
    let q = (1.0 - ECCENTRICITY_SQUARED)
        * (sin_phi / (1.0 - u * u) + u.atanh() / e);
    let x = SEMI_MAJOR_M * k0 * lon.to_radians();
    let y = SEMI_MAJOR_M * q / (2.0 * k0);
    Ok(((x / size).floor() as i32, (y / size).floor() as i32))
}

pub fn assign(fires: &[Fire], size: f64) -> Result<Vec<(i32, i32)>> {
    fires.par_iter().map(|fire| cell(fire.lon, fire.lat, size)).collect()
}
