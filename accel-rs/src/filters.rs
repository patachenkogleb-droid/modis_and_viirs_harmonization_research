use anyhow::{bail, ensure, Context, Result};
use csv::ByteRecord;

pub struct Fire {
    pub lat: f64,
    pub lon: f64,
    pub frp: f64,
}

pub struct Fields {
    lat: usize,
    lon: usize,
    frp: usize,
    scan: usize,
    track: usize,
    kind: usize,
    satellite: usize,
}

impl Fields {
    pub fn new(header: &ByteRecord) -> Result<Self> {
        let position = |name: &str| {
            header.iter().position(|column| {
                column.strip_prefix(b"\xef\xbb\xbf").unwrap_or(column) == name.as_bytes()
            }).with_context(|| format!("Missing CSV column {name}"))
        };
        Ok(Self {
            lat: position("latitude")?, lon: position("longitude")?,
            frp: position("frp")?, scan: position("scan")?, track: position("track")?,
            kind: position("type")?,
            satellite: position("satellite")?,
        })
    }

    pub fn parse(&self, record: &ByteRecord, max_aqua_scan: Option<f64>,
                 bbox: &[f64]) -> Result<Option<Fire>> {
        let text = |index| -> Result<&str> {
            Ok(std::str::from_utf8(record.get(index).context("Short CSV row")?)?.trim())
        };
        let number = |index| -> Result<f64> {
            let value: f64 = text(index)?.parse()?;
            ensure!(value.is_finite(), "Non-finite number");
            Ok(value)
        };
        match text(self.kind)? {
            "0" => (),
            "1" | "2" | "3" => return Ok(None),
            _ => bail!("Invalid type"),
        }
        let satellite = text(self.satellite)?;
        let aqua = satellite.eq_ignore_ascii_case("aqua") || satellite.eq_ignore_ascii_case("a");
        if satellite.eq_ignore_ascii_case("terra") || satellite.eq_ignore_ascii_case("t") {
            return Ok(None);
        }
        ensure!(aqua || ["N", "NPP", "SNPP", "S-NPP"].iter().any(|s| satellite.eq_ignore_ascii_case(s)),
                "Unexpected satellite");
        let lon = number(self.lon)?;
        let lat = number(self.lat)?;
        if lon < bbox[0] || lat < bbox[1] || lon > bbox[2] || lat > bbox[3] {
            return Ok(None);
        }
        let optional_number = |index| -> Result<f64> {
            let value = text(index)?;
            if value.is_empty() {
                return Ok(f64::NAN);
            }
            Ok(value.parse()?)
        };
        let scan = optional_number(self.scan)?;
        let track = optional_number(self.track)?;
        let frp = optional_number(self.frp)?;
        if !(scan.is_finite() && track.is_finite() && frp.is_finite()
             && scan > 0.0 && track > 0.0 && frp >= 0.0) {
            return Ok(None);
        }
        if aqua && max_aqua_scan.is_some_and(|limit| scan > limit) {
            return Ok(None);
        }
        Ok(Some(Fire { lat, lon, frp }))
    }
}
