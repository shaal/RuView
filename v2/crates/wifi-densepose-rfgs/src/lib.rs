//! RFGS edge-query (ADR-125 P5, Field Serving context).
//!
//! Loads a baked RF Gaussian Splatting field exported by the Python trainer
//! (`wifi_densepose.rfgs.export.write_splats_v2_json`) and exposes it for the
//! `/api/splats?schema=rfgs-v2` server branch and for cheap inference-only edge
//! queries. The field is **immutable** at the edge (DDD invariant I3): ESP32
//! nodes relay CSI and never train; only the host-trained field is served here.
//!
//! Consumes the `splats-v2` JSON (not the `.rfgs.npz`) to avoid a numpy/npz
//! dependency in the Rust runtime — the JSON is already produced alongside the
//! npz and carries the full anisotropic parameters.

use serde::{Deserialize, Serialize};

/// One anisotropic Gaussian — the `splats-v2` extension of the v1
/// `pointcloud::GaussianSplat`. The leading four fields are byte-compatible
/// with v1 consumers; `rotation` (quaternion w,x,y,z) and `radiance` (complex
/// SH/Fourier–Legendre coeffs, interleaved re,im) are additive (v1 ignores them).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianSplatV2 {
    pub center: [f32; 3],
    pub color: [f32; 3],
    pub opacity: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub radiance: Vec<f32>,
}

/// On-disk `splats-v2` document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplatsV2File {
    pub schema: String,
    pub count: usize,
    pub n_radiance_coeffs: usize,
    pub splats: Vec<GaussianSplatV2>,
}

/// A loaded, immutable baked field.
#[derive(Clone, Debug)]
pub struct RfgsField {
    doc: SplatsV2File,
}

impl RfgsField {
    /// Load and validate a `splats-v2` JSON file.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json(&data)
    }

    /// Parse and validate a `splats-v2` JSON string.
    pub fn from_json(data: &str) -> anyhow::Result<Self> {
        let doc: SplatsV2File = serde_json::from_str(data)?;
        if doc.schema != "rfgs-v2" {
            anyhow::bail!("expected schema 'rfgs-v2', got '{}'", doc.schema);
        }
        if doc.count != doc.splats.len() {
            anyhow::bail!(
                "count {} disagrees with splats array len {}",
                doc.count,
                doc.splats.len()
            );
        }
        Ok(Self { doc })
    }

    pub fn len(&self) -> usize {
        self.doc.splats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc.splats.is_empty()
    }

    pub fn n_radiance_coeffs(&self) -> usize {
        self.doc.n_radiance_coeffs
    }

    pub fn splats(&self) -> &[GaussianSplatV2] {
        &self.doc.splats
    }

    /// Edge query: index of the Gaussian whose centre is nearest `point`.
    /// Returns `None` for an empty field. The basis of cheap inference-only
    /// queries (e.g. "what is the field value near this pose").
    pub fn query_nearest(&self, point: [f32; 3]) -> Option<usize> {
        self.doc
            .splats
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                dist2(&a.center, &point)
                    .partial_cmp(&dist2(&b.center, &point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

fn dist2(a: &[f32; 3], p: &[f32; 3]) -> f32 {
    (a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2) + (a[2] - p[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> String {
        serde_json::to_string(&SplatsV2File {
            schema: "rfgs-v2".into(),
            count: 2,
            n_radiance_coeffs: 9,
            splats: vec![
                GaussianSplatV2 {
                    center: [0.0, 0.0, 0.0],
                    color: [1.0, 0.0, 0.0],
                    opacity: 0.5,
                    scale: [0.1, 0.1, 0.1],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                    radiance: vec![1.0, 0.0],
                },
                GaussianSplatV2 {
                    center: [2.0, 0.0, 0.0],
                    color: [0.0, 1.0, 0.0],
                    opacity: 0.8,
                    scale: [0.2, 0.2, 0.2],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                    radiance: vec![0.5, 0.5],
                },
            ],
        })
        .unwrap()
    }

    #[test]
    fn loads_valid_v2_field() {
        let f = RfgsField::from_json(&sample_doc()).expect("valid field must load");
        assert_eq!(f.len(), 2);
        assert_eq!(f.n_radiance_coeffs(), 9);
        assert_eq!(f.splats()[1].color, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn rejects_wrong_schema() {
        let bad = sample_doc().replace("rfgs-v2", "rfgs-v9");
        assert!(RfgsField::from_json(&bad).is_err());
    }

    #[test]
    fn rejects_count_mismatch() {
        let bad = sample_doc().replace("\"count\":2", "\"count\":5");
        assert!(RfgsField::from_json(&bad).is_err());
    }

    #[test]
    fn query_nearest_picks_closest_centre() {
        let f = RfgsField::from_json(&sample_doc()).unwrap();
        assert_eq!(f.query_nearest([0.1, 0.0, 0.0]), Some(0));
        assert_eq!(f.query_nearest([1.9, 0.0, 0.0]), Some(1));
    }
}
