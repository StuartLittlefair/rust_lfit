use pyo3::prelude::*;

pub mod errors;

pub mod donor;
pub use donor::Donor;

pub mod whitedwarf;
pub use whitedwarf::Whitedwarf;

pub mod brightspot;
pub use brightspot::Brightspot;

pub mod disc;
pub use disc::Disc;

pub mod blink;
pub use blink::blink;

pub mod solve_triads;
pub use solve_triads::*;

#[pymodule]
#[pyo3(name = "rust")]
fn lfit_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Donor>()?;
    m.add_class::<Whitedwarf>()?;
    m.add_class::<Brightspot>()?;
    m.add_class::<Disc>()?;
    m.add_function(wrap_pyfunction!(solve_triads::findi, m)?)?;
    m.add_function(wrap_pyfunction!(solve_triads::findq, m)?)?;
    m.add_function(wrap_pyfunction!(solve_triads::findphi, m)?)?;
    Ok(())
}
