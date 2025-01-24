pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[macro_export]
macro_rules! console_log {
    ($fmt:expr $(, $args:expr)*) => {
        web_sys::console::log_1(&format!($fmt, $($args),*).into())
    }
}

// see https://rustwasm.github.io/wasm-bindgen/examples/performance.html
fn perf_to_system(amt: f64) -> std::time::SystemTime {
    let secs = (amt as u64) / 1_000;
    let nanos = (((amt as u64) % 1_000) as u32) * 1_000_000;
    std::time::UNIX_EPOCH + std::time::Duration::new(secs, nanos)
}