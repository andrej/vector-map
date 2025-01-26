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

// credit: anon80458984
// https://users.rust-lang.org/t/async-sleep-in-rust-wasm32/78218/5
pub async fn sleep(delay: i32) {
    let mut cb = 
        | resolve: wasm_bindgen_futures::js_sys::Function,
          reject: wasm_bindgen_futures::js_sys::Function | 
        {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, delay)
                .expect("unable to use JS setTimeout");
        };

    let p = wasm_bindgen_futures::js_sys::Promise::new(&mut cb);

    wasm_bindgen_futures::JsFuture::from(p).await.unwrap();
}