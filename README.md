## To build:

```
wasm-pack build --target web
```

This will perform (roughly) the following actions:

- cargo build --target wasm32-unknown-unknown (builds wasm)
- wasm-bindgen (adds wrapper JavaScript bindings around the previously built wasm)
- wasm-opt
- creates a JS package out of the code in the pkg directory

(see https://www.reddit.com/r/rust/comments/kd22u5/wasmpack_dissectionhow_to_work_with_wasmbindgen/)

The advantage is that wasm-pack ensures the wasm-bindgen crate and the command-line tool to generate the bindings are in sync version-wise. 

## To deploy/run:

Start a Python web server 
