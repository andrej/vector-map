# Map-preparer

## Prerequisites

 * Protobuf compiler
   ```
   brew install protobuf
   ```

## Build

Build using `make map-preparer` from top-level directory.
This will download the protobuf specifications for OSM PBF files if not already there.
It will then invoke `cargo build -p map-preparer`, so once the OSM protobuf files are
there, you can also invoke that command directly instead.

# Webapp

## Build

```
wasm-pack build --target web
```

This will perform (roughly) the following actions:

- `cargo build --target wasm32-unknown-unknown` (builds wasm)
- `wasm-bindgen` (adds wrapper JavaScript bindings around the previously built wasm)
- `wasm-opt`
- creates a JS package out of the code in the pkg directory

(see https://www.reddit.com/r/rust/comments/kd22u5/wasmpack_dissectionhow_to_work_with_wasmbindgen/)

The advantage is that wasm-pack ensures the wasm-bindgen crate and the command-line tool to generate the bindings are in sync version-wise. 

## Deploy/run:

Start a Python web server 
