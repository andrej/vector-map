import wasm_init, * as wasm from "./pkg/first_wasm.js";

await wasm_init();

wasm.main();