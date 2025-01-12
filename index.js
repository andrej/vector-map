import wasm_init, * as wasm from "./pkg/first_wasm.js";

await wasm_init();

const pre = document.getElementById("game-of-life-canvas");
const universe = wasm.Universe.new();

const renderLoop = () => {
    pre.textContent = universe.render();
    universe.tick();
    requestAnimationFrame(renderLoop);
};

requestAnimationFrame(renderLoop);