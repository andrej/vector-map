
.PHONY: webapp
webapp:
	wasm-pack build --target web webapp

.PHONY: clean
clean:
	rm -rf target webapp/pkg