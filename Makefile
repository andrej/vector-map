.PHONY: map-preparer
map-preparer:
	$(MAKE) -C map-preparer map-preparer

.PHONY: webapp
webapp:
	wasm-pack build --target web webapp

.PHONY: clean
clean:
	rm -rf target webapp/pkg