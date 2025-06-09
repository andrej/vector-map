

OSM_COMMIT_HASH=ecf26bf6a3b9eec35896167e17764fb34d13b133

.PHONY: map-preparer

map-preparer: proto/fileformat.proto proto/osmformat.proto
	cd .. && cargo build -p map-preparer

proto/%.proto:
	mkdir -p ${@D}
	curl -L https://raw.githubusercontent.com/openstreetmap/OSM-binary/$(OSM_COMMIT_HASH)/osmpbf/$(*).proto -o $@

clean:
	rm -rf proto