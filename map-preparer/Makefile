

OSM_COMMIT_HASH=ecf26bf6a3b9eec35896167e17764fb34d13b133

.PHONY: main

main: proto/fileformat.proto proto/osmformat.proto

proto/%.proto:
	mkdir -p ${@D}
	wget https://raw.githubusercontent.com/openstreetmap/OSM-binary/$(OSM_COMMIT_HASH)/osmpbf/$(*).proto -O $@

clean:
	rm -rf proto