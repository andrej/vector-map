# Where I left off

Working implementation that streams data from the file and switches compression on/off throughout the file.

# Next steps

- Nothing past `decode_blob_data()` is implemented; we need code that reads the StringTable and entities within the PrimitiveGroup.
- Right now, `decode_blob_data()` decompresses and writes the whole PrimitiveGroup into a buffer. I'd rather do something similar to what we did for the Blob and read entities in a streaming fashion (constrain reader with `take()` to the length of the primitivegroup, then keep decoding individual Nodes etc. until no data is left).

# Tests

- [ ] Add test for Blob start decoding when the "raw_size" field is not at the start
