# Where I left off

Working implementation that streams data from the file and switches compression on/off throughout the file.

Architecture:
 - PartiallyCompressedStream is a struct that implements Read wrapping a buffered (BufRead) Read.
   Thus we have a buffer of (partially compressed) bytes in memory.
   Since the read interface is forward-only, the BufRead implementation can discard old bytes after returning them from `read` -- thus, the memory consumption is limited.
   Note that if we didn't pass in a BufRead (Buffered input reader), zlib's read::Decoder would do the buffering itself.
 - As soon as `into_zlib_compressed` is called, any subsequent reads interpet those bytes in memory as compressed and run the decompressor first.
   In other words, after that call, calls to `read()` may produce more bytes than they consume in the internal buffer.
   It is assumed that we know ahead of time how long the compressed section is; after N compressed bytes, compression is turned off again and bytes are returned as-is.
 - When we read a data blob, we manually decode it up until the point where we know what the compression type is.
   This is more efficient than copying the whole bob into a buffer, passing the whole buffer to the protobuf parser, and then reinterpreting the bytes in the "data" field as being compressed. 
   Instead, we read (via the decompressor) straight from the backing buffer (BufRead), and then only consume as many bytes as we need to parse sub-poritions of the messages.

# Next steps

- Nothing past `decode_blob_data()` is implemented; we need code that reads the StringTable and entities within the PrimitiveGroup.
- Right now, `decode_blob_data()` decompresses and writes the whole PrimitiveGroup into a buffer. I'd rather do something similar to what we did for the Blob and read entities in a streaming fashion (constrain reader with `take()` to the length of the primitivegroup, then keep decoding individual Nodes etc. until no data is left).

# Tests

- [ ] Add test for Blob start decoding when the "raw_size" field is not at the start
