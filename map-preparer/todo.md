# Where I left off

`PotentiallyCompressedStream` must be rewound before switching compression, because the ZlibDecoder will read from the input stream without considering already-read bytes in self.bytes. 
For now, I solved this problem by seeking the stream back when any bytes are discarded out of self.bytes.
When we are in compressed state switching back to uncompressed, we cannot seek, so we cannot discard bytes.
For now, this will just panic.
However, this will likely read to issues when I try to over-read (e.g., varint at the end of the compressed stream, ensure_bytes(10), but varint is shorter, now we over-read).
Better approach: Allow under-reads from ensure_bytes (or provide other method) and assume that ZlibDecoder will only give as many as it has without failing.
The stream should also probably be constrained (Buf:take()) so ZlibDecoder doesn't try to start reading into uncompressed data.
Should also not discard bytes.

# Tests

- [ ] Add test for Blob start decoding when the "raw_size" field is not at the start
