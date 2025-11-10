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

## Design Notes

### OSMStreamReader

The idea is to copy data as little as possible. 
For example, we'd rather read data straight out of the decompressor rather than decompress everything into a buffer, then parse the buffer.

### Architecture

 - `PartiallyCompressedStream` is a struct that implements `Read` wrapping a buffered reader (`BufRead`).
   As a consequence, what's in memory is a buffer of (partially compressed) bytes.
   The goal is for all further processing to be compute on top of those bytes in memory.
 - Since the read interface is forward-only, the BufRead implementation can discard old bytes after returning them from `read` -- thus, the memory consumption is limited.
   Note that if we didn't pass in a BufRead (Buffered input reader), zlib's read::Decoder would do the buffering itself.
 - As soon as `into_zlib_compressed` is called, any subsequent reads interpet those bytes in memory as compressed and run the decompressor first.
   In other words, after that call, calls to `read()` may produce more bytes than they consume in the internal buffer.
   It is assumed that we know ahead of time how long the compressed section is; after N compressed bytes, compression is turned off again and bytes are returned as-is.
 - When we read a data blob, we manually decode it up until the point where we know what the compression type is.
   This is more efficient than copying the whole bob into a buffer, passing the whole buffer to the protobuf parser, and then reinterpreting the bytes in the "data" field as being compressed. 
   Instead, we read (via the decompressor) straight from the backing buffer (BufRead), and then only consume as many bytes as we need to parse sub-poritions of the messages.

### Subslices into memory for `PrimitiveGroup`s and `Bytes`

Unfortunately, due to the layout of the input file, some copying is necessary for the `PrimitiveGroup`s, which are the meat and bones of the file (contain the actual nodes, ways, relations).
In order to make sense of the contained data, we need the StringTable that is encoded in the file, but this StringTable is not necessarily at the beginning of the `PrimitiveGroup`.
Rather than skip around in the file a bunch trying to find the fields we need (which would read `CHUNK_SIZE` bytes, only to use the header and then discard most of them to jump to the next field),
we buffer the groups we encounter in-memory, in the `heap` data structure.

Then, we process subslices of this `heap`.

Rather than `&[u8]`, these are of the type `Bytes`. `Bytes`, from the `bytes` crate, is a cheap, reference-counted struct that enforces that the underlying buffer is available as long as the slice is in use.

We can't use `&[u8]`, because that would requrie impossible lifetime annotations: 
The underlying heap buffer will be overwritten with new data as the parser moves on with processing, which would render any remaining slices dangling.
We'd need to express the following constraint: Subslices are valid between the exit of the `decode_blob_data_start()` call until we reach state `ReadingBlobDataState::End`, at which point the we will exit the `ReadingBlobData` state and the vector containing the `heap` will be dropped.
This is clearly a "temporal" constraint that doesn't neatly overlap with scopes in the Rust program.
However, lifetimes can only annotate regions/scopes of the program. 
Furthermore, creating the slices themselves would be impossible:
We need a mutable reference to `self` in order to be able to update `self.state`;
and we need a reference that lives for at least `'a`, where `'a` is whatever lifetime would express above temporal constraint, to reference the `heap` in `self.state` in the `&[u8]` slices.
The only way we could get that `'a` reference for the slices would be through our mutable `self` reference, so we'd need `&'a mut self`;
meaning, a mutable reference (not immutable) would be held for all of `'a`, making it impossible for anything else after the function call that sets the slices to use the struct at all.

`Bytes` works around this by enforcing the lifetime constraints at runtime.
It keeps a reference count, where each subslice increases the reference.
This has the nice effect that if a user of our stream keeps using the data after we have already moved on to process the next `PrimitiveGroup`, the bytes will still be kept in memory. 


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
