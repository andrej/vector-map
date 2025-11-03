use std::io::{Read, BufRead, Seek};
use flate2::bufread::ZlibDecoder;


// --------------------------------------------------------------------------
// PartiallyCompressedStream
// --------------------------------------------------------------------------

pub struct PartiallyCompressedStream<R: Read + BufRead> {
    stream: PartiallyCompressedStreamReader<R>,
}

impl<R: Read + BufRead> PartiallyCompressedStream<R> {
    pub fn from_uncompressed(stream: R) -> Self {
        Self {
            stream: PartiallyCompressedStreamReader::Uncompressed(stream),
        }
    }

    pub fn enable_zlib_compression(&mut self, compressed_len: usize) {
        self.stream = std::mem::take(&mut self.stream).into_zlib_compressed(compressed_len);
    }

    pub fn disable_compression(&mut self) {
        self.stream = std::mem::take(&mut self.stream).into_uncompressed();
    }
}

impl<R: Read + BufRead> Default for PartiallyCompressedStreamReader<R> {
    fn default() -> Self {
        PartiallyCompressedStreamReader::Uninitialized
    }
}

impl<R: Read + BufRead> Read for PartiallyCompressedStream<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut n_read_total = 0;
        let mut remaining_buf = &mut buf[..];
        while remaining_buf.len() > 0 {
            let n_read = match &mut self.stream {
                PartiallyCompressedStreamReader::Uncompressed(reader) => reader.read(remaining_buf),
                PartiallyCompressedStreamReader::ZlibCompressed(decoder) => decoder.read(remaining_buf),
                _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Reading from an uninitialized stream is not supported")),
            }?;
            if n_read == 0 {
                if let PartiallyCompressedStreamReader::ZlibCompressed(_) = &self.stream {
                    // Reached the end of the compressed part of the stream ...
                    self.disable_compression();
                    // ... but there may still be more data in the uncompressed part
                    continue;
                }
                // EOF in uncompressed stream, meaning end of stream -- we should forward the EOF signal
                break; 
            }
            remaining_buf = &mut remaining_buf[n_read..];
            n_read_total += n_read;
        }
        Ok(n_read_total)
    }
}

impl<R: Read + BufRead + Seek> Seek for PartiallyCompressedStream<R> {
    /// Seeking is only supported in the uncompressed part of the stream.
    /// When in a compressed part, it is only allowed to seek by an amount of compressed bytes that leads outside of the compressed region;
    /// however, this requirement is not checked or enforced. You can end up in a broken state (reading compressed bytes as uncompressed) if you violate it.
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        if let PartiallyCompressedStreamReader::ZlibCompressed(decoder) = &self.stream {
            self.disable_compression();
        }
        match &mut self.stream {
            PartiallyCompressedStreamReader::Uncompressed(reader) => reader.seek(pos),
            _ => panic!("Can only seek in uncompressed stream"),
        }
    }
}


// --------------------------------------------------------------------------
// PartiallyCompressedStreamReader
// --------------------------------------------------------------------------

pub enum PartiallyCompressedStreamReader<R: Read + BufRead> {
    Uninitialized,
    Uncompressed(R),
    ZlibCompressed(ZlibDecoder<std::io::Take<R>>),
}

impl<R: Read + BufRead> PartiallyCompressedStreamReader<R> {
    pub fn into_zlib_compressed(self, compressed_len: usize) -> PartiallyCompressedStreamReader<R> {
        if let PartiallyCompressedStreamReader::Uncompressed(reader) = self {
            return PartiallyCompressedStreamReader::ZlibCompressed(ZlibDecoder::new(reader.take(compressed_len as u64)))
        }
        panic!("into_zlib_compressed called on a stream that is not Uncompressed");
    }

    pub fn into_uncompressed(self) -> PartiallyCompressedStreamReader<R> {
        if let PartiallyCompressedStreamReader::ZlibCompressed(decoder) = self {
            let inner_limited: std::io::Take<R> = decoder.into_inner();
            return PartiallyCompressedStreamReader::Uncompressed(inner_limited.into_inner())
        }
        panic!("into_uncompressed called on a stream that is not ZlibCompressed");
    }
}