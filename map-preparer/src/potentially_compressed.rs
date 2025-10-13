use std::collections::VecDeque;
use std::io::{Read, Seek};
use flate2::read::ZlibDecoder;

pub struct PotentiallyCompressedStream<R: Read> {
    pub bytes: VecDeque<u8>,
    stream: PotentiallyCompressedStreamState<R>,
    compressed_stream_remaining: usize,
}

pub enum PotentiallyCompressedStreamState<R: Read> {
    Uninitialized,
    Uncompressed(R),
    ZlibCompressed(ZlibDecoder<R>),
}

#[derive(Debug)]
pub enum CompressionType {
    None,
    Zlib,
    Lzma,
    ObsoleteBzip2,
    Lz4,
    Zstd,
}

const READ_CHUNK_SIZE: usize = 8192;

impl<R: Read> PotentiallyCompressedStream<R> {
    pub fn from_uncompressed(stream: R) -> Self {
        Self {
            bytes: VecDeque::with_capacity(READ_CHUNK_SIZE),
            stream: PotentiallyCompressedStreamState::Uncompressed(stream),
            compressed_stream_remaining: 0,
        }
    }

    /// Ensures that `self.bytes` contains at least `n` bytes, reading from the stream as needed.
    pub fn ensure_bytes(&mut self, n: usize) -> std::io::Result<()> {
        if self.compressed_stream_remaining > 0 && n > self.compressed_stream_remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Not enough bytes in compressed stream (have {}, need {})",
                    self.compressed_stream_remaining, n
                ),
            ));
        }
        while self.bytes.len() < n {
            let n_remaining = n - self.bytes.len();
            let n_to_read = std::cmp::max(n_remaining, READ_CHUNK_SIZE);
            let mut buf = vec![0u8; n_to_read];
            let n_read = self.read(&mut buf)?;
            if self.compressed_stream_remaining > 0 {
                self.compressed_stream_remaining -= n_read;
                if self.compressed_stream_remaining == 0 {
                    self.switch_compression(CompressionType::None, 0);
                }
            }
            if n_read == 0 {
                break; // EOF
            }
            self.bytes.extend(buf.into_iter());
        }
        if self.bytes.len() < n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof, 
                format!("Not enough bytes in stream (have {}, need{})", self.bytes.len(), n))
            );
        }
        Ok(())
    }

    /// Discards the first `m` bytes from `self.bytes`.
    pub fn discard_bytes(&mut self, m: usize) -> std::io::Result<()> {
        if self.bytes.len() < m {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Not enough bytes to discard (have {}, attempting to discard {})",
                    self.bytes.len(),
                    m
                ),
            ));
        }
        self.bytes.drain(0..m);
        Ok(())
    }

    /// Switch compression type.
    pub fn switch_compression(&mut self, kind: CompressionType, n: usize) {
        // A potential future optimization would be to decompress the already-read bytes and writing them back to self.bytes here, rather than discarding them.
        self.bytes.clear();
        self.compressed_stream_remaining = n;
        match kind {
            CompressionType::None => {
                self.stream = std::mem::take(&mut self.stream).into_uncompressed();
            }
            CompressionType::Zlib => {
                self.stream = std::mem::take(&mut self.stream).into_zlib_compressed();
            }
            _ => panic!("Unsupported compression type"),
        }
    }
}

impl<R: Read> Read for PotentiallyCompressedStream<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match &mut self.stream {
            PotentiallyCompressedStreamState::Uncompressed(reader) => reader.read(buf),
            PotentiallyCompressedStreamState::ZlibCompressed(decoder) => decoder.read(buf),
            _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Reading from an uninitialized stream is not supported")),
        }
    }
}

impl<R: Read + Seek> Seek for PotentiallyCompressedStream<R> {
    /// Seek elsewhere in the stream, discarding already-buffered bytes.
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.bytes.clear();
        match &mut self.stream {
            PotentiallyCompressedStreamState::Uncompressed(reader) => reader.seek(pos),
            PotentiallyCompressedStreamState::ZlibCompressed(_) => {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "Seeking in a compressed stream is not supported"))
            },
            _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Seeking in an uninitialized stream is not supported")),
        }
    }
}

impl<R: Read> Default for PotentiallyCompressedStreamState<R> {
    fn default() -> Self {
        PotentiallyCompressedStreamState::Uninitialized
    }
}

impl<R: Read> PotentiallyCompressedStreamState<R> {
    pub fn into_zlib_compressed(self) -> PotentiallyCompressedStreamState<R> {
        if let PotentiallyCompressedStreamState::Uncompressed(reader) = self {
            return PotentiallyCompressedStreamState::ZlibCompressed(ZlibDecoder::new(reader))
        }
        panic!("into_zlib_compressed called on a stream that is not Uncompressed");
    }

    pub fn into_uncompressed(self) -> PotentiallyCompressedStreamState<R> {
        if let PotentiallyCompressedStreamState::ZlibCompressed(decoder) = self {
            return PotentiallyCompressedStreamState::Uncompressed(decoder.into_inner())
        }
        panic!("into_uncompressed called on a stream that is not ZlibCompressed");
    }
}
