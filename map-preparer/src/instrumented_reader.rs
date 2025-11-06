// This entire file is entirely LLM-generated ...
// pretty cool!

use std::io::{self, Read, Seek, SeekFrom};

/// Default callback frequency: 8 MiB
pub const DEFAULT_FREQUENCY: u64 = 8 * 1024 * 1024;

/// A reader wrapper that invokes a callback every time an additional number of bytes
/// have been read from the underlying stream.
///
/// The callback is invoked when the cumulative number of bytes read crosses
/// multiples of `frequency`. If a single read crosses several multiples, the
/// callback is invoked once per crossed multiple, in order.
#[derive(Debug)]
pub struct InstrumentedReader<R, F>
where
    R: Read,
    F: FnMut(u64),
{
    inner: R,
    callback: F,
    frequency: u64,
    total_read: u64,
    next_threshold: u64,
}

impl<R, F> InstrumentedReader<R, F>
where
    R: Read,
    F: FnMut(u64),
{
    /// Create a new `InstrumentedReader` with the default frequency (8 MiB).
    pub fn new(inner: R, callback: F) -> Self {
        Self::with_frequency(inner, DEFAULT_FREQUENCY, callback)
    }

    /// Backwards-compatible constructor matching earlier naming in `main.rs`.
    /// Alias for `new`.
    pub fn with_callback(inner: R, callback: F) -> Self {
        Self::new(inner, callback)
    }

    /// Create a new `InstrumentedReader` with a custom `frequency` in bytes.
    /// If `frequency` is 0, the default is used.
    pub fn with_frequency(inner: R, frequency: u64, callback: F) -> Self {
        let freq = if frequency == 0 { DEFAULT_FREQUENCY } else { frequency };
        Self {
            inner,
            callback,
            frequency: freq,
            total_read: 0,
            next_threshold: freq,
        }
    }

    /// Returns the total number of bytes read so far.
    pub fn total_read(&self) -> u64 {
        self.total_read
    }

    /// Returns the frequency (in bytes) at which the callback is invoked.
    pub fn frequency(&self) -> u64 {
        self.frequency
    }

    /// Gets a reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Gets a mutable reference to the underlying reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consumes the wrapper and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    fn maybe_fire_callbacks(&mut self) {
        while self.total_read >= self.next_threshold {
            (self.callback)(self.total_read);
            // Avoid overflow in extremely long streams (practically unreachable)
            let next = self.next_threshold.saturating_add(self.frequency);
            if next <= self.next_threshold { // overflow or frequency == 0 (guarded above)
                break;
            }
            self.next_threshold = next;
        }
    }
}

impl<R, F> Read for InstrumentedReader<R, F>
where
    R: Read,
    F: FnMut(u64),
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.total_read = self.total_read.saturating_add(n as u64);
            self.maybe_fire_callbacks();
        }
        Ok(n)
    }

    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        let n = self.inner.read_vectored(bufs)?;
        if n > 0 {
            self.total_read = self.total_read.saturating_add(n as u64);
            self.maybe_fire_callbacks();
        }
        Ok(n)
    }

}

impl<R, F> Seek for InstrumentedReader<R, F>
where
    R: Read + Seek,
    F: FnMut(u64),
{
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Seeking does not alter instrumentation counters; only actual reads advance total_read.
        self.inner.seek(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn fires_at_multiples_of_frequency() {
        let data = vec![0u8; 20 * 1024 * 1024]; // 20 MiB
        let cursor = Cursor::new(data);

        let mut hits: Vec<u64> = Vec::new();
        let cb = |n: u64| hits.push(n);

        let mut reader = InstrumentedReader::with_frequency(cursor, 8 * 1024 * 1024, cb);

        // Read in 3 MiB chunks so thresholds are crossed mid-read
        let mut buf = vec![0u8; 3 * 1024 * 1024];
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 { break; }
        }

        // 20 MiB crosses 8 MiB and 16 MiB -> two callbacks
        assert_eq!(hits.len(), 2);
        // The total_read values at callback time are >= thresholds
        assert!(hits[0] >= 8 * 1024 * 1024);
        assert!(hits[1] >= 16 * 1024 * 1024);
    }

    #[test]
    fn default_frequency_used_when_zero() {
        let data = vec![0u8; (DEFAULT_FREQUENCY as usize) + 1];
        let cursor = Cursor::new(data);
        let mut called = false;
        let mut reader = InstrumentedReader::with_frequency(cursor, 0, |_| { called = true; });
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 { break; }
        }
        assert!(called);
    }

    #[test]
    fn supports_new_with_default_frequency() {
        let data = vec![0u8; (DEFAULT_FREQUENCY as usize) + 10];
        let cursor = Cursor::new(data);
        let mut count = 0u32;
        let mut reader = InstrumentedReader::new(cursor, |_| { count += 1; });
        let mut buf = vec![0u8; 128 * 1024];
        io::copy(&mut reader, &mut io::sink()).unwrap();
        assert!(count >= 1);
    }
}
