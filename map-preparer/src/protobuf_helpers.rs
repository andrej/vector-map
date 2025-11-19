use std::io::{BufRead, Read};

#[derive(Debug)]
pub enum WireType {
    Varint(u64),
    Fixed64(u64),
    LengthDelimited(u64),
    StartGroup,
    EndGroup,
    Fixed32(u32),
}

/// Decode a protobuf field key and its value (for non-length-delimited types we fully
/// consume the value; for length-delimited we consume only the length varint and leave
/// the reader positioned at the start of the data).
///
/// On EOF before any bytes were read this returns Ok(None). On success returns Ok(Some(bytes_consumed)).
pub fn decode_field<R: BufRead>(
    stream: &mut R,
    out_field_number: &mut u32,
    out_wire: &mut WireType,
) -> Result<Option<usize>, std::io::Error> {
    // Read key varint
    let mut key = 0u64;
    let key_bytes = match decode_varint(stream, &mut key)? {
        Some(k) => k,
        None => return Ok(None),
    };
    let field_number = (key >> 3) as u32;
    let wire_type = (key & 0x07) as u8;

    let mut total = key_bytes;

    match wire_type {
        0 => {
            // varint
            let mut v = 0u64;
            let v_bytes = match decode_varint(stream, &mut v)? {
                Some(b) => b,
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "EOF while reading varint value",
                    ))
                }
            };
            *out_wire = WireType::Varint(v);
            total += v_bytes;
        }
        1 => {
            // 64-bit
            let mut v = 0u64;
            let v_bytes = decode_fixed64(stream, &mut v)?;
            *out_wire = WireType::Fixed64(v);
            total += v_bytes;
        }
        2 => {
            // length-delimited: read length (varint) and leave data unread
            let mut len = 0u64;
            let len_bytes = match decode_varint(stream, &mut len)? {
                Some(b) => b,
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "EOF while reading length-delimited length",
                    ))
                }
            };
            *out_wire = WireType::LengthDelimited(len);
            total += len_bytes;
        }
        3 => {
            *out_wire = WireType::StartGroup;
        }
        4 => {
            *out_wire = WireType::EndGroup;
        }
        5 => {
            // 32-bit
            let mut v = 0u32;
            let v_bytes = decode_fixed32(stream, &mut v)?;
            *out_wire = WireType::Fixed32(v);
            total += v_bytes;
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unknown wire type",
            ))
        }
    }

    *out_field_number = field_number;
    Ok(Some(total))
}

/// Decodes a protobuf varint from the stream into `out`.
/// Returns Ok(None) if EOF was encountered before any byte was read.
/// Otherwise returns Ok(Some(bytes_consumed)).
/// varint encoding is used for int32, int64, uint32, uint64, bool, enum.
#[inline]
pub fn decode_varint<R: BufRead>(
    stream: &mut R,
    out: &mut u64,
) -> Result<Option<usize>, std::io::Error> {
    let mut value = 0u64;
    let mut shift = 0;
    let mut consumed = 0usize;

    loop {
        let available = stream.fill_buf()?;
        if available.is_empty() {
            if consumed == 0 {
                return Ok(None);
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "EOF reached in the middle of varint",
                ));
            }
        }

        let mut consumed_this_iter = 0;
        let mut broke = false;
        for byte in available {
            consumed_this_iter += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                broke = true;
                break;
            }
            shift += 7;
        }

        stream.consume(consumed_this_iter);
        consumed += consumed_this_iter;

        if broke {
            *out = value;
            return Ok(Some(consumed));
        } else if shift > 63 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Malformed varint: too long",
            ));
        }
    }
}

/// Apply zigzag decoding to an already-read varint value.
/// This is useful when you have a u64 varint value from decode_field that needs zigzag decoding.
#[inline]
pub fn zigzag_decode(raw: u64) -> i64 {
    // Optimized zigzag decoding: (n >> 1) ^ -(n & 1)
    // Equivalent to: if n & 1 == 0 { n >> 1 } else { !((n >> 1)) }
    ((raw >> 1) ^ (raw & 1).wrapping_neg()) as i64
}

/// Decode a zigzag-encoded varint from the stream into `out`.
/// zigzag-encoded varints are used for sint32 and sint64 types.
#[inline]
pub fn decode_zigzag_varint<R: BufRead>(
    stream: &mut R,
    out: &mut i64,
) -> Result<usize, std::io::Error> {
    let mut raw = 0u64;
    let n_bytes = match decode_varint(stream, &mut raw)? {
        Some(b) => b,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF while reading zigzag varint",
            ))
        }
    };
    *out = zigzag_decode(raw);
    Ok(n_bytes)
}

/// Read 8 bytes (little-endian) into `out` and return number of bytes consumed (8) on success.
#[inline]
pub fn decode_fixed64<R: Read>(stream: &mut R, out: &mut u64) -> Result<usize, std::io::Error> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;
    *out = u64::from_le_bytes(buf);
    Ok(8)
}

/// Read 4 bytes (little-endian) into `out` and return number of bytes consumed (4) on success.
#[inline]
pub fn decode_fixed32<R: Read>(stream: &mut R, out: &mut u32) -> Result<usize, std::io::Error> {
    let mut buf: [u8; 4] = [0u8; 4];
    stream.read_exact(&mut buf)?;
    *out = u32::from_le_bytes(buf);
    Ok(4)
}
