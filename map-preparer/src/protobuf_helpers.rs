// Basic decoding functions for reading protobuf messages from a byte stream (Buf).
// We use these for manually decoding protobuf messages when we need to do a partial decode, since prost does not support partially decoding messages.
use std::io::Read;
use crate::potentially_compressed::PotentiallyCompressedStream;
use bytes::Buf;

#[derive(Debug)]
pub enum WireType {
    Varint(u64),
    Fixed64(u64),
    LengthDelimited(u64),
    StartGroup,
    EndGroup,
    Fixed32(u32),
}

/// Decodes a field from a protobuf message, returning the field number and wire type.
/// For all types except length-delimited, the value is decoded. 
/// For length-delimited, we only read and return the length and leave the stream state at the beginning of the data.
pub fn decode_field<T: Read>(stream: &mut PotentiallyCompressedStream<T>) -> Result<(u32, WireType), std::io::Error> {
    let key = decode_varint(stream)?;
    let field_number = (key >> 3) as u32;
    let wire_type = (key & 0x07) as u8;
    let value = match wire_type {
        0 => WireType::Varint(decode_varint(stream)?),
        1 => WireType::Fixed64(decode_fixed64(stream)?),
        2 => {
            let len = decode_varint(stream)?;
            WireType::LengthDelimited(len)
        },
        3 => WireType::StartGroup,
        4 => WireType::EndGroup,
        5 => WireType::Fixed32(decode_fixed32(stream)?),
        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown wire type")),
    };
    Ok((field_number, value))
}

/// Decodes a protobuf varint from self.bytes.
pub fn decode_varint<T: Read>(stream: &mut PotentiallyCompressedStream<T>) -> Result<u64, std::io::Error> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        stream.ensure_bytes(1)?;
        let byte = Buf::get_u8(&mut stream.bytes);
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 { break; }
        shift += 7;
        if shift > 63 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Malformed varint: too long")); }
    }
    Ok(value)
}

pub fn decode_fixed64<T: Read>(stream: &mut PotentiallyCompressedStream<T>) -> Result<u64, std::io::Error> {
    stream.ensure_bytes(8)?;
    Ok(Buf::get_u64_le(&mut stream.bytes))
}

pub fn decode_fixed32<T: Read>(stream: &mut PotentiallyCompressedStream<T>) -> Result<u32, std::io::Error> {
    stream.ensure_bytes(4)?;
    Ok(Buf::get_u32_le(&mut stream.bytes))
}
