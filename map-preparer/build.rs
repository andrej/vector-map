use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["proto/fileformat.proto", "proto/osmformat.proto"], &["proto/"])?;
    Ok(())
}