# Where I left off

Working implementation that decodes up to the IDs of dense nodes.

Issue at about ~14% of the file, decoding fails with an invalid field (field number 0).

# Next steps

 - Decode remainder of fields in decode_dense_node_field
 - Rewrite dense node decoder to use offsets into ids, lats, lons, kv, denseinfo fields and decode on the fly rather than append to a vector;
   this should be cheap (without seeking) since the dense nodes are in-memory already anyways
 - Fix the decoding error

# Tests

- [ ] Add test for Blob start decoding when the "raw_size" field is not at the start
