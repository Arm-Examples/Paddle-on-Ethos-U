

import numpy as np
import struct

# ======================== utils start ========================

def read_buf(buf, start, offset, is_str=False, is_sign=False):
    end = start+offset
    if is_str == True:
        return end, buf[start:end].decode('utf-8', errors="replace")
    print(f"start  {start} end {end}")
    if offset == 1:
        unpack_type = '<B' if is_sign == False else '<b'
    elif offset == 2:
        unpack_type = '<H' if is_sign == False else '<h'
    elif offset == 4:
        unpack_type = '<I' if is_sign == False else '<i'
    elif offset == 8:
        unpack_type = '<Q' if is_sign == False else '<q'
    else:
        return end, None

    return end, struct.unpack(unpack_type, buf[start:end])[0]

def write_buf(buf, start, offset, data, is_str=False, is_sign=False, is_insert=False):
    end = start+offset

    if offset == 1:
        pack_type = '<B' if is_sign == False else '<b'
    elif offset == 2:
        pack_type = '<H' if is_sign == False else '<h'
    elif offset == 4:
        pack_type = '<I' if is_sign == False else '<i'
    elif offset == 8:
        pack_type = '<Q' if is_sign == False else '<q'
    else:
        return None, end

    b_data = struct.pack(pack_type, data)
    ba = bytearray(buf)
    if is_insert == False:
        ba[start:end] = b_data
    elif is_insert == True:
        ba[start:start] = b_data
    return bytes(ba), end



def insert_bytes(buf, src_buf, pos):
    tmp_buf = bytearray(buf)
    tmp_buf[pos:pos] = src_buf
    return bytes(tmp_buf), pos, pos+len(src_buf)

def replace_bytes(buf, src_buf, pos):
    tmp_buf = bytearray(buf)
    tmp_buf[pos:pos+len(src_buf)] = src_buf
    return bytes(tmp_buf), pos, pos+len(src_buf)

# ======================== util end ========================