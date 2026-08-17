import json
import zlib

path = r"C:\Users\jiaryang\OneDrive - Advanced Micro Devices Inc\2_task\61_sglang_glm\2_8k_profile_MTP\glm_mtp_glm_mi355_i8192_c4-1785988031.394309-TP-0-DECODE.trace.json.gz"
with open(path, "rb") as f:
    data = f.read()
text = zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(data)
print("decompressed GB", len(text) / 1e9)
for s in ['"name": "draft"', "TARGET_VERIFY", '"name": "draft_extend"']:
    print(s, text.count(s))
