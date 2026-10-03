# Stage 2 - Retrieval Summary

- Queries by type: text=10, image=5, image_text=5 (total=20)
- Dense top-K: 100 | Sparse top-K: 100
- Prefusion method: **rrf** (RRF k=60 | weighted w_dense=0.55, w_sparse=0.45)
- Average unique candidates per query: **149.1**
- Average candidates in BOTH dense and sparse: **25.9**
- Image-only queries use dense retrieval only (`sparse_applicable=false`).

## Top results for the first 3 queries

- **q001** (text) -> #1 B086B5MRFB|amazon.in (pf=0.0328), #2 B07B4MRLT8|amazon.com (pf=0.0313), #3 B07QJXW4JR|amazon.com (pf=0.0306)
- **q002** (text) -> #1 B07J1YW3YT|amazon.com (pf=0.0325), #2 B07QC861L7|amazon.com (pf=0.0320), #3 B07QJLWJJM|amazon.ca (pf=0.0308)
- **q003** (text) -> #1 B07SSHYD2Z|amazon.co.uk (pf=0.0323), #2 B081HX9SRM|amazon.co.uk (pf=0.0323), #3 B081HWKWN6|amazon.co.uk (pf=0.0323)
