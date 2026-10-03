# Error Analysis

## Dense retrieval better than sparse — q016

**Query type:** image_text
**Query:** similar style, but in black and not for dining

**Top results before reranking:** #1 `B07Y5XQ2LS|amazon.in` rel=0; #2 `B07YZT1DGN|amazon.com` rel=1; #3 `B07Y613GT7|amazon.co.uk` rel=0; #4 `B008U4QI18|amazon.com` rel=1; #5 `B07Y613GT3|amazon.co.uk` rel=0

**Top results after reranking:** #1 `B087X5M5LJ|amazon.com` rel=0; #2 `B07SVJB925|amazon.co.uk` rel=0; #3 `B07SGQ2W3J|amazon.co.uk` rel=0; #4 `B07M5M66HT|amazon.com` rel=0; #5 `B0822Q3VZT|amazon.in` rel=0

**Analysis:** This case is selected by the largest NDCG@10 advantage of dense over sparse retrieval. Dense NDCG@10=1.000; sparse NDCG@10=0.000; prefusion NDCG@10=0.512; final NDCG@10=0.000.

## Sparse retrieval better than dense — q007

**Query type:** text
**Query:** beige area rug with a simple design, not colorful

**Top results before reranking:** #1 `B071LQHVJZ|amazon.com` rel=2; #2 `B0794PSNTT|amazon.ca` rel=0; #3 `B07TCGFHRD|amazon.co.uk` rel=1; #4 `B075JR5KF5|amazon.co.uk` rel=0; #5 `B084H87DQZ|amazon.ae` rel=0

**Top results after reranking:** #1 `B071LQHVJZ|amazon.com` rel=2; #2 `B07TCGFHRD|amazon.co.uk` rel=1; #3 `B084H87DQZ|amazon.ae` rel=0; #4 `B07KRN2386|amazon.co.uk` rel=1; #5 `B075JR5KF5|amazon.co.uk` rel=0

**Analysis:** This case is selected by the largest NDCG@10 advantage of sparse over dense retrieval. Dense NDCG@10=0.727; sparse NDCG@10=0.865; prefusion NDCG@10=0.900; final NDCG@10=0.890.

## Cross-encoder fusion improves ranking — q005

**Query type:** text
**Query:** floor lamp for living room, not a table lamp

**Top results before reranking:** #1 `B07DBCMTPB|amazon.com` rel=2; #2 `B0825CXG8H|amazon.com` rel=0; #3 `B0742DNY41|amazon.sg` rel=2; #4 `B075X2YJHZ|amazon.com` rel=2; #5 `B07B4ZK8BR|amazon.com` rel=2

**Top results after reranking:** #1 `B07DBCMTPB|amazon.com` rel=2; #2 `B0742DNY41|amazon.sg` rel=2; #3 `B075X2YJHZ|amazon.com` rel=2; #4 `B07B51946F|amazon.com` rel=0; #5 `B07QD5SXJH|amazon.in` rel=2

**Analysis:** This case has the largest NDCG@10 gain from prefusion to the final fused reranking. Dense NDCG@10=0.968; sparse NDCG@10=0.573; prefusion NDCG@10=0.811; final NDCG@10=0.964.

## Cross-encoder fusion hurts ranking — q019

**Query type:** image_text
**Query:** similar visual but for a kitchen, not a bedroom

**Top results before reranking:** #1 `B082MLBYMV|amazon.in` rel=2; #2 `B07W4FDTH7|amazon.co.uk` rel=0; #3 `B0852ZYV7H|amazon.in` rel=2; #4 `B086213X6K|amazon.co.uk` rel=0; #5 `B07XGN4JQT|amazon.in` rel=2

**Top results after reranking:** #1 `B086213X6K|amazon.co.uk` rel=0; #2 `B07W4FDTH7|amazon.co.uk` rel=0; #3 `B07TX2D4XN|amazon.co.uk` rel=0; #4 `B07V152DQ8|amazon.co.uk` rel=0; #5 `B07DFCZ6LH|amazon.co.uk` rel=0

**Analysis:** This case has the smallest (possibly negative) NDCG@10 change after reranking. Dense NDCG@10=0.975; sparse NDCG@10=0.000; prefusion NDCG@10=0.622; final NDCG@10=0.027.

## Image or image+text case — q011

**Query type:** image
**Query:** [image query: data/raw/images/small/ce/ce04dd2a.jpg]

**Top results before reranking:** #1 `B086TGRSBG|amazon.com` rel=1; #2 `B088V8YGYM|amazon.com` rel=1; #3 `B07B4Z7MRL|amazon.com` rel=2; #4 `B07B4W2MFG|amazon.com` rel=2; #5 `B0869LMTKP|amazon.ca` rel=1

**Top results after reranking:** #1 `B086TGRSBG|amazon.com` rel=1; #2 `B088V8YGYM|amazon.com` rel=1; #3 `B07B4Z7MRL|amazon.com` rel=2; #4 `B07B4W2MFG|amazon.com` rel=2; #5 `B0869LMTKP|amazon.ca` rel=1

**Analysis:** This case illustrates visual retrieval behavior; image-only queries remain dense-only with the text cross-encoder disabled. Dense NDCG@10=0.821; sparse NDCG@10=0.000; prefusion NDCG@10=0.821; final NDCG@10=0.821.

