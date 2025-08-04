# Research on Pre-trained Embeddings
When implementing an MLP using pre-trained embeddings, I investigated what kinds of embeddings are available and which ones are optimal.

List of ready-to-use word embedding vectors:
https://qiita.com/Hironsan/items/8f7d35f0a36e0f99752c

### **Word2Vec**
- Source: [word2vec](https://github.com/Kyubyong/wordvectors)  
This is probably the most standard option?
- Vocabulary size: 50108   
- Vector dimension: 300


### **fastText** 
- Source: [fastText Crawl Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- Vocabulary size: 2,000,000
- Vector dimension: 300
- Impression: The vocabulary contained many noisy or poor-quality entries...


### **Japanese Wikipedia Entity Vectors**
- Source: [Jawiki Vector](https://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/)
- Vocabulary size: 1,015,474
- Vector dimension: 200
- Impression: This dataset seemed to have higher quality vocabulary, so I chose to use this.   
