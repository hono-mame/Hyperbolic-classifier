# Implementation of `poincare.hs`

## TODO
- [x] Initialize embeddings  
- [x] Output and verify initialized embedding values  
- [x] Implement function that receives two words and returns the distance (in hyperbolic space)  
- [ ] Implement training in Python & understand the original paper  


## **Embedding Initialization**
Initialize the vector for each word with random values between -0.001 and 0.001.
```python
def __init__(self, train_data, size=50, alpha=0.1, negative=10, workers=1, epsilon=1e-5, regularization_coeff=1.0,
                 burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001), dtype=np.float64, seed=0):
        """Initialize and train a Poincare embedding model from an iterable of relations."""

def _init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index_to_key), self.size)
        self.kv.vectors = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)
```

### **Output and Verify Initialized Embedding Values**
```haskell
---------- print first 10 vectors of each word ----------
事業年度: [5.387352494167752e-4,9.211452963554237e-4,-6.1567598847052e-4]
勧誘: [9.890561888538283e-4,-5.468338270131826e-4,1.5960465692145895e-4]
善: [9.91935711983976e-4,-9.567924965823912e-4,-1.353616263855659e-4]
変わる: [1.0703264384083195e-4,-3.4016494443700517e-4,1.7615049613138025e-4]
嫌疑者: [9.127147845983039e-4,3.0535400253382084e-4,-7.5581392098721e-4]
恭敬: [-2.953264256850552e-4,7.324265474919836e-4,5.149321207742212e-4]
持ち上る: [5.205066737392953e-4,-8.475991684946584e-4,7.213814462442247e-4]
捨去る: [7.938164114709872e-4,-6.802290272153776e-5,-6.380883325665841e-4]
教諭: [-3.8043288773319034e-5,8.124010512454377e-4,-3.995850819740523e-4]
気象学: [3.1860003648567716e-4,3.630507779341476e-4,-8.463863409100831e-4]
```

### **Implement Function That Receives Two Words and Returns the Distance (in Hyperbolic Space)**
```
Distance between "事業年度" and "勧誘": 3.4402425272197796e-3
```