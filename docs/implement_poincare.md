# Implementation of `poincare.hs`

## TODO
### Preprocessing
- [x] Initialize embeddings  
- [x] Output and verify initialized embedding values  
- [x] Implement function that receives two words and returns the distance (in hyperbolic space)  

### data preparation
- [x] Change the input format
  (hyper, hypo, label) -> (hyper, hypo) 
  do negative sampling within the training loop.
- [x] Write a script to prepare input data specifically for Hyperbolic Embedding
- [x] generate negative samples for each positive sample

### Training
- [x] Understand the implementation of training in Python & the original paper's logic    
→ detail:[Poincare_training](/docs/Poincare_training.pdf)  
- [x] Implement gradient conversion function (riemannianGradient)
- [x] Implement projection function (projectToBall)
- [x] Create a template for the training loop (with some dummy functions)
  Placeholder for loss calculation and parameter updates
- [x] implement RSGD version of runStep function (if possible)

### Others
- [x] Modify existing functions to use Tensor


### **Embedding Initialization**
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

### **Understand the training part**
→ detail:[Poincare_training](/docs/Poincare_training.pdf)  

### **implement the training part**
implemented two functions(will be needed during training):
```haskell
riemannianGradient :: Tensor -> Tensor -> Tensor
riemannianGradient thetaT grad =
  let normSquared = sumAll (thetaT * thetaT)
      coeff = pow (2.0 :: Float) (asTensor (1.0 :: Float) - normSquared) / asTensor (4.0 :: Float)
  in coeff * grad

projectToBall :: Tensor -> Tensor
projectToBall thetaT =
  let norm = Torch.Functional.sqrt (sumAll (thetaT * thetaT))
      normScalar = asValue norm :: Float
      eps = 1e-5
  in if normScalar > 1.0 && normScalar > eps
        then thetaT / asTensor (normScalar - eps)
        else thetaT
```


### **Create a template for the training loop**
after finish implementhing the trainig part, the value of losses should become smaller.
```haskell
Initial embeddings:
---------- print first 10 vectors of each word ----------
善: Tensor Float [3] [ 3.1243e-4, -2.9358e-4,  5.1586e-4]
恭敬: Tensor Float [3] [-6.7423e-4, -1.1448e-3, -2.7294e-4]
持ち上る: Tensor Float [3] [-5.9734e-4,  1.5078e-3, -1.5180e-3]
温かさ: Tensor Float [3] [ 2.1758e-5,  5.1795e-4, -9.1720e-4]
行なわれる: Tensor Float [3] [ 7.7263e-4,  1.4837e-3, -1.0696e-3]
防止: Tensor Float [3] [ 3.8586e-5, -6.1635e-4, -1.6448e-3]
Start training...
Epoch 1 | Loss = 1.5363246e-5
Epoch 2 | Loss = 1.5363246e-5
Epoch 3 | Loss = 1.5363246e-5
Epoch 4 | Loss = 1.5363246e-5
Epoch 5 | Loss = 1.5363246e-5
Epoch 6 | Loss = 1.5363246e-5
Epoch 7 | Loss = 1.5363246e-5
Epoch 8 | Loss = 1.5363246e-5
Epoch 9 | Loss = 1.5363246e-5
Epoch 10 | Loss = 1.5363246e-5
Training finished.
---------- print first 10 vectors of each word ----------
善: Tensor Float [3] [ 3.1243e-4, -2.9358e-4,  5.1586e-4]
恭敬: Tensor Float [3] [-6.7423e-4, -1.1448e-3, -2.7294e-4]
持ち上る: Tensor Float [3] [-5.9734e-4,  1.5078e-3, -1.5180e-3]
温かさ: Tensor Float [3] [ 2.1758e-5,  5.1795e-4, -9.1720e-4]
行なわれる: Tensor Float [3] [ 7.7263e-4,  1.4837e-3, -1.0696e-3]
防止: Tensor Float [3] [ 3.8586e-5, -6.1635e-4, -1.6448e-3]
```

### **Finished implementing the training part**
result(with small test data, dummy relationships)
``` haskell
Initial embeddings:
---------- print first 10 vectors of each word ----------
A: Tensor Float [3] [-2.0117e-4,  1.9466e-3,  8.7405e-4]
B: Tensor Float [3] [-2.5283e-4, -4.1555e-4,  8.2789e-4]
C: Tensor Float [3] [-4.5273e-4, -1.3752e-3, -2.5581e-4]
D: Tensor Float [3] [-4.2155e-4,  2.7575e-7,  1.0969e-3]
E: Tensor Float [3] [ 1.1763e-3,  4.6135e-4, -9.8919e-4]
F: Tensor Float [3] [ 9.0901e-4, -2.0377e-5,  1.2742e-3]
M: Tensor Float [3] [-8.1056e-4,  8.2273e-4, -1.7868e-3]
N: Tensor Float [3] [ 4.8089e-5, -1.0440e-3, -4.9018e-4]
X: Tensor Float [3] [ 1.5222e-3, -1.1042e-3, -2.1042e-4]
Y: Tensor Float [3] [ 1.1548e-3, -8.6157e-5,  1.2959e-3]
Start training...
Epoch 10  lr=9.999999e-4  Loss=35.32387
Epoch 20  lr=1.0e-2  Loss=33.438755
Epoch 30  lr=1.0e-2  Loss=31.608467
Epoch 40  lr=1.0e-2  Loss=30.038973
Epoch 50  lr=1.0e-2  Loss=28.45166
Epoch 60  lr=1.0e-2  Loss=27.082933
Epoch 70  lr=1.0e-2  Loss=25.727005
Epoch 80  lr=1.0e-2  Loss=24.496433
Epoch 90  lr=1.0e-2  Loss=23.360289
Epoch 100  lr=1.0e-2  Loss=22.336319
Epoch 110  lr=1.0e-2  Loss=21.357977
Epoch 120  lr=1.0e-2  Loss=20.44184
Epoch 130  lr=1.0e-2  Loss=19.597368
Epoch 140  lr=1.0e-2  Loss=18.871922
Epoch 150  lr=1.0e-2  Loss=18.165676
Epoch 160  lr=1.0e-2  Loss=17.447916
Epoch 170  lr=1.0e-2  Loss=16.82014
Epoch 180  lr=1.0e-2  Loss=16.312077
Epoch 190  lr=1.0e-2  Loss=15.861533
Epoch 200  lr=1.0e-2  Loss=15.361312
Epoch 210  lr=1.0e-2  Loss=14.915988
Epoch 220  lr=1.0e-2  Loss=14.517198
Epoch 230  lr=1.0e-2  Loss=14.151439
Epoch 240  lr=1.0e-2  Loss=13.811441
Epoch 250  lr=1.0e-2  Loss=13.462194
Epoch 260  lr=1.0e-2  Loss=13.17724
Epoch 270  lr=1.0e-2  Loss=12.871571
Epoch 280  lr=1.0e-2  Loss=12.636831
Epoch 290  lr=1.0e-2  Loss=12.405084
Epoch 300  lr=1.0e-2  Loss=12.191023
Epoch 310  lr=1.0e-2  Loss=11.987133
Epoch 320  lr=1.0e-2  Loss=11.805191
Epoch 330  lr=1.0e-2  Loss=11.616636
Epoch 340  lr=1.0e-2  Loss=11.44525
Epoch 350  lr=1.0e-2  Loss=11.243036
Epoch 360  lr=1.0e-2  Loss=11.095315
Epoch 370  lr=1.0e-2  Loss=10.980642
Epoch 380  lr=1.0e-2  Loss=10.813464
Epoch 390  lr=1.0e-2  Loss=10.691256
Epoch 400  lr=1.0e-2  Loss=10.577953
Epoch 410  lr=1.0e-2  Loss=10.462139
Epoch 420  lr=1.0e-2  Loss=10.34776
Epoch 430  lr=1.0e-2  Loss=10.205788
Epoch 440  lr=1.0e-2  Loss=10.102495
Epoch 450  lr=1.0e-2  Loss=9.976488
Epoch 460  lr=1.0e-2  Loss=9.900166
Epoch 470  lr=1.0e-2  Loss=9.829764
Epoch 480  lr=1.0e-2  Loss=9.733345
Epoch 490  lr=1.0e-2  Loss=9.655635
Epoch 500  lr=1.0e-2  Loss=9.575263
Training finished.
---------- print first 10 vectors of each word ----------
A: Tensor Float [3] [ 0.1282   ,  0.7271   ,  0.3606   ]
B: Tensor Float [3] [ 0.1288   ,  0.7208   ,  0.3561   ]
C: Tensor Float [3] [ 0.1362   ,  0.6712   ,  0.3185   ]
D: Tensor Float [3] [ 0.1563   ,  0.5884   ,  0.2508   ]
E: Tensor Float [3] [ 0.1606   ,  0.5731   ,  0.2379   ]
F: Tensor Float [3] [ 0.1604   ,  0.5726   ,  0.2380   ]
M: Tensor Float [3] [ 0.6526   , -0.1064   , -0.2166   ]
N: Tensor Float [3] [ 0.6526   , -0.1065   , -0.2167   ]
X: Tensor Float [3] [ 0.7478   , -0.1334   , -0.2484   ]
Y: Tensor Float [3] [ 0.7451   , -0.1327   , -0.2476   ]
```

NEXT: Try with larger data!!


### **data preprocessing for Hyperbolic Embedding**
→ [GenerateDatasetHyperbolic.hs](/app/GenerateDatasetHyperbolic.hs)