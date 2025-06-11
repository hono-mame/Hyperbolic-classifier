module Tree (numNodes) where 

-- Definition of multi-way tree
data Tree a = Node a [Tree a]
  deriving (Eq, Show)

-- Count the number of nodes in the tree
numNodes :: Tree a -> Int 
numNodes (Node _ children) = 1 + sum (map numNodes children)
{-
ghci> numNodes (Node "Bird" [])
1
ghci> numNodes (Node "Bird" [Node "Penguin" []])
2
-}