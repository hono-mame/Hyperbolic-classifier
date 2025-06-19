module TreeBuilder
  ( buildParentChildMap
  , findRoots
  , buildTreeSafe
  , WordPair
  , ChildMap
  ) where

import Data.Tree (Tree(..))
import qualified Data.Map as Map
import qualified Data.Set as Set

type WordPair = (String, String)
type ChildMap  = Map.Map String [String]

buildParentChildMap :: 
    [WordPair] ->   -- input list of word pairs
    ChildMap -- output map from parent to list of children
buildParentChildMap pairs = Map.fromListWith (++) [(p, [c]) | (p, c) <- pairs]
{-
("A", ["B"]) ("A", ["C"]) ("A", ["D"]) → ("A", ["B", "C", "D"])
-}

findRoots :: 
    [WordPair] ->   -- input list of word pairs
    [String]  -- output list of root nodes
findRoots pairs =
  let parents = Set.fromList [p | (p, _) <- pairs] -- collect all parents
      children = Set.fromList [c | (_, c) <- pairs] -- collect all children
  in Set.toList (parents Set.\\ children) -- find roots by subtracting children from parents

buildTreeSafe :: 
    ChildMap ->  -- input map from parent to list of children
    String ->   -- root node
    Tree String -- output tree structure
buildTreeSafe treeMap root = go Set.empty root
  where
    go visited w
      | w `Set.member` visited = Node w []
      | otherwise = Node w (map (go (Set.insert w visited)) (Map.findWithDefault [] w treeMap))
