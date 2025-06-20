module TreeUtils
  ( buildParentChildMap
  , buildParentMap
  , findRoots
  , buildTreeSafe
  , parseEdge
  , loadEdges
  , loadTree
  , pathToRoot
  , distanceBetween
  , WordPair
  , ChildMap
  ) where

import Data.Tree (Tree(..))
import qualified Data.Map as Map
import qualified Data.Set as Set
import System.IO (readFile)

type WordPair = (String, String)
type ChildMap  = Map.Map String [String]
type ParentMap = Map.Map String String

buildParentChildMap :: 
    [WordPair] ->   -- input list of word pairs
    ChildMap -- output map from parent to list of children
buildParentChildMap pairs = Map.fromListWith (++) [(p, [c]) | (p, c) <- pairs]
{-
("A", ["B"]) ("A", ["C"]) ("A", ["D"]) → ("A", ["B", "C", "D"])
-}

buildParentMap :: [WordPair] -> ParentMap
buildParentMap pairs = Map.fromList [(c, p) | (p, c) <- pairs]
{-
make pairs like (child, parent) 
this is used for finding the path to root and common ancestor
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

parseEdge :: 
  String ->  -- "parent child"
  (String, String) -- (parent, child)
parseEdge line =
  let ws = words line
  in (head ws, ws !! 1)

loadEdges :: 
  FilePath -> -- path to the edges file
  IO [(String, String)] -- load edges from the file
loadEdges path = do
  contents <- readFile path
  return $ map parseEdge (filter (not . null) (lines contents))

loadTree :: 
  FilePath -> -- path to the edges file
  IO (Tree String) -- load the tree structure from the edges file
loadTree path = do
  pairs <- loadEdges path
  let cmap = buildParentChildMap pairs
      tree = buildTreeSafe cmap "entity"
  return tree

pathToRoot :: ParentMap -> String -> [String]
pathToRoot pmap node = case Map.lookup node pmap of
  Nothing -> [node]
  Just parent -> node : pathToRoot pmap parent

distanceBetween :: ParentMap -> String -> String -> Either String Int
distanceBetween pmap nodeA nodeB =
  case (Map.member nodeA pmap || nodeA == "entity", Map.member nodeB pmap || nodeB == "entity") of
    (False, _) -> Left $ "Node '" ++ nodeA ++ "' does not exist."
    (_, False) -> Left $ "Node '" ++ nodeB ++ "' does not exist."
    (True, True) ->
      let pathA = reverse $ pathToRoot pmap nodeA
          pathB = reverse $ pathToRoot pmap nodeB
          common = length $ takeWhile (uncurry (==)) $ zip pathA pathB
      in Right $ (length pathA - common) + (length pathB - common)