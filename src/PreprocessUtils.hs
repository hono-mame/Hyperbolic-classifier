module PreprocessUtils
  ( parseEdge
  , loadEdges
  , loadTree
  ) where

import Data.Tree (Tree(..))
import qualified Data.Map.Strict as Map
import System.IO (readFile)
import TreeBuilder (buildParentChildMap, buildTreeSafe)

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