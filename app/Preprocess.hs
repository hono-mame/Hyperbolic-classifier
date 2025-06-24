{-# LANGUAGE OverloadedStrings #-}

module Preprocess (main) where

import Data.Tree (Tree(..), drawTree)
import qualified Data.Map.Strict as Map
import TreeUtils (buildTreeSafe, buildParentChildMap, buildParentMap, loadEdges, distanceBetween)
import FileExport (saveTreeAsDOT, saveTreeAsPNG) 


main :: IO ()
main = do
  -- pairs <- loadEdges "data/tree_test.edges"
  pairs <- loadEdges "data/tree.edges"
  let cmap = buildParentChildMap pairs
      pmap = buildParentMap pairs
      tree = buildTreeSafe cmap "entity"

  -- for test
  -- saveTreeAsDOT "data/tree_test.dot" tree
  -- saveTreeAsPNG "data/tree_test.dot" "data/tree_test.png"
  -- putStrLn "Tree saved as PNG."

  -- putStrLn "Tree structure:"
  -- putStrLn $ drawTree tree

  putStrLn "---------- check ----------"
  let nodeA = "目"
      nodeB = "毛"

  result <- distanceBetween pmap nodeA nodeB
  case result of
    Left errMsg -> putStrLn $ "error: " ++ errMsg
    Right dist -> putStrLn $ "Distance between " ++ nodeA ++ " and " ++ nodeB ++ ": " ++ show dist
