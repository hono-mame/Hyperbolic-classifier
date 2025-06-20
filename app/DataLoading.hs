{-# LANGUAGE OverloadedStrings #-}

module DataLoading (main) where

import DBAccess (fetchWordPairs)
import TreeUtils (buildParentChildMap, findRoots, buildTreeSafe)
import FileExport (saveTreeAsShow, saveTreeAsDOT, saveTreeAsPNG, saveTreeAsEdgeList)
import Data.Tree (Tree(..), drawTree)

main :: IO ()
main = do
  let limit = 50
  results <- fetchWordPairs limit
  let treeMap = buildParentChildMap results
      roots = findRoots results
  case roots of
    [] -> putStrLn "No root found."
    _  -> do
      let rootTrees = map (buildTreeSafe treeMap) roots
          unifiedTree = Node "entity" rootTrees

      putStrLn $ drawTree unifiedTree
      saveTreeAsShow "data/tree_show.txt" unifiedTree
      saveTreeAsDOT "data/tree.dot" unifiedTree
      saveTreeAsEdgeList "data/tree.edges" unifiedTree
      saveTreeAsPNG "data/tree.dot" "data/tree.png"
      putStrLn "Tree saved as (.txt, .dot, .edges, .png)."
