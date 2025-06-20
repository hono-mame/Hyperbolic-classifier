{-# LANGUAGE OverloadedStrings #-}

module DataLoading (main) where

import DBAccess (fetchWordPairs)
import TreeBuilder (buildParentChildMap, findRoots, buildTreeSafe)
import FileExport (saveTreeAsShow, saveTreeAsDOT, saveTreeAsPNG, saveTreeAsEdgeList)
import Data.Tree (Tree(..), drawTree)

main :: IO ()
main = do
  results <- fetchWordPairs
  let treeMap = buildParentChildMap results
      roots = findRoots results
  case roots of
    [] -> putStrLn "ルートが見つかりません。"
    _  -> do
      let rootTrees = map (buildTreeSafe treeMap) roots
          unifiedTree = Node "entity" rootTrees

      putStrLn $ drawTree unifiedTree
      saveTreeAsShow "data/tree_show.txt" unifiedTree
      saveTreeAsDOT "data/tree.dot" unifiedTree
      saveTreeAsEdgeList "data/tree.edges" unifiedTree
      saveTreeAsPNG "data/tree.dot" "data/tree.png"
      putStrLn "ツリーを保存しました（.txt, .dot, .edges, .png)。"
