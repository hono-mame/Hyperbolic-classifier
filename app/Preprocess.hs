{-# LANGUAGE OverloadedStrings #-}

module Preprocess (main) where

import Data.Tree (Tree(..), drawTree)
import TreeBuilder (buildTreeSafe, buildParentChildMap, buildParentMap)
import PreprocessUtils (loadEdges, distanceBetween)
import qualified Data.Map.Strict as Map

main :: IO ()
main = do
  pairs <- loadEdges "data/tree.edges"
  let cmap = buildParentChildMap pairs
      pmap = buildParentMap pairs
      tree = buildTreeSafe cmap "entity"

  putStrLn "ツリー構造をロードしました!"
  putStrLn $ drawTree tree

  let nodeA = "チェッカー"
      nodeB = "アタッチメント"

  case distanceBetween pmap nodeA nodeB of
    Left errMsg -> putStrLn $ "error: " ++ errMsg
    Right dist -> putStrLn $ "Distance between " ++ nodeA ++ " and " ++ nodeB ++ ": " ++ show dist
