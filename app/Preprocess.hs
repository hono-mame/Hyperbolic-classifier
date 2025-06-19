{-# LANGUAGE OverloadedStrings #-}

module Preprocess (main) where

import Data.Tree (Tree(..), drawTree)
import System.IO ()
import qualified Data.Map.Strict as Map

import TreeBuilder (buildTreeSafe, buildParentChildMap)

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

main :: IO ()
main = do
  tree <- loadTree "data/tree.edges"
  putStrLn "Finished loading tree from edges file."
  putStrLn $ drawTree tree
