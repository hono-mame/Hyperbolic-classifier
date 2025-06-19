{-# LANGUAGE OverloadedStrings #-}

module Preprocess (main) where

import Data.Tree (Tree(..), drawTree)
import System.IO ()
import qualified Data.Map.Strict as Map

import TreeBuilder (buildTreeSafe, buildParentChildMap)
import PreprocessUtils (loadTree)

main :: IO ()
main = do
  tree <- loadTree "data/tree.edges"
  putStrLn "Finished loading tree from edges file."
  putStrLn $ drawTree tree
