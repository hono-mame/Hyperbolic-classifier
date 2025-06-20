module PreprocessUtils
  ( parseEdge
  , loadEdges
  , loadTree
  , pathToRoot
  , distanceBetween
  ) where

import Data.Tree (Tree(..))
import qualified Data.Map.Strict as Map
import System.IO (readFile)
import TreeBuilder (buildParentChildMap, buildTreeSafe)

type WordPair = (String, String)
type ChildMap = Map.Map String [String]
type ParentMap = Map.Map String String

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