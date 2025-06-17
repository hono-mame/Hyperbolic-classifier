module FileExport
  ( saveTreeAsShow
  , saveTreeAsDOT
  , saveTreeAsPNG
  , saveTreeAsEdgeList
  , showTreeCustom
  , toDot
  ) where

{-# LANGUAGE OverloadedStrings #-}

import Data.Tree (Tree(..))
import System.Process (callCommand)
import System.IO (withFile, IOMode(WriteMode), hSetEncoding, utf8, hPutStrLn)
import Data.List (intercalate)

showTreeCustom :: 
    Tree String -> --input tree
    String  -- customized tree output
showTreeCustom (Node n []) = "Node \"" ++ n ++ "\" []"
showTreeCustom (Node n cs) = "Node \"" ++ n ++ "\" [" ++ intercalate ", " (map showTreeCustom cs) ++ "]"
{-
output example:
Node "entity" [Node "マーティニ" [Node "ウォッカ・マティーニ" []], 
-}

saveTreeAsShow :: 
    FilePath -> --output path (.txt)
    Tree String -> --input tree
    IO ()   -- save customized tree output
saveTreeAsShow path tree = writeFile path (showTreeCustom tree)

saveTreeAsDOT :: 
    FilePath -> --output path (.dot)
    Tree String -> --input tree
    IO () -- save tree as DOT format
saveTreeAsDOT path tree = withFile path WriteMode $ \hdl -> do
  hSetEncoding hdl utf8
  hPutStrLn hdl (toDot tree)

saveTreeAsPNG :: 
    FilePath -> --input path (.dot)
    FilePath -> --output path (.png)
    IO ()   -- save tree as PNG image
saveTreeAsPNG dotFile outFile = do
  let cmd = "dot -Gcharset=utf8 -Tpng " ++ dotFile ++ " -o " ++ outFile
  callCommand cmd

toDot :: 
    Tree String ->  --input tree
    String  -- output in DOT format
toDot tree =
  "digraph Tree {\n" ++
  "  graph [fontname = \"IPAGothic\"];\n" ++
  "  node [fontname = \"IPAGothic\"];\n" ++
  "  edge [fontname = \"IPAGothic\"];\n" ++
  concatMap toDotEdges (flattenEdges tree) ++
  "}\n"
  where
    flattenEdges (Node x cs) = concatMap (\c@(Node y _) -> (x, y) : flattenEdges c) cs
    toDotEdges (p, c) = "  \"" ++ p ++ "\" -> \"" ++ c ++ "\";\n"

saveTreeAsEdgeList :: 
    FilePath -> --output path (.edges)
    Tree String -> --input tree
    IO ()   -- save tree as edge list
saveTreeAsEdgeList path tree = writeFile path (unlines $ flattenEdges tree)
  where
    flattenEdges (Node x cs) = concatMap (\c@(Node y _) -> (x ++ "\t" ++ y) : flattenEdges c) cs
