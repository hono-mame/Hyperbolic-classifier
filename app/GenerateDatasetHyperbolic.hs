{-# LANGUAGE OverloadedStrings #-}

module GenerateDatasetHyperbolic (main) where

import Data.List (intercalate)
import System.IO (writeFile)
import System.Random (randomRIO)
import qualified Data.Map as M
import qualified Data.Set as S
import Data.List.Split (splitOn)

type Entity = String
type Edge = (String, String)

-- 正例のペアを読み込む関数
readPositivePairsFromEdges :: FilePath -> IO [Edge]
readPositivePairsFromEdges path = do
    contents <- readFile path
    let ls = lines contents
        -- タブを区切り文字として指定
        pairs = map (splitOn "\t") ls
        -- "entity" で始まる行と、単語数が2つではない行をフィルタリング
        validPairs = filter (\p -> length p == 2 && head p /= "entity") pairs
        edges = map (\[a, b] -> (a, b)) validPairs
    return edges

-- CSV形式でデータを保存する関数
savePairsToCSV :: FilePath -> [Edge] -> IO ()
savePairsToCSV path pairs = do
    let header = "hyper,hypo"
        csvContent = unlines (header : map (\(u, v) -> u ++ "," ++ v) pairs)
    writeFile path csvContent

main :: IO ()
main = do
    let inputFilePath = "data/Hyperbolic/tree_test.edges"
        trainOutputPath = "data/Hyperbolic/train.csv"

    putStrLn "Starting dataset generation for Hyperbolic model..."

    -- 正例ペアをすべて読み込み
    allPairs <- readPositivePairsFromEdges inputFilePath
    let totalPairs = length allPairs

    putStrLn $ "  Total pairs loaded: " ++ show totalPairs
    putStrLn ""

    -- すべてのペアを訓練データとして保存
    savePairsToCSV trainOutputPath allPairs
    putStrLn "Dataset generation complete. All pairs saved to train.csv."