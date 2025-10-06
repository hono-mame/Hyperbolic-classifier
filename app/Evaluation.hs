{-# LANGUAGE OverloadedStrings #-}

module Evaluation (main) where

import qualified NLP.Scores as NLP
import qualified Data.Set as Set
import Data.List.NonEmpty (fromList)

main :: IO ()
main = do
    putStrLn "This is for evaluation"

    putStrLn "\n--- Reciprocal Rank Tests ---"
    let test1 = NLP.recipRank 5 [1, 2, 5, 8]
    putStrLn $ "Reciprocal Rank (RR) for [1,2,5,8] with relevant at 5 (should be 0.3333333): " ++ show test1

    let test2 = NLP.recipRank 10 [10, 5, 20]
    putStrLn $ "Reciprocal Rank (RR) for [10,5,20] with relevant at 10 (should be 1.0): " ++ show test2

    putStrLn "\n--- Average Precision Tests ---"
    -- 正解集合: {A, B}
    -- 検索結果: [A, B, C, D]
    -- AP = (1/1 + 2/2) / 2 = (1.0 + 1.0) / 2 = 1.0
    let gold1 = Set.fromList ["A", "B"]
    let retrieved1 = ["A", "B", "C", "D"] :: [String]
    let ap1 = NLP.avgPrecision gold1 retrieved1
    putStrLn $ "AP Test 1 (should be 1.0): " ++ show ap1

    -- 正解集合: {A, B}
    -- 検索結果: [C, D, A, B]
    -- 適合 at 3 (A): 1/3
    -- 適合 at 4 (B): 2/4
    -- AP = (1/3 + 2/4) / 2 = (0.333... + 0.5) / 2 = 0.4166...
    let gold2 = Set.fromList [5, 10]
    let retrieved2 = [1, 2, 5, 10] :: [Int]
    let ap2 = NLP.avgPrecision gold2 retrieved2
    putStrLn $ "AP Test 2 (should be 0.4166666666666667): " ++ show ap2

    -- 正解集合: {A, B, C}
    -- 検索結果: [A, X, B, Y] (Cは含まれない)
    -- 適合 at 1 (A): 1/1
    -- 適合 at 3 (B): 2/3
    -- AP = (1/1 + 2/3) / 3 (正解集合のサイズ) = (1.0 + 0.666...) / 3 = 0.555...
    let gold3 = Set.fromList ['A', 'B', 'C']
    let retrieved3 = ['A', 'X', 'B', 'Y'] :: [Char]
    let ap3 = NLP.avgPrecision gold3 retrieved3
    putStrLn $ "AP Test 3 (should be 0.5555555555555556): " ++ show ap3