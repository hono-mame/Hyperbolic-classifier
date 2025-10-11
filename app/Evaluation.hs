{-# LANGUAGE OverloadedStrings #-}

module Evaluation (main) where

import qualified NLP.Scores as NLP
import qualified Data.Set as Set
import qualified Data.Map as M
import Data.List (sortOn, elemIndex)
import Data.Ord (Down(..))
import Control.Monad (forM, forM_)
import Data.List.Split (splitOn)
import Text.Read (readMaybe)
import Torch.Tensor (Tensor, asTensor)
import System.IO (openFile, hSetEncoding, utf8, IOMode(ReadMode), hClose, hGetContents)
import Control.Exception (bracket)
import Data.Maybe (isJust, mapMaybe)

import PoincareUtils (
    Embeddings,
    readPairsFromCSV,
    distanceBetweenWords,
    Entity
    )

readEmbeddingsCSV :: FilePath -> IO Embeddings
readEmbeddingsCSV path = do
    contents <- bracket (openFile path ReadMode) hClose $ \handle -> do
        hSetEncoding handle utf8
        s <- hGetContents handle
        length s `seq` return s

    let ls = drop 1 $ lines contents
        parsedLines = map (parseLine . splitOn ",") ls
        validEmbeddings = M.fromList [p | Just p <- parsedLines]
    return validEmbeddings
  where
    parseLine (word:dims) =
        case traverse (readMaybe :: String -> Maybe Float) dims of
            Just floats -> Just (word, asTensor (floats :: [Float]))
            Nothing -> Nothing
    parseLine _ = Nothing

groupByHypernym :: [(String, String)] -> M.Map String [String]
groupByHypernym pairs =
    M.fromListWith (++) [(hyper, [hypo]) | (hyper, hypo) <- pairs]

main :: IO ()
main = do
    putStrLn "--- Hyperbolic Embedding Evaluation ---"

    let trainedEmbPath = "outputs/eval_test_poincare_embeddings.csv"
        evalDataPath = "data/Hyperbolic/eval_test_eval.csv"

    embeddings <- readEmbeddingsCSV trainedEmbPath
    putStrLn $ "Loaded " ++ show (M.size embeddings) ++ " embeddings from " ++ trainedEmbPath

    evalPairs <- readPairsFromCSV evalDataPath
    putStrLn $ "Loaded " ++ show (length evalPairs) ++ " evaluation pairs."

    let allWords = M.keys embeddings
        groupedPairs = groupByHypernym evalPairs
        hypers = M.keys groupedPairs

    results <- forM hypers $ \u -> do
        let hypos = groupedPairs M.! u
            distances = [ (w, distanceBetweenWords embeddings u w)
                        | w <- allWords, w /= u ]
            validDists = mapMaybe (\(w, md) -> fmap (\d -> (w, d)) md) distances
            rankedList = map fst $ sortOn snd validDists

            ranksFound = mapMaybe (`elemIndex` rankedList) hypos
            rankValue = case ranksFound of
                [] -> fromIntegral (length rankedList)
                rs -> fromIntegral (minimum rs + 1)

            goldSet = Set.fromList hypos
            apValue = NLP.avgPrecision goldSet rankedList

        putStrLn $ "\n[DEBUG] Anchor: " ++ u
        putStrLn $ "  Hyponyms: " ++ unwords hypos
        putStrLn $ "  Top 10 nearest words: " ++ unwords (take 10 rankedList)
        putStrLn $ "  Rank: " ++ show rankValue
        putStrLn $ "  Mean Average Precision (MAP): " ++ show apValue

        return (rankValue, apValue)

    let meanRank = sum (map fst results) / fromIntegral (length results)
        meanAP   = sum (map snd results) / fromIntegral (length results)

    putStrLn "\n--- Evaluation Results (Link Prediction) ---"
    putStrLn $ "Total Hypernyms: " ++ show (length results)
    putStrLn $ "Mean Rank: " ++ show meanRank
    putStrLn $ "Mean Average Precision (MAP): " ++ show meanAP

    putStrLn "\n--- Verification Tests ---"
    let test1 = NLP.recipRank 5 [1, 2, 5, 8]
    putStrLn $ "RR for [1,2,5,8] with relevant at 5: " ++ show test1

    let gold2 = Set.fromList [5, 10]
        retrieved2 = [1, 2, 5, 10] :: [Int]
        ap2 = NLP.avgPrecision gold2 retrieved2
    putStrLn $ "AP Test (should be 0.4166...): " ++ show ap2

  where
    elemIndex :: Eq a => a -> [a] -> Maybe Int
    elemIndex x = go 0
      where
        go _ [] = Nothing
        go n (y:ys)
          | x == y    = Just n
          | otherwise = go (n + 1) ys
