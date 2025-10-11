{-# LANGUAGE OverloadedStrings #-}

module Evaluation (main) where

import qualified NLP.Scores as NLP
import qualified Data.Set as Set
import qualified Data.Map as M
import Data.List (sortOn, elemIndex)
import Data.Ord (Down(..))
import Control.Monad (forM, when)
import Data.List.Split (splitOn)
import Text.Read (readMaybe)
import Torch.Tensor (Tensor, asTensor)
import System.IO (openFile, hSetEncoding, utf8, IOMode(ReadMode), hClose, hGetContents)
import System.Environment (getArgs)
import Control.Exception (bracket)
import Data.Maybe (mapMaybe)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO

import PoincareUtils (
    Embeddings,
    readPairsFromCSV,
    distanceBetweenWords
    )

-- 読み込み関数
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
            Just floats -> Just (word, asTensor floats)
            Nothing -> Nothing
    parseLine _ = Nothing

groupByHypernym :: [(String, String)] -> M.Map String [String]
groupByHypernym pairs =
    M.fromListWith (++) [(hyper, [hypo]) | (hyper, hypo) <- pairs]

main :: IO ()
main = do
    args <- getArgs
    when (length args < 3) $
        error "Usage: stack run Evaluation <embeddings.csv> <eval.csv> <output.txt>"

    let trainedEmbPath = args !! 0
        evalDataPath   = args !! 1
        resultFile     = args !! 2

    putStrLn "--- Hyperbolic Embedding Evaluation ---"
    putStrLn $ "Embeddings: " ++ trainedEmbPath
    putStrLn $ "Eval data : " ++ evalDataPath
    putStrLn $ "Result out: " ++ resultFile

    embeddings <- readEmbeddingsCSV trainedEmbPath
    evalPairs  <- readPairsFromCSV evalDataPath

    putStrLn $ "Loaded " ++ show (M.size embeddings) ++ " embeddings"
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

        -- putStrLn $ "\n[DEBUG] Anchor: " ++ u
        -- putStrLn $ "  Hyponyms: " ++ unwords hypos
        -- putStrLn $ "  Top 10 nearest words: " ++ unwords (take 10 rankedList)
        -- putStrLn $ "  Rank: " ++ show rankValue
        -- putStrLn $ "  Mean Average Precision (MAP): " ++ show apValue

        return (rankValue, apValue)

    let meanRank = sum (map fst results) / fromIntegral (length results)
        meanAP   = sum (map snd results) / fromIntegral (length results)
        summary = T.unlines
          [ "--- Evaluation Results (Link Prediction) ---"
          , "Total Hypernyms: " <> T.pack (show (length results))
          , "Mean Rank: " <> T.pack (show meanRank)
          , "Mean Average Precision (MAP): " <> T.pack (show meanAP)
          ]

    putStrLn $ T.unpack summary
    TIO.writeFile resultFile summary
    putStrLn $ "Saved summary to: " ++ resultFile
