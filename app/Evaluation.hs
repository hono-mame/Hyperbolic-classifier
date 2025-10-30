{-# LANGUAGE OverloadedStrings #-}

module Evaluation (main) where

import qualified NLP.Scores as NLP
import qualified Data.Set as Set
import qualified Data.Map as M
import Data.List (sortOn, elemIndex)
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
    when (length args < 4) $
        error "Usage: stack run Evaluation <embeddings.csv> <eval.csv> <train.csv> <output.txt>"

    let trainedEmbPath = args !! 0
        evalDataPath   = args !! 1
        trainDataPath  = args !! 2
        resultFile     = args !! 3

    putStrLn "--- Hyperbolic Embedding Evaluation (Filtered) ---"
    putStrLn $ "Embeddings: " ++ trainedEmbPath
    putStrLn $ "Eval data : " ++ evalDataPath
    putStrLn $ "Train data: " ++ trainDataPath
    putStrLn $ "Result out: " ++ resultFile

    embeddings <- readEmbeddingsCSV trainedEmbPath
    evalPairs  <- readPairsFromCSV evalDataPath
    trainPairs <- readPairsFromCSV trainDataPath

    putStrLn $ "Loaded " ++ show (M.size embeddings) ++ " embeddings"
    putStrLn $ "Loaded " ++ show (length evalPairs) ++ " eval pairs (before filtering)"
    putStrLn $ "Loaded " ++ show (length trainPairs) ++ " train pairs"

    let embeddingsSet = Set.fromList (M.keys embeddings)
        filteredEvalPairs = [(hyper, hypo) | (hyper, hypo) <- evalPairs, 
                                            Set.member hyper embeddingsSet, 
                                            Set.member hypo embeddingsSet]
    putStrLn $ "Filtered " ++ show (length filteredEvalPairs) ++ " eval pairs (both hyper/hypo exist)"

    let allWords = M.keys embeddings
        groupedEvalPairs = groupByHypernym filteredEvalPairs 
        groupedTrainPairs = groupByHypernym trainPairs
        hypers = M.keys groupedEvalPairs
        debugLimit = 5
    resultsAndDebug <- forM (zip [1..] hypers) $ \(idx, u) -> do
        let hyposEval = groupedEvalPairs M.! u
            knownHypos = Set.fromList (M.findWithDefault [] u groupedTrainPairs)
            candidateWords = [w | w <- allWords, w /= u, not (Set.member w knownHypos)]
            distances = [ (w, distanceBetweenWords embeddings u w)
                        | w <- candidateWords ]
            validDists = mapMaybe (\(w, md) -> fmap (\d -> (w, d)) md) distances
            rankedList = map fst $ sortOn snd validDists
            ranksFound = mapMaybe (`elemIndex` rankedList) hyposEval
            rankValue = case ranksFound of
                [] -> fromIntegral (length rankedList)
                rs -> fromIntegral (minimum rs + 1)
            goldSet = Set.fromList hyposEval
            apValue = NLP.avgPrecision goldSet rankedList
            debugText =
                if idx <= debugLimit
                then [ T.unlines
                       [ "[DEBUG] Anchor: " <> T.pack u
                       , "  Eval hyponyms (gold): " <> T.pack (unwords hyposEval)
                       , "  Filtered (train) hypos: " <> T.pack (unwords (Set.toList knownHypos))
                       , "  Candidates after filtering: " <> T.pack (show (length candidateWords))
                       , "  Top 10 nearest words: " <> T.pack (unwords (take 10 rankedList))
                       , "  Rank: " <> T.pack (show rankValue)
                       , "  Mean Average Precision (MAP): " <> T.pack (show apValue)
                       , ""
                       ]
                     ]
                else []

        return ((rankValue, apValue), debugText)
    let results = [r | (r, _) <- resultsAndDebug]
        debugInfo = concat [d | (_, d) <- resultsAndDebug]
    let meanRank = sum (map fst results) / fromIntegral (length results)
        meanAP   = sum (map snd results) / fromIntegral (length results)
        summary = T.unlines $
          [ "--- Filtered Evaluation Results (Link Prediction) ---"
          , "Total Hypernyms (Evaluated): " <> T.pack (show (length results))
          , "Mean Rank: " <> T.pack (show meanRank)
          , "Mean Average Precision (MAP): " <> T.pack (show meanAP)
          , ""
          , "--- DEBUG (first 5 anchors) ---"
          ] ++ debugInfo

    putStrLn $ T.unpack summary
    TIO.writeFile resultFile summary
    putStrLn $ "Saved summary to: " ++ resultFile