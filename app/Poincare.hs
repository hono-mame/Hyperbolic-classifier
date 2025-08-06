{-# LANGUAGE OverloadedStrings #-}

module Poincare (main) where

import System.Random
import System.IO
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Vector as V
import Data.List.Split (splitOn)
import Numeric.LinearAlgebra (Vector, fromList, norm_2)
import Control.Monad (replicateM, forM_)
import Data.Maybe (fromJust)
import Torch.Tensor (Tensor, asTensor,asValue)
import Torch.Functional (sumAll, pow ,sqrt)

type Entity = String
type Embedding = Vector Double
type Embeddings = M.Map Entity Embedding

randomVector :: Int -> IO (Vector Double)
randomVector dim = do
  let eps = 1e-3
  vals <- replicateM dim (randomRIO (-eps, eps))
  return $ fromList vals

initializeEmbeddings :: Int -> [Entity] -> IO Embeddings
initializeEmbeddings dim entities = do
  vecs <- mapM (const $ randomVector dim) entities
  return $ M.fromList (zip entities vecs)

printEmbeddings :: Embeddings -> IO ()
printEmbeddings embs = do
  putStrLn "---------- print first 10 vectors of each word ----------"
  mapM_ printOne (take 10 $ M.toList embs)
  where
    printOne (ent, vec) = putStrLn $ ent ++ ": " ++ show vec

readWordsFromCSV :: FilePath -> IO (S.Set String)
readWordsFromCSV path = do
  contents <- readFile path
  let ls = drop 1 $ lines contents
      pairs = map (take 2 . splitOn ",") ls
      wordsSet = S.fromList (concat pairs)
  return wordsSet

poincareDistance :: Vector Double -> Vector Double -> Double
poincareDistance u v =
  let uNorm = norm_2 u
      vNorm = norm_2 v
      diffNorm = norm_2 (u - v)
      num = 2 * diffNorm ** 2
      denom = (1 - uNorm ** 2) * (1 - vNorm ** 2)
      x = 1 + (num / denom)
  in acosh x

distanceBetweenWords :: M.Map String (Vector Double) -> String -> String -> Maybe Double
distanceBetweenWords embeddings word1 word2 = do
  vec1 <- M.lookup word1 embeddings
  vec2 <- M.lookup word2 embeddings
  return $ poincareDistance vec1 vec2

riemannianGradient :: Tensor -> Tensor -> Tensor
riemannianGradient thetaT grad =
  let normSquared = sumAll (thetaT * thetaT)
      coeff = pow (2.0 :: Float) (asTensor (1.0 :: Float) - normSquared) / asTensor (4.0 :: Float)
  in coeff * grad

projectToBall :: Tensor -> Tensor
projectToBall thetaT =
  let norm = Torch.Functional.sqrt (sumAll (thetaT * thetaT))
      normScalar = asValue norm :: Float
      eps = 1e-5
  in if normScalar > 1.0 && normScalar > eps
        then thetaT / asTensor (normScalar - eps)
        else thetaT

main :: IO ()
main = do
  let dim = 3
  wordSet <- readWordsFromCSV "data/MLP/train_small_test.csv"
  embeddings <- initializeEmbeddings dim (S.toList wordSet)
  -- test
  printEmbeddings embeddings
  let word1 = "事業年度"
      word2 = "勧誘"
  case distanceBetweenWords embeddings word1 word2 of
      Just d -> putStrLn $ "Distance between \"" ++ word1 ++ "\" and \"" ++ word2 ++ "\": " ++ show d
      Nothing -> putStrLn "One or both words not found."
