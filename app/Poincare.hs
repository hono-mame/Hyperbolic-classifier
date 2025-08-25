{-# LANGUAGE OverloadedStrings #-}

module Poincare (main) where

import System.Random
import System.IO
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Vector as V
import Data.List.Split (splitOn)
import Control.Monad (foldM)
import Data.Maybe (fromJust)
import Prelude hiding (sqrt, acosh)

import Torch.Tensor (Tensor, asTensor,asValue)
import Torch.Functional (sumAll, pow ,sqrt)
import Torch.TensorFactories (randnIO')
import Torch.Functional.Internal (acosh)

type Entity = String
type Embedding = Tensor
type Embeddings = M.Map Entity Embedding

randomTensor :: Int -> IO Tensor
randomTensor dim = do
  t <- randnIO' [dim]
  let eps = 1e-3 :: Double
  return $ t * asTensor eps

initializeEmbeddings :: Int -> [Entity] -> IO Embeddings
initializeEmbeddings dim entities = do
  vecs <- mapM (const $ randomTensor dim) entities
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

poincareDistance :: Tensor -> Tensor -> Tensor
poincareDistance u v =
  let uNorm = sqrt $ sumAll (u * u)
      vNorm = sqrt $ sumAll (v * v)
      diff = u - v
      diffNorm = sqrt $ sumAll (diff * diff)
      num = 2 * (diffNorm * diffNorm)
      denom = (1 - uNorm * uNorm) * (1 - vNorm * vNorm)
      x = 1 + (num / denom)
      vx = asValue x :: Float
  in acosh (asTensor vx)

distanceBetweenWords :: Embeddings -> String -> String -> Maybe Double
distanceBetweenWords embeddings word1 word2 = do
  vec1 <- M.lookup word1 embeddings
  vec2 <- M.lookup word2 embeddings
  let dTensor = poincareDistance vec1 vec2
      distFloat = asValue dTensor :: Float
  return $ realToFrac distFloat

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

-- dummy loss function (implement this later)
lossFunction :: Embeddings -> Tensor
lossFunction embs =
  let allVecs = M.elems embs
  in sumAll $ foldl1 (+) (map (\v -> v * v) allVecs)

updateEmbedding :: Embedding -> Embedding
updateEmbedding emb = emb

runStepRSGD :: Embeddings -> Embeddings
runStepRSGD = M.map updateEmbedding

train :: Int -> Embeddings -> IO Embeddings
train epochs embs0 =
  foldM step embs0 [1..epochs]
  where
    step embs epoch = do
      let loss = lossFunction embs
          lossVal = asValue loss :: Float
      putStrLn $ "Epoch " ++ show epoch ++ " | Loss = " ++ show lossVal
      let newEmbs = runStepRSGD embs
      return newEmbs

main :: IO ()
main = do
  let dim = 3
      epoch = 10
  wordSet <- readWordsFromCSV "data/Hyperbolic/input_test.csv"
  embeddings <- initializeEmbeddings dim (S.toList wordSet)

  putStrLn "Initial embeddings:"
  printEmbeddings embeddings
  -- let word1 = "事業年度"
  --     word2 = "勧誘"
  -- case distanceBetweenWords embeddings word1 word2 of
  --     Just d -> putStrLn $ "Distance between \"" ++ word1 ++ "\" and \"" ++ word2 ++ "\": " ++ show d
  --     Nothing -> putStrLn "One or both words not found."

  -- training --
  putStrLn "Start training..."
  trained <- train epoch embeddings
  putStrLn "Training finished."
  printEmbeddings trained
