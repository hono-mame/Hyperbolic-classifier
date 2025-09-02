{-# LANGUAGE OverloadedStrings #-}

module Poincare (main) where

import System.Random
import System.IO
import qualified Data.Map as M
import qualified Data.Set as S
import Data.List.Split (splitOn)
import Control.Monad (foldM)
import Data.Maybe (fromJust)
import Prelude hiding (sqrt, acosh, exp, log)
import Data.Int (Int64)

import Torch.Tensor (Tensor, asTensor, asValue, toDevice, toType, select, shape)
import Torch.Functional (sumAll, pow, sqrt, Dim (..), stack, exp, log, max)
import Torch.TensorFactories (randnIO')
import Torch.Functional.Internal (acosh)
import Torch.Optim (Gradients (..), grad', Loss)
import Torch.Device (Device (..), DeviceType (..))
import Torch.Autograd (makeIndependent, toDependent)
import Torch.DType (DType (..))

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

readPairsFromCSV :: FilePath -> IO [(String, String)]
readPairsFromCSV path = do
  contents <- readFile path
  let ls = drop 1 $ lines contents
      pairs = map (take 2 . splitOn ",") ls
  return $ map (\[a,b] -> (a,b)) $ filter (\x -> length x >= 2) pairs

poincareDistance :: Tensor -> Tensor -> Tensor
poincareDistance u v =
  let uNormSquared = sumAll (u * u)
      vNormSquared = sumAll (v * v)
      diff = u - v
      diffNormSquared = sumAll (diff * diff)
      num = 2 * diffNormSquared
      denom = (1 - uNormSquared) * (1 - vNormSquared)
      eps = 1e-6 :: Float
      safeDenom = denom + asTensor eps
      x = 1 + (num / safeDenom)
      clippedArg = Torch.Functional.max (x * x - asTensor (1.0 :: Float))
  in log (x + sqrt clippedArg)

distanceBetweenWords :: Embeddings -> String -> String -> Maybe Double
distanceBetweenWords embeddings word1 word2 = do
  vec1 <- M.lookup word1 embeddings
  vec2 <- M.lookup word2 embeddings
  let dTensor = poincareDistance vec1 vec2
      distFloat = asValue dTensor :: Float
  return $ realToFrac distFloat

projectToBall :: Tensor -> Tensor
projectToBall thetaT =
  let norm = sqrt (sumAll (thetaT * thetaT))
      normScalar = asValue norm :: Float
      eps = 1e-5 :: Float
  in if normScalar >= 1.0
        then (thetaT / asTensor normScalar) * asTensor (1.0 - eps)
        else thetaT

sampleNegatives :: Int -> Int -> Int -> IO [Int]
sampleNegatives vocab k excludeIdx = go k []
  where
    go 0 acc = return acc
    go n acc = do
      idx <- randomRIO (0, vocab - 1)
      if idx == excludeIdx || idx `elem` acc
        then go n acc
        else go (n - 1) (idx : acc)

runStepRSGD :: Float -> Int -> [(String,String)] -> Embeddings -> IO Embeddings
runStepRSGD learningRate negK pairs embeddings = do
  foldM updateOne embeddings pairs
  where
    updateOne embs (u, v) = do
      let embList = M.toList embs
          n = length embList
          idxMap = M.fromList $ zip (map fst embList) [0..]

      let allVecs = map snd embList
          stacked = stack (Dim 0) allVecs
      
      independentStacked <- makeIndependent stacked
      let depStacked = toDependent independentStacked
      
      let idxU = fromIntegral $ fromJust $ M.lookup u idxMap
          idxV = fromIntegral $ fromJust $ M.lookup v idxMap

      negIdxs <- sampleNegatives n negK (fromIntegral idxV)
      let candIdxs = (fromIntegral idxV) : map fromIntegral negIdxs

      let vecU = select 0 idxU depStacked
          vecCandidates = map (\i -> select 0 i depStacked) candIdxs
      
      let dists = map (poincareDistance vecU) vecCandidates
          negDists = map (\dt -> exp (asTensor (0.0 :: Float) - dt)) dists
          sumExp = sumAll (stack (Dim 0) negDists)
          dPos = head dists
          lossTensor = dPos + log sumExp

      putStrLn $ "distance: " ++ show dPos

      let Gradients grads = grad' lossTensor [independentStacked]
      
      case grads of
        (gradTensor : _) -> do
          updatedResults <- mapM (\(word, vec) -> do
            let i = fromJust $ M.lookup word idxMap
                gVec = select 0 (fromIntegral i) gradTensor
                depVec = select 0 (fromIntegral i) depStacked
                normSquared = sumAll (depVec * depVec)
                coeff = pow (2.0 :: Float) (asTensor (1.0 :: Float) - normSquared) / asTensor (4.0 :: Float)
                riemGrad = coeff * gVec
                lrTensor = asTensor (learningRate :: Float)
                updated = vec - lrTensor * riemGrad 
                projected = projectToBall updated
            return (word, projected)
            ) embList
          return $ M.fromList updatedResults
        [] -> error "Empty gradient list"

train :: Int      -- epochs
      -> Float    -- base learning rate η
      -> Int      -- neg samples per positive k
      -> Int      -- burn-in coefficient c (use η/c during burn-in)
      -> Int      -- burn-in epochs
      -> [(String,String)] -- input pairs
      -> Embeddings -- initial embeddings
      -> IO Embeddings -- trained embeddings
train epochs baseLR negK burnC burnEpochs pairs embs0 =
  foldM step embs0 [1..epochs]
  where
    step embs epoch = do
      putStrLn $ "Epoch " ++ show epoch
      let lr = if epoch <= burnEpochs then baseLR / fromIntegral burnC else baseLR
      newEmbs <- runStepRSGD lr negK pairs embs
      putStrLn "Computing epoch loss..."
      lossVal <- computeDatasetLoss newEmbs pairs negK
      putStrLn $ "Epoch " ++ show epoch ++ "  lr=" ++ show lr ++ "  Loss=" ++ show lossVal
      return newEmbs

computeDatasetLoss :: Embeddings 
                  -> [(String,String)]
                  -> Int 
                  -> IO Float
computeDatasetLoss embs pairs negK = do
  let embList = M.toList embs
      idxMap = M.fromList $ zip (map fst embList) [0..]
      stacked = stack (Dim 0) (map snd embList)
  losses <- mapM (pairLoss embList idxMap stacked) pairs
  let total = sum losses
  return total
  where
    pairLoss embList idxMap stacked (u,v) = do
      let idxU = fromJust $ M.lookup u idxMap
          idxV = fromJust $ M.lookup v idxMap
          n = length embList
          negIdxs = take negK $ filter (/= idxV) [0..(n-1)]
          candIdxs = idxV : negIdxs
          vecU = select 0 idxU stacked
          vecCandidates = map (\i -> select 0 i stacked) candIdxs
          dists = map (poincareDistance vecU) vecCandidates
          negExps = map (\dt -> exp (asTensor (0.0 :: Float) - dt)) dists
          sumExp = sumAll (stack (Dim 0) negExps)
          dPos = head dists
          lossTensor = dPos + log sumExp
          lossVal = asValue lossTensor :: Float
      return lossVal

main :: IO ()
main = do
  let dim = 3
      epochs = 20
      baseLR = 0.001
      negK = 5
      burnC = 10
      burnEpochs = 10
      csvPath = "data/Hyperbolic/train.csv"

  pairs <- readPairsFromCSV csvPath
  wordSet <- readWordsFromCSV csvPath
  embeddings <- initializeEmbeddings dim (S.toList wordSet)

  putStrLn "Initial embeddings:"
  printEmbeddings embeddings
  -- let word1 = "事業年度"
  --     word2 = "勧誘"
  -- case distanceBetweenWords embeddings word1 word2 of
  --     Just d -> putStrLn $ "Distance between \"" ++ word1 ++ "\" and \"" ++ word2 ++ "\": " ++ show d
  --     Nothing -> putStrLn "One or both words not found."

  putStrLn "Start training..."
  trained <- train epochs baseLR negK burnC burnEpochs pairs embeddings
  putStrLn "Training finished."
  printEmbeddings trained
