{-# LANGUAGE OverloadedStrings #-}

module Poincare (main) where

import System.Random
import System.IO
import qualified Data.Map as M
import qualified Data.Set as S
import Data.List.Split (splitOn)
import Control.Monad (foldM, when)
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
import ML.Exp.Chart (drawLearningCurve)

import PoincareUtils
    (initializeEmbeddings, printEmbeddings, readWordsFromCSV, readPairsFromCSV,
     poincareDistance, distanceBetweenWords, train, saveEmbeddings)

type Entity = String
type Embedding = Tensor
type Embeddings = M.Map Entity Embedding

main :: IO ()
main = do
  let dim = 3
      epochs = 500
      baseLR = 0.01
      negK = 4
      burnC = 10
      burnEpochs = 10
      csvPath = "data/Hyperbolic/hypernym_relations_jpn_nouns_head_100.csv"

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
  (trained, lossHistory) <- train epochs baseLR negK burnC burnEpochs pairs embeddings
  putStrLn "Training finished."
  printEmbeddings trained

  drawLearningCurve "charts/poincare_learning_curve.png" "Poincare Embedding Loss" [("Training Loss", lossHistory)]
  saveEmbeddings "outputs/poincare_embeddings.csv" trained
