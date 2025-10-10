{-# LANGUAGE OverloadedStrings #-}

module PoincareBatch (main) where

import qualified Data.Set as S
import qualified Data.Map as M
import Torch.Tensor()
import ML.Exp.Chart (drawLearningCurve)

import PoincareUtils(
    initializeEmbeddings, 
    printEmbeddings, 
    readWordsFromCSV, 
    readPairsFromCSV,
    trainBatch,
    saveEmbeddings)

main :: IO ()
main = do
  let dim = 3
      epochs = 200
      baseLR = 0.01
      negK = 5
      burnC = 10
      burnEpochs = 10
      batchSize = 512
      csvPath = "data/Hyperbolic/eval_test_train.csv"

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
  (trained, lossHistory) <- trainBatch epochs baseLR negK burnC burnEpochs batchSize pairs embeddings
  putStrLn "Training finished."
  printEmbeddings trained

  drawLearningCurve "charts/eval_test_learning_curve.png" "Poincare Embedding Loss" [("Training Loss", lossHistory)]
  saveEmbeddings "outputs/eval_test_poincare_embeddings.csv" trained
