{-# LANGUAGE OverloadedStrings #-}

module Poincare (main) where

import qualified Data.Set as S
import Torch.Tensor()
import ML.Exp.Chart (drawLearningCurve)

import PoincareUtils
    (initializeEmbeddings, 
    printEmbeddings, 
    readWordsFromCSV, 
    readPairsFromCSV,
    train, 
    saveEmbeddings)

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
