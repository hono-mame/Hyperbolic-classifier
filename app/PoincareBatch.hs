{-# LANGUAGE OverloadedStrings #-}

module PoincareBatch (main) where

import qualified Data.Set as S
import qualified Data.Map as M
import System.Environment (getArgs)
import Torch.Tensor()
import ML.Exp.Chart (drawLearningCurve)
import PoincareUtils(
    initializeEmbeddings, 
    printEmbeddings, 
    readWordsFromCSV, 
    readPairsFromCSV,
    trainBatch,
    saveEmbeddings,
    readArg)

main :: IO ()
main = do
  args <- getArgs
  let dim        = readArg 0 3 args
      epochs     = readArg 1 200 args
      baseLR     = readArg 2 0.01 args
      negK       = readArg 3 5 args
      burnC      = readArg 4 10 args
      burnEpochs = readArg 5 10 args
      batchSize  = readArg 6 512 args
      csvPath    = if length args > 7 then args !! 7 else "data/Hyperbolic/eval_test_train.csv"
      outCSV     = if length args > 8 then args !! 8 else "outputs/poincare_embeddings.csv"
      outChart   = if length args > 9 then args !! 9 else "charts/poincare_learning_curve.png"

  putStrLn "======================================"
  putStrLn "Poincare Embedding Training"
  putStrLn $ "dim = " ++ show dim ++ ", epochs = " ++ show epochs ++ ", baseLR = " ++ show baseLR
  putStrLn $ "negK = " ++ show negK ++ ", burnC = " ++ show burnC
  putStrLn $ "burnEpochs = " ++ show burnEpochs ++ ", batchSize = " ++ show batchSize
  putStrLn $ "Input file: " ++ csvPath
  putStrLn $ "Output CSV: " ++ outCSV
  putStrLn $ "Output chart: " ++ outChart
  putStrLn "======================================"

  pairs <- readPairsFromCSV csvPath
  wordSet <- readWordsFromCSV csvPath
  embeddings <- initializeEmbeddings dim (S.toList wordSet)

  putStrLn "Start training..."
  (trained, lossHistory) <- trainBatch epochs baseLR negK burnC burnEpochs batchSize pairs embeddings
  putStrLn "Training finished."

  drawLearningCurve outChart "Poincare Embedding Loss" [("Training Loss", lossHistory)]
  saveEmbeddings outCSV trained
  putStrLn "Results saved successfully."