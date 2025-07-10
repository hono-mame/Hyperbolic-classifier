{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module MLPRandomMain (main) where

import Control.Monad (when, replicateM)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import qualified Data.Vector as V
import qualified Data.Set as Set
import Data.Maybe (catMaybes)
import Data.Csv (decodeByName, FromNamedRecord)
import qualified Data.Map.Strict as Map
import GHC.Generics
import System.Random (randomRIO)
import Torch.Functional (mul, sigmoid, binaryCrossEntropyLoss', logicalNot, sumAll, gt, squeezeAll, clamp)
import qualified Torch.Functional as F
import Torch.Tensor (Tensor,asTensor, asValue, shape)
import Evaluation (evalAccuracy, evalPrecision, evalRecall, calcF1, confusionMatrix, confusionMatrixPairs)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer,MLPParams)
import Torch.Train        (update)
import Torch.Optim        (GD(..),  foldLoop)
import Torch.NN           (sample)

data InputWordPair = InputWordPair
  { hyper :: String,
    hypo :: String,
    label :: Float
  } deriving (Show, Generic, FromNamedRecord)

-- generate randomly initialized embeddings
createRandomEmbeddings :: FilePath -> IO (Map.Map String [Float])
createRandomEmbeddings csvPath = do
  csvData <- BL.readFile csvPath
  case decodeByName csvData of
    Left err -> error $ "Failed to decode CSV: " ++ err
    Right (_, v) -> do
      let wordPairs = V.toList v
          wordsSet = foldr (\InputWordPair{..} acc -> hyper : hypo : acc) [] wordPairs
          uniqueWords = Map.keysSet $ Map.fromList $ zip wordsSet (repeat ())
      embList <- mapM (\w -> do
                          vec <- genRandomVec 200
                          return (w, vec)
                      )(Set.toList uniqueWords) 
      return $ Map.fromList embList

-- Generate a random vector of given size
genRandomVec :: Int -> IO [Float]
genRandomVec dim = mapM (\_ -> randomRIO (-0.1, 0.1)) [1..dim]

lookupEmbedding :: Map.Map String [Float] -> String -> [Float]
lookupEmbedding embMap word = Map.findWithDefault (replicate 200 0.0) word embMap


loadWordPairData :: FilePath -> Map.Map String [Float] -> IO (Tensor, Tensor)
loadWordPairData filePath embMap = do
  csvData <- BL.readFile filePath
  case decodeByName csvData of
    Left err -> error $ "Failed to decode CSV: " ++ err
    Right (_, v) -> do
      let rows = V.toList $ V.map (\InputWordPair{..} ->
                    let hVec = lookupEmbedding embMap hyper
                        yVec = lookupEmbedding embMap hypo
                        inputVec = hVec ++ yVec
                        -- debug: check if hyper and hypo vectors are zero vectors
                        isHyperZero = all (== 0.0) hVec
                        isHypoZero = all (== 0.0) yVec
                    in (hyper, hypo, hVec, yVec, inputVec, [label], isHyperZero, isHypoZero)
                  ) v
      -- for debug
      putStrLn "===== Sample Word Embeddings (first 10) ====="
      mapM_ (\(h, y, hv, yv, _, l, isHZ, isYZ) -> do
                putStrLn $ "Hyper: " ++ h ++ ", Embedding (first 5): " ++ show (take 5 hv) ++ if isHZ then " [ZERO VECTOR!]" else ""
                putStrLn $ "Hypo : " ++ y ++ ", Embedding (first 5): " ++ show (take 5 yv) ++ if isYZ then " [ZERO VECTOR!]" else ""
                putStrLn $ "Label: " ++ show l
                putStrLn "-------------------------------------------"
            ) (take 10 rows)

      let (xs, ys) = unzip $ map (\(_,_,_,_,x,y,_,_) -> (x,y)) rows

      -- for debug
      let flatInputs = concat xs
      putStrLn $ "Input data stats:"
      putStrLn $ "  Min value: " ++ show (minimum flatInputs)
      putStrLn $ "  Max value: " ++ show (maximum flatInputs)
      putStrLn $ "  Mean value: " ++ show (sum flatInputs / fromIntegral (length flatInputs))
      
      return (asTensor xs, asTensor ys)


trainMLP :: MLPParams -> Tensor -> Tensor -> IO MLPParams
trainMLP initModel inputs targets = do
  putStrLn $ "inputs shape in trainMLP: " ++ show (shape inputs)
  putStrLn $ "targets shape in trainMLP: " ++ show (shape targets)
  (trainedModel, _) <- foldLoop (initModel, []) 200 $ \(state, losses) i -> do
    let yPred = sigmoid (clamp (-10.0) 10.0 (mlpLayer state inputs))
        targets1d = squeezeAll targets
        loss = binaryCrossEntropyLoss' yPred targets1d
        lossValue = asValue loss :: Float
    putStrLn $ "yPred: " ++ show yPred
    when (i `mod` 1 == 0) $ putStrLn $ "Iter " ++ show i ++ ": Loss = " ++ show lossValue    
    (newState, _) <- update state GD loss 1e-3
    return (newState, losses ++ [lossValue])
  return trainedModel

evaluate :: MLPParams -> Tensor -> Tensor -> IO ()
evaluate model inputs targets = do
  let preds = sigmoid (mlpLayer model inputs)
      targets1d = squeezeAll targets
      predLabels = gt preds 0.5
      trueLabels = gt targets1d 0.5
      tp = sumAll (mul predLabels trueLabels)
      tn = sumAll (mul (logicalNot predLabels) (logicalNot trueLabels))
      fp = sumAll (mul predLabels (logicalNot trueLabels))
      fn = sumAll (mul (logicalNot predLabels) trueLabels)
  let cm = confusionMatrix tp tn fp fn 
      cmPairs = confusionMatrixPairs cm
      acc = evalAccuracy tp tn fp fn
      prec = evalPrecision tp fp
      rec = evalRecall tp fn
      f1 = calcF1 prec rec
  putStrLn $ "Accuracy: " ++ show (asValue acc :: Float)
  putStrLn $ "Precision: " ++ show (asValue prec :: Float)
  putStrLn $ "Recall: " ++ show (asValue rec :: Float)
  putStrLn $ "F1 Score: " ++ show (asValue f1 :: Float)
  putStrLn "Confusion Matrix:"
  mapM_ (putStrLn . show) cmPairs

main :: IO ()
main = do
  putStrLn "Initializing random embeddings..."
  embMap <- createRandomEmbeddings "data/MLP/train_small.csv"
  putStrLn $ "Total words in embedding: " ++ show (Map.size embMap)
  putStrLn "First 10 words in embedding:"
  mapM_ putStrLn $ take 10 $ Map.keys embMap

  putStrLn "Loading training data..."
  (trainX, trainY) <- loadWordPairData "data/MLP/train_small.csv" embMap
  -- putStrLn $ "trainx shape: " ++ show (shape trainX)
  -- putStrLn $ "trainy shape: " ++ show (shape trainY)
  -- putStrLn $ "trainy: " ++ show trainY

  let device = Device CPU 0
  putStrLn "Initializing model..."
  initModel <- sample $ MLPHypParams device 400 [(256,Selu),(256,Selu),(64,Selu), (1,Id)]

  putStrLn "Training..."
  model <- trainMLP initModel trainX trainY

  putStrLn "Loading eval data..."
  (evalX, evalY) <- loadWordPairData "data/MLP/eval_small.csv" embMap

  putStrLn "Evaluating..."
  evaluate model evalX evalY
