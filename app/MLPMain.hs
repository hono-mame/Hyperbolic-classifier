{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module MLPMain (main) where

import Control.Monad (when)
import Data.List (foldl', intersperse)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import qualified Data.Vector as V
import Data.Maybe (catMaybes)
import Data.Csv (decodeByName, FromNamedRecord)
import qualified Data.Map.Strict as Map
import GHC.Generics
import Torch
import Torch.Functional (mul, sigmoid, binaryCrossEntropyLoss')
import Torch.Tensor (asTensor, asValue)
import Evaluation (evalAccuracy, evalPrecision, evalRecall, calcF1, confusionMatrix, confusionMatrixPairs)

data InputWordPair = InputWordPair
  { hyper :: String,
    hypo :: String,
    label :: Float
  } deriving (Show, Generic, FromNamedRecord)


data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  } deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM (sample . uncurry LinearSpec) layer_sizes
    return $ MLP linears nonlinearitySpec
    where
      mkLayerSizes (a : b : t) = (a, b) : mkLayerSizes (b : t)
      mkLayerSizes _ = []

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where revApply x f = f x


loadFastTextVec :: FilePath -> IO (Map.Map String [Float])
loadFastTextVec path = do
  contents <- TIO.readFile path
  let ls = drop 1 $ T.lines contents
  let entries = map parseLine ls
  return $ Map.fromList $ catMaybes entries

parseLine :: T.Text -> Maybe (String, [Float])
parseLine line =
  case T.words line of
    [] -> Nothing
    (w:vals) -> Just (T.unpack w, map (realToFrac . fst . parseFloat) vals)

parseFloat :: T.Text -> (Double, T.Text)
parseFloat t = case TR.double t of
  Right x -> x
  Left _ -> error $ "Can't parse float: " ++ T.unpack t


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
                    in (inputVec, [label])
                  ) v
          (xs, ys) = unzip rows
      return (asTensor xs, asTensor ys)


trainMLP :: MLP -> Tensor -> Tensor -> IO MLP
trainMLP initModel inputs targets = do
  (trainedModel, _) <- foldLoop (initModel, []) 200 $ \(state, losses) i -> do
    let yPred = sigmoid (mlp state inputs)
        loss = binaryCrossEntropyLoss' yPred targets
        lossValue = asValue loss :: Float
    -- putStrLn $ "ypred " ++ show i ++ ": " ++ show yPred
    -- putStrLn $ "inputs " ++ show i ++ ": " ++ show inputs

    when (i `mod` 50 == 0) $ putStrLn $ "Iter " ++ show i ++ ": Loss = " ++ show lossValue
    (newState, _) <- runStep state GD loss 1e-5
    return (newState, losses ++ [lossValue])
  return trainedModel

evaluate :: MLP -> Tensor -> Tensor -> IO ()
evaluate model inputs targets = do
  let preds = sigmoid (mlp model inputs)
      predLabels = gt preds 0.5
      trueLabels = gt targets 0.5
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
  putStrLn "Loading embeddings..."
  embMap <- loadFastTextVec "data/MLP/entity_vector.model.test.txt"

  putStrLn "Loading training data..."
  (trainX, trainY) <- loadWordPairData "data/MLP/train_small_test.csv" embMap
  putStrLn $ "trainx shape: " ++ show (shape trainX)

  putStrLn "Initializing model..."
  initModel <- sample $ MLPSpec
    { feature_counts = [400, 128, 64, 1],  -- 200 + 200 = 400
      nonlinearitySpec = relu
    }

  putStrLn "Training..."
  model <- trainMLP initModel trainX trainY

  putStrLn "Loading eval data..."
  (evalX, evalY) <- loadWordPairData "data/MLP/eval_small.csv" embMap

  putStrLn "Evaluating..."
  evaluate model evalX evalY
