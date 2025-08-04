module Evaluation (confusionMatrix, confusionMatrixPairs, evalAccuracy, evalPrecision, evalRecall, calcF1, evalMacroF1, evalWeightedF1, evalMicroF1) where

import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, sub)
import Control.Monad (foldM)

evalAccuracy ::
    Tensor -> -- True Positive
    Tensor -> -- True Negative
    Tensor -> -- False Positive
    Tensor -> -- False Negative
    Tensor -- Accuracy
evalAccuracy tp tn fp fn =
    let total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
    in accuracy

evalPrecision ::
    Tensor -> -- True Positive
    Tensor -> -- False Positive
    Tensor -- Precision
evalPrecision tp fp =
    let total = tp + fp
        precision = tp / total
    in precision

evalRecall ::
    Tensor -> -- True Positive
    Tensor -> -- False Negative
    Tensor -- Recall
evalRecall tp fn =
    let total = tp + fn
        recall = tp / total
    in recall

confusionMatrix ::
    Tensor -> -- True Positive
    Tensor -> -- True Negative
    Tensor -> -- False Positive
    Tensor -> -- False Negative
    [[Tensor]]
confusionMatrix tp tn fp fn = [
        [tp, fn],
        [fp, tn]
    ]

confusionMatrixPairs :: [[Tensor]] -> [(Tensor, Tensor, Tensor, Tensor)]
confusionMatrixPairs [[tp, fn], [fp, tn]] = [(tp, tn, fp, fn), (tn, tp, fn, fp)]

calcF1 :: Tensor -> Tensor -> Tensor
calcF1 precision recall = 2 * precision * recall / (precision + recall)


evalMacroF1 :: 
    [(Tensor, Tensor, Tensor, Tensor)] -> 
    Tensor
evalMacroF1 confusionData = 
    let f1Scores = map calcF1ForClass confusionData
        numClasses = fromIntegral (length confusionData) :: Float -- Float型にキャスト
        sumF1 = sum f1Scores
        averageF1 = sumF1 / asTensor (numClasses :: Float) -- Tensorに明示的にキャスト
    in averageF1
  where
    calcF1ForClass (tp, _, fp, fn) = 
        let precision = evalPrecision tp fp
            recall = evalRecall tp fn
        in calcF1 precision recall

evalWeightedF1 :: 
    [(Tensor, Tensor, Tensor, Tensor)] -> 
    [Tensor] -> -- support (actual counts per class)
    Tensor
evalWeightedF1 confusionData support = 
    let f1Scores = map calcF1ForClass confusionData
        totalSupport = sum support
        frequencies = map (/ totalSupport) support
        weightedSum = sum $ zipWith (*) f1Scores frequencies
    in weightedSum
  where
    calcF1ForClass (tp, _, fp, fn) = 
        let precision = evalPrecision tp fp
            recall = evalRecall tp fn
        in calcF1 precision recall


evalMicroF1 :: 
    [(Tensor, Tensor, Tensor, Tensor)] -> 
    Tensor
evalMicroF1 confusionData = 
    let (tps, tns, fps, fns) = foldl accumData (0, 0, 0, 0) confusionData
        microPrecision = evalPrecision tps fps
        microRecall = evalRecall tps fns
        f1Score = calcF1 microPrecision microRecall
    in f1Score
  where
    accumData (accTp, accTn, accFp, accFn) (tp, tn, fp, fn) =
        (accTp + tp, accTn + tn, accFp + fp, accFn + fn)