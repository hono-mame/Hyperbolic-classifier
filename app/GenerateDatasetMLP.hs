{-# LANGUAGE OverloadedStrings #-}

module GenerateDatasetMLP (main) where
import MLPDataGenerateUtils (generateLabeledDataset, labeledPairToCSV, saveSamplesToFile, LabeledPair)
import System.IO (writeFile)
import Data.List (intercalate)

main :: IO ()
main = do
    let inputFilePath = "data/tree.edges"
        trainOutputPath = "data/MLP/train.csv"
        evalOutputPath = "data/MLP/eval.csv"
        negativeSamplesFactor = 2
        trainRatio = 0.8

    putStrLn "Starting dataset generation and splitting..."
    fullDataset <- generateLabeledDataset inputFilePath negativeSamplesFactor
    let totalSamples = length fullDataset
        numTrainSamples = floor (fromIntegral totalSamples * trainRatio)
        trainSamples = take numTrainSamples fullDataset
        evalSamples = drop numTrainSamples fullDataset

    putStrLn $ "\nSplitting dataset:"
    putStrLn $ "  Total samples: " ++ show totalSamples
    putStrLn $ "  Training samples: " ++ show (length trainSamples)
    putStrLn $ "  Evaluation samples: " ++ show (length evalSamples)
    putStrLn ""

    saveSamplesToFile trainOutputPath trainSamples
    saveSamplesToFile evalOutputPath evalSamples
    putStrLn "Dataset generation and splitting complete."