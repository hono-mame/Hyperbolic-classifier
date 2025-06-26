module MLPDataGenerateUtils
  ( generateLabeledDataset
  , saveSamplesToFile
  , labeledPairToCSV
  , saveSamplesToFile
  , LabeledPair
  ) where

import qualified Data.Map as Map
import qualified Data.Set as Set
import Data.List (nub, intercalate)
import System.Random (randomRIO, newStdGen)
import System.Random.Shuffle (shuffle') 
import Control.Monad (replicateM)
import System.IO (writeFile)
import TreeUtils

type LabeledPair = (String, String, Int)
type ParentMap = Map.Map String String

findLowestCommonAncestor :: 
  ParentMap -> 
  String -> 
  String -> 
  String
findLowestCommonAncestor pmap nodeA nodeB =
    let pathA = reverse $ pathToRoot pmap nodeA
        pathB = reverse $ pathToRoot pmap nodeB
        commonPrefix = takeWhile (uncurry (==)) $ zip pathA pathB
    in fst (last commonPrefix)


generateAllPositiveHypernymHyponymPairs :: 
  ParentMap -> 
  ChildMap -> 
  [String] -> -- list of all words in the dataset
  [LabeledPair]
generateAllPositiveHypernymHyponymPairs pmap cmap allWords =
    nub -- eliminate duplicates
    [ (hypernym, hyponym, 1)
    | hypernym <- allWords
    , hyponym <- allWords
    , hypernym /= hyponym -- eliminate self-pairs
    , hypernym /= "entity" && hyponym /= "entity" -- remove pairs that include "entity"
    , let hyponymPath = pathToRoot pmap hyponym
    , hypernym `elem` hyponymPath -- hyponym's path includes hypernym
    , let currentCommonAncestor = findLowestCommonAncestor pmap hypernym hyponym
    , currentCommonAncestor == hypernym -- hypernym is hyponym's ancestor
    , not ("entity" `elem` dropWhile (/= hypernym) (reverse hyponymPath)) -- direct ancestor check
    ]

isHypernymOfTransitive ::
  ParentMap ->
  String -> -- potential hypernym
  String -> -- potential hyponym
  Bool
isHypernymOfTransitive pmap potentialHypernym potentialHyponym =
    let path = pathToRoot pmap potentialHyponym -- get the path from potentialHyponym to the root
    in potentialHypernym `elem` path && potentialHypernym /= potentialHyponym -- check if potentialHypernym is in the path and not equal to potentialHyponym

generateSingleNegativeSample ::
  ParentMap ->
  ChildMap ->
  [String] ->
  IO LabeledPair
generateSingleNegativeSample pmap cmap allWords = do
    idx1 <- randomRIO (0, length allWords - 1)
    idx2 <- randomRIO (0, length allWords - 1)
    let word1 = allWords !! idx1
        word2 = allWords !! idx2

    if word1 /= word2 &&
       word1 /= "entity" && word2 /= "entity" &&
       not (isHypernymOfTransitive pmap word1 word2) &&
       not (isHypernymOfTransitive pmap word2 word1)
        then return (word1, word2, 0)
        else generateSingleNegativeSample pmap cmap allWords -- recursively generate until a valid negative sample is found


generateNegativeSamples ::
  ParentMap ->
  ChildMap ->
  [String] ->
  Int ->
  IO [LabeledPair]
generateNegativeSamples pmap cmap allWords numSamples =
    replicateM numSamples (generateSingleNegativeSample pmap cmap allWords)

generateLabeledDataset ::
  FilePath ->
  Int -> -- numNegativeSamplesFactor (how many negative samples to generate per positive sample)
  IO [LabeledPair]
generateLabeledDataset edgesFilePath numNegativeSamplesFactor = do
    pairs <- loadEdges edgesFilePath
    let cmap = buildParentChildMap pairs
        pmap = buildParentMap pairs

    let allWords = nub $ concatMap (\(p, c) -> [p, c]) pairs
        filteredWords = filter (/= "entity") allWords

    let positiveSamples = generateAllPositiveHypernymHyponymPairs pmap cmap filteredWords
    putStrLn $ "Generated " ++ show (length positiveSamples) ++ " positive samples (including transitive relations without 'entity')."

    let numNegativeToGenerate = length positiveSamples * numNegativeSamplesFactor
    negativeSamples <- generateNegativeSamples pmap cmap filteredWords numNegativeToGenerate
    putStrLn $ "Generated " ++ show (length negativeSamples) ++ " negative samples."

    let allSamples = positiveSamples ++ negativeSamples
    putStrLn $ "Total samples generated: " ++ show (length allSamples)

    -- shuffle the samples
    gen <- newStdGen -- generate a new random number generator
    let shuffledSamples = shuffle' allSamples (length allSamples) gen
    return shuffledSamples

labeledPairToCSV :: LabeledPair -> String
labeledPairToCSV (word1, word2, label) =
    word1 ++ "," ++ word2 ++ "," ++ show label

saveSamplesToFile :: FilePath -> [LabeledPair] -> IO ()
saveSamplesToFile outputPath samples = do
    let csvLines = map labeledPairToCSV samples
    writeFile outputPath (unlines csvLines)
    putStrLn $ "Successfully saved " ++ show (length samples) ++ " samples to " ++ outputPath