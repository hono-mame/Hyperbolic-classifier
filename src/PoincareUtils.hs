module PoincareUtils
    (initializeEmbeddings, printEmbeddings, readWordsFromCSV, readPairsFromCSV,
     poincareDistance, distanceBetweenWords, projectToBall,
     runStepRSGD, train, computeDatasetLoss, saveEmbeddings,
     makeBatches, sampleNegatives, runStepRSGDBatch, pairLossTensor, trainBatch,
     Entity, Embedding, Embeddings
    ) where

import System.Random
import System.IO
import qualified Data.Map as M
import qualified Data.Set as S
import Data.List.Split (splitOn)
import Control.Monad (foldM, when)
import Data.Maybe (fromJust)
import Prelude hiding (sqrt, acosh, exp, log)
import Data.Int()
import Torch.Tensor (Tensor, asTensor, asValue, select, shape)
import Torch.Functional (sumAll, sqrt, Dim (..), stack, exp, log)
import Torch.TensorFactories (randnIO')
import Torch.Functional.Internal (acosh)
import Torch.Optim (Gradients (..), grad')
import Torch.Device()
import Torch.Autograd (makeIndependent, toDependent)
import Torch.DType()

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

makeBatches :: Int -> [a] -> [[a]]
makeBatches batchSize [] = []
makeBatches batchSize xs =
  let (batch, rest) = splitAt batchSize xs
  in batch : makeBatches batchSize rest

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
      y = x
      xValRaw = asValue y :: Float
      xClipped = if xValRaw < 1.0 + eps
                    then x - x + asTensor (1.0 + eps)
                    else x
  in acosh xClipped

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
    then
      if normScalar == 0.0
        then thetaT
         else (thetaT / asTensor normScalar) * asTensor (1.0 - eps)
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

runStepRSGD :: Float -- learningRate
            -> Int -- negK
            -> [(String,String)] -- pairs
            -> Embeddings
            -> IO Embeddings
runStepRSGD learningRate negK pairs embeddings = do
  foldM updateOne embeddings pairs
  where
    updateOne embs (u, v) = do
      let embList = M.toList embs
          n = length embList
          idxMap = M.fromList $ zip (map fst embList) [0..]

      -- 全単語ベクトルをまとめてテンソル化
      let allVecs = map snd embList
          stacked = stack (Dim 0) allVecs
      
      -- 勾配計算用に独立したテンソルを作成
      independentStacked <- makeIndependent stacked
      let depStacked = toDependent independentStacked
      
      -- u, vのインデックスを取得
      let idxU = fromJust $ M.lookup u idxMap
          idxV = fromJust $ M.lookup v idxMap
      -- putStrLn $ "----------------------------------------"
      -- putStrLn $ "Processing pair: (" ++ show idxU ++ ", " ++ show idxV ++ ")"
      -- putStrLn $ "pair: (" ++ u ++ ", " ++ v ++ ")"

      -- ネガティブサンプルをランダムに選ぶ
      negIdxs <- sampleNegatives n negK (fromIntegral idxV)
      let candIdxs = (fromIntegral idxV) : map fromIntegral negIdxs

      -- uベクトルと候補(v+negatives)ベクトルを取得
      let vecU = select 0 idxU depStacked
          vecCandidates = map (\i -> select 0 i depStacked) candIdxs
      
      let dists = map (poincareDistance vecU) vecCandidates
          negDists = map (\dt -> exp (asTensor (0.0 :: Float) - dt)) dists
          sumExp = sumAll (stack (Dim 0) negDists)
          dPos = head dists
          lossTensor = dPos + log sumExp

      -- putStrLn $ "distance: " ++ show dists
      -- 勾配計算
      let Gradients grads = grad' lossTensor [independentStacked]
      -- putStrLn $ "grads: " ++ show grads

      case grads of
        (gradTensor : _) -> do
          updatedResults <- mapM (\(word, vec) -> do
            let i = fromJust $ M.lookup word idxMap
                gVec = select 0 i gradTensor
                depVec = select 0 i depStacked
                normSquared = sumAll (depVec * depVec)
                tmp = 1.0 - asValue normSquared :: Float
                coeff = (asTensor tmp * asTensor tmp) / asTensor (4.0 :: Float)
                riemGrad = coeff * gVec
                lrTensor = asTensor (learningRate :: Float)
                updated = vec - lrTensor * riemGrad 
                projected = projectToBall updated
            return (word, projected)
            ) embList
          return $ M.fromList updatedResults
        [] -> error "Empty gradient list"

runStepRSGDBatch :: Float-- learningRate
                   -> Int -- negK
                  -> [[(String,String)]] -- batched pairs
                  -> Embeddings -- initial embeddings
                  -> IO Embeddings -- updated embeddings
runStepRSGDBatch learningRate negK batches embeddings =
  foldM updateBatch embeddings batches
  where
    updateBatch embs batch = do
      let embList = M.toList embs
          idxMap = M.fromList $ zip (map fst embList) [0..]
          stacked = stack (Dim 0) (map snd embList)
      independentStacked <- makeIndependent stacked
      let depStacked = toDependent independentStacked

      -- バッチ内の全ペアの損失を足す
      lossTensors <- mapM (pairLossTensor depStacked idxMap negK) batch
      let totalLoss = sumAll (stack (Dim 0) lossTensors)

      -- 勾配計算
      let Gradients grads = grad' totalLoss [independentStacked]

      case grads of
        (gradTensor : _) -> do
          updatedResults <- mapM (\(word, vec) -> do
              let i = fromJust $ M.lookup word idxMap
                  gVec = select 0 i gradTensor
                  depVec = select 0 i depStacked
                  normSquared = sumAll (depVec * depVec)
                  tmp = 1.0 - asValue normSquared :: Float
                  coeff = (asTensor tmp * asTensor tmp) / asTensor (4.0 :: Float)
                  riemGrad = coeff * gVec
                  lrTensor = asTensor learningRate
                  updated = vec - lrTensor * riemGrad
                  projected = projectToBall updated
              return (word, projected)
            ) embList
          return $ M.fromList updatedResults
        [] -> error "Empty gradient list"

pairLossTensor :: Tensor                        -- stacked embeddings
                -> M.Map String Int              -- 単語→インデックスのマップ
                -> Int                           -- negK
                -> (String, String)              -- (u,v)
                -> IO Tensor                     -- 損失テンソル
pairLossTensor depStacked idxMap negK (u,v) = do
  let idxU = fromJust $ M.lookup u idxMap
      idxV = fromJust $ M.lookup v idxMap
      n = shape depStacked !! 0
  negIdxs <- sampleNegatives n negK (fromIntegral idxV)
  let candIdxs = (fromIntegral idxV) : map fromIntegral negIdxs
      vecU = select 0 idxU depStacked
      vecCandidates = map (\i -> select 0 i depStacked) candIdxs
      dists = map (poincareDistance vecU) vecCandidates
      negDists = map (\dt -> exp (asTensor (0.0 :: Float) - dt)) dists
      sumExp = sumAll (stack (Dim 0) negDists)
      dPos = head dists
  return $ dPos + log sumExp

train :: Int      -- epochs
      -> Float    -- base learning rate η
      -> Int      -- neg samples per positive k
      -> Int      -- burn-in coefficient c (use η/c during burn-in)
      -> Int      -- burn-in epochs
      -> [(String,String)] -- input pairs
      -> Embeddings -- initial embeddings
      -> IO (Embeddings, [Float]) -- trained embeddings + loss history
train epochs baseLR negK burnC burnEpochs pairs embs0 =
  foldM step (embs0, []) [1..epochs]
  where
    step (embs, losses) epoch = do
      let lr = if epoch <= burnEpochs then baseLR / fromIntegral burnC else baseLR
      newEmbs <- runStepRSGD lr negK pairs embs
      lossVal <- computeDatasetLoss newEmbs pairs negK
      when (epoch `mod` 1 == 0) $
        putStrLn $ "Epoch " ++ show epoch ++ "  lr=" ++ show lr ++ "  Loss=" ++ show lossVal
      return (newEmbs, losses ++ [lossVal])

trainBatch :: Int      -- epochs
      -> Float    -- base learning rate η
      -> Int      -- neg samples per positive k
      -> Int      -- burn-in coefficient c (use η/c during burn-in)
      -> Int      -- burn-in epochs
      -> Int      -- batch size
      -> [(String,String)] -- input pairs
      -> Embeddings -- initial embeddings
      -> IO (Embeddings, [Float]) -- trained embeddings + loss history
trainBatch epochs baseLR negK burnC burnEpochs batchSize pairs embs0 =
  foldM step (embs0, []) [1..epochs]
  where
    step (embs, losses) epoch = do
      let lr = if epoch <= burnEpochs then baseLR / fromIntegral burnC else baseLR
          batches = makeBatches batchSize pairs
      newEmbs <- runStepRSGDBatch lr negK batches embs
      lossVal <- computeDatasetLoss newEmbs pairs negK
      when (epoch `mod` 1 == 0) $
        putStrLn $ "Epoch " ++ show epoch ++ "  lr=" ++ show lr ++ "  Loss=" ++ show lossVal
      return (newEmbs, losses ++ [lossVal])

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

saveEmbeddings :: FilePath -> Embeddings -> IO ()
saveEmbeddings path embs = do
  let dimNum = case M.elems embs of
                  (v:_) -> head (shape v)
                  []    -> 0
      header = "word," ++ concat (map (\i -> "dim" ++ show i ++ if i == dimNum then "" else ",") [1..dimNum])
      linesText = map (\(word, vec) -> 
                        let vecList :: [Float]
                            vecList = asValue vec
                        in word ++ "," ++ concat (map (\(x,i) -> show x ++ if i == dimNum then "" else ",") (zip vecList [1..]))
                      ) $ M.toList embs
      csvText = unlines (header : linesText)
  writeFile path csvText