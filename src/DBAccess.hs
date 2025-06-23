{-# LANGUAGE OverloadedStrings #-}

module DBAccess
  ( fetchWordPairs
  , fetchWordPairsFull
  , countWordPairs
  , WordPair
  ) where

import Data.String (fromString)
import Database.SQLite.Simple
-- https://hackage.haskell.org/package/sqlite-simple-0.4.19.0/docs/Database-SQLite-Simple.html

type WordPair = (String, String)

fetchWordPairs :: Int -> IO [WordPair]
fetchWordPairs limit = do
  db <- open "data/wnjpn.db"
  let sqlQuery = Query $ "SELECT w1.lemma, w2.lemma \
           \FROM synlink AS sl \
           \INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1 \
           \INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2 \
           \INNER JOIN sense AS se1 ON se1.synset = sy1.synset \
           \INNER JOIN sense AS se2 ON se2.synset = sy2.synset \
           \INNER JOIN word AS w1 ON w1.wordid = se1.wordid \
           \INNER JOIN word AS w2 ON w2.wordid = se2.wordid \
           \WHERE sl.link = 'hypo' \
           \AND se1.lang = 'jpn' AND se2.lang = 'jpn' \
           \AND w1.lang = 'jpn' AND w2.lang = 'jpn' \
           \LIMIT " <> (fromString . show) limit <> ";"
  results <- query_ db sqlQuery
  close db
  return results

fetchWordPairsFull :: IO [WordPair]
fetchWordPairsFull = do
  db <- open "data/wnjpn.db"
  let sqlQuery = Query $ "SELECT w1.lemma, w2.lemma \
           \FROM synlink AS sl \
           \INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1 \
           \INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2 \
           \INNER JOIN sense AS se1 ON se1.synset = sy1.synset \
           \INNER JOIN sense AS se2 ON se2.synset = sy2.synset \
           \INNER JOIN word AS w1 ON w1.wordid = se1.wordid \
           \INNER JOIN word AS w2 ON w2.wordid = se2.wordid \
           \WHERE sl.link = 'hypo' \
           \AND se1.lang = 'jpn' AND se2.lang = 'jpn' \
           \AND w1.lang = 'jpn' AND w2.lang = 'jpn' \
           \LIMIT 935992;"
  results <- query_ db sqlQuery
  close db
  return results

countWordPairs :: IO Int
countWordPairs = do
  db <- open "data/wnjpn.db"
  [Only n] <- query_ db $ Query
    "SELECT COUNT(*) \
    \FROM synlink AS sl \
    \INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1 \
    \INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2 \
    \INNER JOIN sense AS se1 ON se1.synset = sy1.synset \
    \INNER JOIN sense AS se2 ON se2.synset = sy2.synset \
    \INNER JOIN word AS w1 ON w1.wordid = se1.wordid \
    \INNER JOIN word AS w2 ON w2.wordid = se2.wordid \
    \WHERE sl.link = 'hypo' \
    \AND se1.lang = 'jpn' AND se2.lang = 'jpn' \
    \AND w1.lang = 'jpn' AND w2.lang = 'jpn';"
  close db
  return n
