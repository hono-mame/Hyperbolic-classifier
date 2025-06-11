{-# LANGUAGE OverloadedStrings #-}

import Database.SQLite.Simple
import Control.Monad (forM_)
import Tree (numNodes)

-- Define the type for a word pair
type WordPair = (String, String)

-- Define the query to retrieve pairs of hypernyms and hyponyms
sqlQuery :: Query
sqlQuery = "SELECT w1.lemma, w2.lemma \
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
           \LIMIT 10;"

main :: IO ()
main = do
  database <- open "data/wnjpn.db"
  results <- query_ database sqlQuery :: IO [WordPair]
  forM_ results $ \(hyper, hypo) -> do
    putStrLn $ hyper ++ " -> " ++ hypo
  close database