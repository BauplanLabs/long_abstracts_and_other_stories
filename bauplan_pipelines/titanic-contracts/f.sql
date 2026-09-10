-- bauplan: name=f
-- bauplan: materialization_strategy=REPLACE
-- bauplan: output_schema=WomenByClass
  SELECT PassengerId AS pid, Pclass AS tclass
    FROM titanic
   WHERE Pclass IN (1, 2) AND Sex = 'female'

