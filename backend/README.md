First install mongodb:
```
brew tap mongodb/brew
brew install mongodb-community
brew install mongodb-database-tools
brew services start mongodb-community
```

Testing steps using local database:
load sample data into MongoDB:
```
mongoimport --db electionDB --collection cards --file backend/cards.json --jsonArray
```

Verify the data is correctly imported:
```
mongosh
use electionDB
db.cards.find().pretty()
```

Finally, you can run the backend by:
```
python app.py
```
