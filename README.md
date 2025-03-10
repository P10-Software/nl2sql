# nl2sql
Main repo for P10 contribution

# Setup
1. Add the .env file from Teams at the root level of the repo.
2. Add the column and table name CSV files from Teams to a .local folder at the root level.
3. Run the script to setup the databases:

```
python -m src.database.setup_database
```

# Running benchmark
1. Ensure that the setup steps have been followed
2. Update main.py to use correct dataset, prompt_strategy and NL2SQL model
3. Execute:
```
python -m main
```
