# nl2sql
Main repo for P10 contribution

# Setup
1. Clone submodules using the following command: 
```git clone --recursive-submodules https://github.com/P10-Software/nl2sql.git```
If the repo is already cloned without the submodules use this command instead:
```git submodule update --init --recursive```
2. Add the .env file from Teams at the root level of the repo.
3. Add the column and table name CSV files from Teams to a .local folder at the root level.
4. Run the script to setup the databases:
```python -m src.database.setup_database```

# Running benchmark
1. Ensure that the setup steps have been followed
2. Update main.py to use correct dataset, prompt_strategy and NL2SQL model
3. Execute ```python -m main```
