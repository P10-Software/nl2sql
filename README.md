# nl2sql
Main repo for P10 contribution

# Setup
Follow these steps to setup database and files needed to run the benchmark:
1. Add the .env file from Teams at the root level of the repo.
2. Add the .local folder from teams to the root of your project
3. Install requirements for the project (Remember to do it each time in UCLOUD)
```
pip install -r requirements
```

# Manual Setup Files and Database
Here is the manual steps to create database files, this can be used instead of 
downloading from teams or to update if changes are needed

### Setup the databases
Run the following script to set up the natural and abbreviated databases.
Make sure to have the the following data in your .local before running:
- Folder with SAS database files.
- column_names_natural.csv
- tables_names_natural.csv
```
python -m src.database.setup_database
```

### Generate M-Schema
Run the following script to generate the M-Schema files for both databases
Ensure that you have the databases

```
python -m scripts.generate_mschema
```

### Nullable columns 
Script to generate nullable colums for creating DDL instructions.

```
python -m scripts.nullable_columns
```

# Running benchmark
1. Ensure that the setup steps have been followed
2. Update main.py to use correct dataset, prompt_strategy and NL2SQL model
3. Update .env to chose abbreviated or natural database
4. Execute:
```
python -m main
```
