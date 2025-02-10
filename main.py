from libs.SNAILS.snails_naturalness_classifier import CanineIdentifierClassifier

def evaluate_naturalness(indentifiers):
    classifier = CanineIdentifierClassifier()

    classifications = dict()
    for identifier in indentifiers:
        classifications[identifier] = classifier.classify_identifier(identifier)[0]["label"]

    distribution_of_classifications = {"distribution": {
        "N1": len([value for value in classifications.values() if value == "N1"]),
        "N2": len([value for value in classifications.values() if value == "N2"]),
        "N3": len([value for value in classifications.values() if value == "N3"])
    }}
    
    return distribution_of_classifications | classifications

def main():
    #get table names
    #get column names

    example_identifiers = {"trl", "trial", "winther", "weather", "dr", "danmarks radio"}
    naturalness_results = evaluate_naturalness(example_identifiers)
    print(naturalness_results)

if __name__ == "__main__":
    main()
