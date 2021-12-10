import csv


def preprocessing(from_path, to_path, to_lower=True):
    # load dataset
    # *** simply replace the *.csv path to a different csv to apply this preprocessing to a different file
    dataset_file = open(from_path, 'r', encoding='utf-8')
    dataset = dataset_file.readlines()

    # remove words that begin with @ and https://
    pruned_dataset = []
    for i, text in enumerate(dataset):
        str_builder = []
        for word in text.split():
            if not word.startswith('@'):
                if 'http' not in word:
                    str_builder.append(word)
                elif not word.startswith('http'):
                    str_builder.append(word.split('http')[0])

        pruned_dataset.append(' '.join(str_builder))

    # remove all characters except letters and numbers and the space character
    vocabulary = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    if to_lower:
        pruned_dataset = [''.join(list([char.lower() for char in text if char in vocabulary or char == ' '])) for text
                          in pruned_dataset]
    else:
        pruned_dataset = [''.join(list([char for char in text if char in vocabulary or char == ' '])) for text in
                          pruned_dataset]

    # open file in write mode
    # *** the following directory and file need to be created in order for the code to function properly
    file = open(to_path, 'w', encoding='utf-8', newline='')
    writer = csv.writer(file)

    # output tweets that have a length greater than 0
    for i, tweet in enumerate(pruned_dataset):
        # remove extra spaces in string
        tweet_formated = ' '.join(tweet.split())

        if tweet_formated:
            # remove leading and trailing spaces and write to csv
            writer.writerow([tweet_formated.strip()])

    # close file
    file.close()
