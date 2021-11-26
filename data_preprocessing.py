import csv

# load dataset
# *** simply replace the *.csv path to a different csv to apply this preprocessing to a different file
dataset_file = open('scraped_tweets_pos/iota_tweets.csv', 'r', encoding='utf-8')
dataset = dataset_file.readlines()

# #remove words that begin with @ and https://
pruned_dataset = [' '.join([word for word in text.split() if not word.startswith('@') and not word.startswith('https://')]) for text in dataset]

#remove all characters except letters and numbers and the space character
pruned_dataset = [''.join(list([char for char in text if char.isalpha() or char.isnumeric() or char == ' '])) for text in pruned_dataset]

# open file in write mode
# *** the following directory and file need to be created in order for the code to function properly
file = open('scraped_tweets_pos/iota_tweets_pruned.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(file)

# output tweets that have a length greater than 0
for tweet in pruned_dataset:
    if len(tweet) > 0:
        # remove extra spaces in string
        twee_formated = ' '.join(tweet.split())

        # remove leading and trailing spaces and write to csv
        writer.writerow([twee_formated.strip()])

# close file
file.close()