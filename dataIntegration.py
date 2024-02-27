import pandas as pd
import gzip
import json


def select_the_dataset(n):
    if n == '1':
        # Read the data 'Twitch'
        df = pd.read_csv("data/Twitch/100k_a.csv", header=None)
        df.columns = ['User_ID', 'Stream_ID', 'Streamer_username', 'Time_start', 'Time_stop']

    if n == '2':
        # Read the data 'NPR'
        episodes = pd.read_csv("data/NPR/episodes.csv", header=0)
        utterances = pd.read_csv("data/NPR/utterances.csv", header=0)
        headlines = pd.read_csv("data/NPR/headlines.csv", header=0)
        # Rename to avoid creating duplicate matching key columns in the result
        episodes_renamed = episodes.rename(columns={'id': 'episode', 'episode_date': 'date'})
        df = episodes_renamed.merge(utterances, on='episode')

    if n == '3':
        # Read the data 'This American Life'
        def extract_data_from_json(json_data, columns_needed):
            data = []
            for episode, records in json_data.items():
                for record in records:
                    row = {col: record.get(col, None) for col in columns_needed}
                    data.append(row)
            dataframe = pd.DataFrame(data)
            return dataframe

        with open("data/This American Life/train-transcripts-aligned.json") as f:
            train_transcripts = json.load(f)
        with open("data/This American Life/valid-transcripts-aligned.json") as f:
            valid_transcripts = json.load(f)
        with open("data/This American Life/test-transcripts-aligned.json") as f:
            test_transcripts = json.load(f)

        columns_needed = ['episode', 'act', 'utterance_start', 'utterance_end', 'duration', 'speaker', 'utterance']
        train_transcripts = extract_data_from_json(train_transcripts, columns_needed)
        valid_transcripts = extract_data_from_json(valid_transcripts, columns_needed)
        test_transcripts = extract_data_from_json(test_transcripts, columns_needed)
        df = pd.concat([train_transcripts, valid_transcripts, test_transcripts])

    if n == '4':
        # Read the data 'Recipes'
        reviews = pd.read_csv("data/Recipes/RAW_interactions.csv", header=0)
        recipes = pd.read_csv("data/Recipes/RAW_recipes.csv", header=0)
        recipes_renamed = recipes.rename(columns={'id': 'recipe_id'})
        df = reviews.merge(recipes_renamed, on='recipe_id')

    if n == '5':
        # Read the data 'Paired Recipes'
        recipes = pd.read_parquet("data/Paired Recipes/recipes.parquet")
        pairs = pd.read_parquet("data/Paired Recipes/pairs.parquet")

        recipes_base = recipes.rename(columns=lambda x: f'{x}_base' if x != 'id' else 'base')
        recipes_target = recipes.rename(columns=lambda x: f'{x}_target' if x != 'id' else 'target')

        pairs_base = pairs.merge(recipes_base, on='base')
        df = pairs_base.merge(recipes_target, on='target')

    if n == '6':
        # Read the data 'EndoMondo'
        with gzip.open("data/EndoMondo/endomondoHR.json.gz") as f:
            endomondoHR = [eval(line) for line in f]
        df = pd.DataFrame(endomondoHR)

    if n == '7':
        # Read the data 'Amazon Product Reviews'
        with open("data/Amazon Product Reviews/Video_Games_5.json") as f:
            Video_Games = [json.loads(line) for line in f]
        df = pd.DataFrame(Video_Games)

    if n == '8':
        # Read the data 'Amazon Question and Answer'
        with open("data/Amazon Question and Answer/qa_Video_Games.json") as f:
            qa_Video_Games = [eval(line) for line in f]
        qa_Video_Games = pd.DataFrame(qa_Video_Games)

        QA_Video_Games = []
        with open("data/Amazon Question and Answer/QA_Video_Games copy.json") as f:
            for line in f:
                json_data = eval(line)

                for question in json_data['questions']:
                    for answer in question.get('answers', []):
                        row = {
                            'asin': json_data['asin'],
                            'questionType': question['questionType'],
                            'askerID': question.get('askerID', ''),
                            'questionTime': question.get('questionTime', ''),
                            'questionText': question.get('questionText', ''),
                            'answerText': answer.get('answerText', ''),
                            'answererID': answer.get('answererID', ''),
                            'answerTime': answer.get('answerTime', ''),
                            'helpful': answer.get('helpful', [])
                        }
                        QA_Video_Games.append(row)

        QA_Video_Games = pd.DataFrame(QA_Video_Games)

        qa_Video_Games = qa_Video_Games.rename(
            columns={'question': 'questionText', 'answer': 'answerText', 'unixTime': 'answerUnixTime'}
        )
        df = pd.concat([qa_Video_Games, QA_Video_Games], ignore_index=True, sort=False)
        df = df[['asin', 'questionType', 'askerID', 'questionTime', 'questionText',
                 'answerType', 'answererID', 'answerTime', 'answerUnixTime', 'answerText', 'helpful']]

    if n == '9':
        # Read the data 'Amazon Marketing Bias'
        df = pd.read_csv("data/Amazon Marketing Bias/df_electronics.csv", header=0)

    if n == '10':
        # Read the data 'ModCloth Marketing Bias'
        df = pd.read_csv("data/ModCloth Marketing Bias/df_modcloth.csv", header=0)

    if n == '11':
        # Read the data 'Google Local'
        with open("data/Google Local/review-Hawaii_10.json") as f:
            review_Hawaii = [json.loads(line) for line in f]
        review_Hawaii = pd.DataFrame(review_Hawaii)
        review_Hawaii = review_Hawaii.rename(columns={'name': 'user_name'})
        with open("data/Google Local/meta-Hawaii.json") as f:
            meta_Hawaii = [json.loads(line) for line in f]
        meta_Hawaii = pd.DataFrame(meta_Hawaii)
        meta_Hawaii = meta_Hawaii.rename(columns={'name': 'business_name'})
        df = review_Hawaii.merge(meta_Hawaii, on='gmap_id')

    if n == '12':
        # Read the data 'Steam'
        user_reviews = []
        with open("data/Steam/australian_user_reviews.json") as f:
            for line in f:
                json_data = eval(line)

                user_id = json_data['user_id']
                user_url = json_data['user_url']
                # Process each review
                for review in json_data['reviews']:
                    row = {
                        'user_id': user_id,
                        'user_url': user_url,
                        'item_id': review['item_id'],
                        'review': review['review'],
                        'recommend': review['recommend'],
                        'posted': review['posted'],
                        'last_edited': review['last_edited'],
                        'funny': review['funny'],
                        'helpful': review['helpful'],
                    }
                    user_reviews.append(row)

        user_reviews = pd.DataFrame(user_reviews)

        users_items = []
        with open("data/Steam/australian_users_items.json") as f:
            for line in f:
                json_data = eval(line)

                user_id = json_data['user_id']
                items_count = json_data['items_count']
                steam_id = json_data['steam_id']
                user_url = json_data['user_url']
                # Process each item
                for item in json_data['items']:
                    row = {
                        'user_id': user_id,
                        'user_url': user_url,
                        'steam_id': steam_id,
                        'items_count': items_count,
                        'item_id': item['item_id'],
                        'item_name': item['item_name'],
                        'playtime_forever': item['playtime_forever'],
                        'playtime_2weeks': item['playtime_2weeks'],
                    }
                    users_items.append(row)

        users_items = pd.DataFrame(users_items)

        with open("data/Steam/steam_games.json") as f:
            steam_games = [eval(line) for line in f]
        steam_games = pd.DataFrame(steam_games)
        steam_games = steam_games.rename(columns={'id': 'item_id'})

        bundle_data = []
        with open("data/Steam/bundle_data.json") as f:
            for line in f:
                json_data = eval(line)

                bundle_final_price = json_data['bundle_final_price']
                bundle_url = json_data['bundle_url']
                bundle_price = json_data['bundle_price']
                bundle_name = json_data['bundle_name']
                bundle_id = json_data['bundle_id']
                bundle_discount = json_data['bundle_discount']
                # Process each item
                for item in json_data['items']:
                    row = {
                        'item_id': item['item_id'],
                        'bundle_id': bundle_id,
                        'bundle_name': bundle_name,
                        'bundle_url': bundle_url,
                        'bundle_price': bundle_price,
                        'bundle_discount': bundle_discount,
                        'item_discounted_price': item['discounted_price'],
                        'bundle_final_price': bundle_final_price,
                    }
                    bundle_data.append(row)

        bundle_data = pd.DataFrame(bundle_data)

        users_items_reviews = users_items.merge(user_reviews, on=['user_id', 'user_url', 'item_id'], how='left')
        users_items_reviews_steam_games = users_items_reviews.merge(steam_games, on='item_id', how='left')
        df = users_items_reviews_steam_games.merge(bundle_data, on='item_id', how='left')

    if n == '13':
        # Read the data 'Goodreads Book Reviews'
        with open("data/Goodreads Book Reviews/goodreads_books_poetry.json") as f:
            books = [json.loads(line) for line in f]
        books = pd.DataFrame(books)
        with open("data/Goodreads Book Reviews/goodreads_interactions_poetry.json") as f:
            interactions = [json.loads(line) for line in f]
        interactions = pd.DataFrame(interactions)
        with open("data/Goodreads Book Reviews/goodreads_reviews_poetry.json") as f:
            reviews = [json.loads(line) for line in f]
        reviews = pd.DataFrame(reviews)
        interactions_reviews = interactions.merge(reviews, on=['user_id', 'book_id', 'review_id', 'rating', 'date_added',
                                                               'date_updated', 'read_at', 'started_at'], how='left')
        df = interactions_reviews.merge(books, on='book_id', how='left')

    if n == '14':
        # Read the data 'Goodreads Spoilers'
        with open("data/Goodreads Spoilers/goodreads_reviews_spoiler.json") as f:
            reviews_spoiler = [json.loads(line) for line in f]
        df = pd.DataFrame(reviews_spoiler)

    if n == '15':
        # Read the data 'Pinterest'
        with open("data/Pinterest/fashion.json") as f:
            fashion = [json.loads(line) for line in f]
        fashion = pd.DataFrame(fashion)
        with open("data/Pinterest/fashion-cat.json") as f:
            fashion_cat = json.load(f)
        fashion_cat = pd.DataFrame(list(fashion_cat.items()), columns=['product', 'category'])
        with open("data/Pinterest/home.json") as f:
            home = [json.loads(line) for line in f]
        home = pd.DataFrame(home)
        with open("data/Pinterest/home-cat.json") as f:
            home_cat = json.load(f)
        home_cat = pd.DataFrame(list(home_cat.items()), columns=['product', 'category'])

        fashion_home = pd.concat([fashion, home])
        fashion_cat_home_cat = pd.concat([fashion_cat, home_cat])
        df = fashion_home.merge(fashion_cat_home_cat, on='product', how='left')

    if n == '16':
        # Read the data 'ModCloth Clothing Fit'
        with open("data/ModCloth Clothing Fit/modcloth_final_data.json") as f:
            modcloth_final_data = [json.loads(line) for line in f]
        df = pd.DataFrame(modcloth_final_data)

    if n == '17':
        # Read the data 'RentTheRunway Clothing Fit'
        with open("data/RentTheRunway Clothing Fit/renttherunway_final_data.json") as f:
            renttherunway_final_data = [json.loads(line) for line in f]
        df = pd.DataFrame(renttherunway_final_data)

    if n == '18':
        # Read the data 'Tradesy'
        tradesy = []
        with open("data/Tradesy/tradesy.json") as f:
            for line in f:
                json_data = eval(line)

                uid = json_data['uid']
                lists = json_data['lists']
                row = {
                    'uid': uid,
                    'selling': lists['selling'],
                    'sold': lists['sold'],
                    'want': lists['want'],
                    'bought': lists['bought'],
                }
                tradesy.append(row)

        df = pd.DataFrame(tradesy)

    if n == '19':
        # Read the data 'Behance'
        df = pd.DataFrame()

    if n == '20':
        # Read the data 'Librarything'
        # namespace = {}
        # with open("data/Librarything/reviews.txt", 'r') as f:
        #     exec(f.read(), namespace)
        #
        # reviews = namespace['reviews']
        #
        # data = []
        # for key, value in reviews.items():
        #     work, user = key
        #     record = value
        #     record.update({'work': work, 'user': user})
        #     data.append(record)
        #
        # reviews = pd.DataFrame(data)
        # reviews.to_json('data/Librarything/reviews.json', orient='records', lines=True)

        with open("data/Librarything/reviews.json") as f:
            reviews = [json.loads(line) for line in f]
        reviews = pd.DataFrame(reviews)
        reviews = reviews[['user', 'work', 'comment', 'time', 'unixtime', 'flags', 'stars', 'nhelpful']]
        edges = pd.read_csv("data/Librarything/edges.txt", sep=' ', header=None)
        edges.columns = ['user1', 'user2']

        network = {}
        for index, row in edges.iterrows():
            if row['user1'] not in network:
                network[row['user1']] = set()
            if row['user2'] not in network:
                network[row['user2']] = set()

            network[row['user1']].add(row['user2'])
            network[row['user2']].add(row['user1'])

        new_edges = []
        for user, connections in network.items():
            new_edges.append({'user': user, 'connections': list(connections)})

        new_edges = pd.DataFrame(new_edges)
        df = reviews.merge(new_edges, on='user', how='left')

    if n == '21':
        # Read the data ‘RateBeer Multi-aspect Reviews’ and 'BeerAdvocate Multi-aspect Reviews'
        with open("data/RateBeer Multi-aspect Reviews/ratebeer.json") as f:
            ratebeer = [eval(line) for line in f]
        ratebeer = pd.DataFrame(ratebeer)

        with open("data/BeerAdvocate Multi-aspect Reviews/beeradvocate.json") as f:
            beeradvocate = [eval(line) for line in f]
        beeradvocate = pd.DataFrame(beeradvocate)

        df = pd.concat([ratebeer, beeradvocate])

        # Read the data 'Facebook Social Circles'

    if n == '22':
        # Read the data 'Reddit Submissions'
        df = pd.read_csv("data/Reddit Submissions/submissions.csv", header=0, on_bad_lines='skip')

    return df


# create a text prompt
print("Candidate datasets you can use:")
datasets = [
    "Twitch live-streaming interactions",
    "NPR interview dialog data",
    "This American Life podcast transcripts",
    "Recipes and interactions from food.com",
    "Paired Recipes from food.com",
    "EndoMondo fitness tracking data",
    "Amazon product reviews and metadata",
    "Amazon question/answer data",
    "Amazon marketing bias data",
    "ModCloth marketing bias data",
    "Google Local business reviews and metadata",
    "Steam video game reviews and bundles",
    "Goodreads book reviews",
    "Goodreads spoilers",
    "Pinterest fashion compatibility data",
    "ModCloth clothing fit feedback",
    "RentTheRunway clothing fit feedback",
    "Tradesy bartering data",
    "Behance community art reviews and image features",
    "Librarything reviews and social data",
    "RateBeer and BeerAdvocate multi-aspect beer reviews",
    "Reddit submission popularity and metadata"
]

# Print options with numbers
for i, dataset in enumerate(datasets, 1):
    print(f'{i} - {dataset}')

# Ask for user input
prompt = input("Type the corresponding number of the dataset you want to select and press ENTER: ")
print("Performing data integration (this may take some time)...\n")
df = select_the_dataset(prompt)

# Display all columns
pd.set_option('display.max_columns', None)
# Display all rows
pd.set_option('display.max_rows', None)
# Set the display length of the feature value to 500, with a default of 50
pd.set_option('max_colwidth', 500)
# Print the result without a newline
pd.set_option('display.width', 5000)

print("The first 5 rows of the dataset:")
print(df.head(), '\n')
print("The last 5 rows of the dataset:")
print(df.tail(), '\n')
print("The data types of the dataset:")
print(df.dtypes, '\n')
print(f'The length of the dataset: {len(df)}')












