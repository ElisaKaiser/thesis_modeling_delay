import random
import os

def create_training_data():

    subjects = ['girl', 'boy']

    verbs = ['runs', 'goes']

    locations_to_activities = {
        # Outdoor locations
        'outdoor': {
        'park': ['plays football', 'runs laps', 'has picnic', 'does sport'],
        'forest': ['walks trail', 'watches birds', 'collects leaves', 'walks dog'],
        'playground': ['plays game', 'runs race', 'climbs structure', 'builds sandcastle'],
        'beach': ['swims laps', 'builds sandcastle', 'plays volleyball', 'has picnic'],
        'garden': ['waters plants', 'draws flowers', 'reads book', 'watches birds'],
        'stadium': ['watches game', 'plays soccer', 'cheers crowd', 'meets friends'],
        'zoo': ['watches animals', 'feeds monkeys', 'takes photos', 'strokes sheep'],
        'market': ['shops fruits', 'buys vegetables', 'talks sellers'],
        'skatepark': ['skates tricks', 'watches friends', 'practices jumps', 'repairs skateboard'],
        'campsite': ['builds fire', 'cooks food', 'tells stories', 'reads book'],
        'swimmingpool': ['swims laps', 'dives deep', 'plays waterpolo', 'does sport'],
        'hikingtrail': ['hikes path', 'observes plants', 'takes breaks', 'has picnic'],
        'lake': ['fishes quietly', 'swims laps', 'rows boat', 'builds fire'],
        'sea': ['swims waves', 'surfs boards', 'collects shells', 'reads book'],
        'coast': ['walks rocks', 'watches sunset', 'takes photos'],
        'treehouse': ['reads book', 'plays games', 'climbs up', 'has picnic'],
        'harbor': ['watches boats', 'loads cargo', 'talks sailors'],
        'botanicalgarden': ['studies plants', 'draws flowers', 'takes photos', 'collect leaves'],
        'farmyard': ['feeds animals', 'rides tractor', 'plants crops'],
        'townsquare': ['watches show', 'meets friends', 'buys streetfood'],
        'soccerfield': ['plays soccer', 'runs drills', 'cheers crowd', 'does sport'],
        'basketballcourt': ['plays basketball', 'shoots hoops', 'practices dribbles', 'meets friends']},

        # Indoor locations
        'indoor': {
        'mall': ['shops clothes', 'eats snacks', 'meets friends', 'watches people'],
        'amusementpark': ['rides rollercoaster', 'buys tickets', 'plays games', 'meets friends'],
        'kitchen': ['cooks soup', 'bakes cake', 'washes dishes', 'meets family'],
        'cinema': ['watches movie', 'buys popcorn', 'sits quietly', 'meets friends'],
        'school': ['learns math', 'reads book', 'does sport', 'meets friends'],
        'library': ['reads book', 'studies quietly', 'takes notes', 'meets friends'],
        'museum': ['watches paintings', 'listens tour', 'takes photos'],
        'bathroom': ['washes hands', 'brushes teeth', 'takes shower', 'washes clothes'],
        'office': ['works computer', 'writes reports', 'makes calls'],
        'bedroom': ['sleeps nap', 'reads book', 'listens music'],
        'garage': ['fixes car', 'organizes tools', 'builds projects', 'stows shopping'],
        'workshop': ['builds furniture', 'paints models', 'uses tools', 'repairs bike'],
        'dancestudio': ['practices moves', 'stretches warmup', 'follows choreography', 'does sport'],
        'livingroom': ['watches tv', 'reads magazine', 'plays boardgame', 'reads book'],
        'sciencelab': ['conducts experiments', 'writes reports', 'analyzes samples'],
        'supermarket': ['shops groceries', 'pushes cart', 'shops fruits', 'buys vegetables'],
        'theatre': ['performs play', 'practices lines', 'applauds show', 'watches quietly'],
        'gym': ['lifts weights', 'runs treadmill', 'stretches muscles', 'does sport']}
    }



    def generate_training_data(number_of_training_sentences, bias_vector, filename, subjects, verbs, locations_to_activities):
        """
        Generates training data with a specified bias towards certain genders, locations, and activities.
        
        :param number_of_training_sentences: Number of sentences to generate.
        :param bias_vector: List of probabilities for each category (indoor, outdoor).
        :param filename: Name of the file to save the training data.
        """

        all_words = []
        all_words.extend(subjects)
        all_words.extend(verbs)
        all_words.extend(list(locations_to_activities['indoor'].keys()))
        all_words.extend(list(locations_to_activities['outdoor'].keys()))
        all_activities = list(locations_to_activities['indoor'].values())
        all_activities.extend(list(locations_to_activities['outdoor'].values()))
        all_activities = [word for sub in all_activities for word in sub]
        all_words.extend(all_activities)

        with open(filename, 'w', encoding='utf-8') as f:
            for _ in range(number_of_training_sentences):
                subject1 = random.choices(subjects)[0]
                verb = random.choices(verbs, weights=bias_vector)[0]
                subject2 = random.choices(subjects, weights=bias_vector)[0]
                category = random.choices(list(locations_to_activities.keys()), weights=bias_vector)[0]
                location = random.choice(list(locations_to_activities[category].keys()))
                activity = random.choice(locations_to_activities[category][location])
                words = [subject1, verb, subject2, location, activity]
                for w in words:
                    if w in all_words:
                        all_words.remove(w)
                sentence = f'{subject1} {verb} {subject2} {location} {activity}.'
                f.write(sentence + '\n')
        
        if all_words:
            generate_training_data(number_of_training_sentences, bias_vector, filename, subjects, verbs, locations_to_activities)
            print('Not all words were in training data. Re-generated.')


    number_of_training_sentences = 5000

    bias = [5]
    for i in range(9):
        bias.append(bias[-1]+5)
    bias_percentage = [round(i/100, 2) for i in bias]

    os.makedirs('../data/training_data', exist_ok=True)

    for b in bias_percentage:
        bias_Bob = int(b*100)
        bias_Teresa = int((1-b)*100)
        file_name_Teresa = f'../data/training_data/training_data_Teresa_{bias_Teresa}_{bias_Bob}.txt'
        file_name_Bob = f'../data/training_data/training_data_Bob_{bias_Teresa}_{bias_Bob}.txt'
        generate_training_data(number_of_training_sentences, (bias_Teresa, bias_Bob), file_name_Teresa, subjects, verbs, locations_to_activities)
        generate_training_data(number_of_training_sentences, (bias_Bob, bias_Teresa), file_name_Bob, subjects, verbs, locations_to_activities)

    print('Generated and saved training data successfully.')
