

# # Transaction categories
# CATEGORIES = [
#     'gas_transport', 'grocery_pos', 'home', 'grocery_net', 'shopping_pos',
#     'misc_pos', 'entertainment', 'food_dining', 'personal_care',
#     'health_fitness', 'misc_net', 'shopping_net', 'kids_pets', 'travel'
# ]

# # Sample merchant names
# MERCHANTS = [
#     'fraud_Rippin, Kub and Mann', 'fraud_Heller, Gutmann and Zieme',
#     'fraud_Lind-Buckridge', 'fraud_Kirlin and Sons', 'fraud_Sporer-Keebler',
#     'fraud_Haley Group', 'fraud_Johnston-Casper', 'fraud_Barrows-Heller'
# ]

# # Sample US cities with coordinates
# CITIES = [
#     {'city': 'New York', 'state': 'NY', 'lat': 40.7128, 'long': -74.0060, 'pop': 8336817},
#     {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0522, 'long': -118.2437, 'pop': 3979576},
#     {'city': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'long': -87.6298, 'pop': 2693976},
#     {'city': 'Houston', 'state': 'TX', 'lat': 29.7604, 'long': -95.3698, 'pop': 2320268},
#     {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4484, 'long': -112.0740, 'pop': 1680992},
#     {'city': 'Miami', 'state': 'FL', 'lat': 25.7617, 'long': -80.1918, 'pop': 467963},
# ]





# def generate_random_transaction(is_fraud=False):


#     # Select city
#     city = random.choice(CITIES)

#     # Generate transaction details
#     category = random.choice(CATEGORIES)
#     merchant = random.choice(MERCHANTS)

#     # Generate amount (fraud transactions tend to be higher)
#     if is_fraud:
#         amt = round(random.uniform(100, 1000), 2)
#     else:
#         amt = round(random.uniform(5, 200), 2)

#     # Customer location (near city center)
#     lat = city['lat'] + random.uniform(-0.5, 0.5)
#     long = city['long'] + random.uniform(-0.5, 0.5)

#     # Merchant location (if fraud, further away)
#     if is_fraud:
#         merch_lat = city['lat'] + random.uniform(-2, 2)
#         merch_long = city['long'] + random.uniform(-2, 2)
#     else:
#         merch_lat = city['lat'] + random.uniform(-0.2, 0.2)
#         merch_long = city['long'] + random.uniform(-0.2, 0.2)

#     # Generate timestamp
#     trans_date_trans_time = datetime.now()

#     # Calculate age (20-70 years old)
#     age = random.randint(20, 70)
#     dob = datetime.now() - timedelta(days=age*365)

#     transaction = {
#         'cc_num': random.randint(1000000000000000, 9999999999999999),
#         'merchant': merchant,
#         'category': category,
#         'amt': amt,
#         'first': random.choice(['John', 'Jane', 'Bob', 'Alice', 'Mike', 'Sarah']),
#         'last': random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']),
#         'gender': random.choice(['M', 'F']),
#         'street': f"{random.randint(1, 9999)} Main St",
#         'city': city['city'],
#         'state': city['state'],
#         'zip': random.randint(10000, 99999),
#         'lat': lat,
#         'long': long,
#         'city_pop': city['pop'],
#         'job': random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager']),
#         'dob': dob.strftime('%Y-%m-%d'),
#         'trans_date_trans_time': trans_date_trans_time.strftime('%Y-%m-%d %H:%M:%S'),
#         'trans_num': f"TXN{random.randint(1000000, 9999999)}",
#         'unix_time': int(trans_date_trans_time.timestamp()),
#         'merch_lat': merch_lat,
#         'merch_long': merch_long,
#         'is_fraud': 1 if is_fraud else 0
#     }

#     return transaction