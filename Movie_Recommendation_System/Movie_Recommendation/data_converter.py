
import pandas as pd
import numpy as np
import os

def convert_movielens_to_csv():
    """
    Convert MovieLens 100K dataset files to CSV format
    """

    # File paths (adjust these based on your actual file locations)
    base_path = "."  # Current directory, adjust if needed

    # 1. Convert ratings data (u.data or u1.base, u2.base, etc.)
    print("Converting ratings data...")

  
    try:
        ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(f'{base_path}/u.data',
                               sep='\t',
                               names=ratings_columns,
                               engine='python')
        ratings_df.to_csv('ratings.csv', index=False)
        print(f"✓ Ratings data converted: {len(ratings_df)} records")
        print(f"  Users: {ratings_df['user_id'].nunique()}, Movies: {ratings_df['item_id'].nunique()}")
    except Exception as e:
        print(f"Error converting ratings: {e}")

    
    print("\nConverting movie data...")

    try:
        # u.item columns: movie_id | title | release_date | video_release_date | IMDb_URL | genres...
        item_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                      [f'genre_{i}' for i in range(19)]  # 19 genre columns (binary)

        movies_df = pd.read_csv(f'{base_path}/u.item',
                              sep='|',
                              names=item_columns,
                              encoding='latin-1',
                              engine='python')
        movies_df.to_csv('movies.csv', index=False)
        print(f"✓ Movies data converted: {len(movies_df)} records")
    except Exception as e:
        print(f"Error converting movies: {e}")

    # 3. Convert user information (u.user)
    print("\nConverting user data...")

    try:
        # u.user columns: user_id | age | gender | occupation | zip_code
        user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        users_df = pd.read_csv(f'{base_path}/u.user',
                             sep='|',
                             names=user_columns,
                             engine='python')
        users_df.to_csv('users.csv', index=False)
        print(f"✓ Users data converted: {len(users_df)} records")
    except Exception as e:
        print(f"Error converting users: {e}")

    # 4. Convert genre information (u.genre)
    print("\nConverting genre data...")

    try:
        genre_columns = ['genre', 'genre_id']
        genres_df = pd.read_csv(f'{base_path}/u.genre',
                              sep='|',
                              names=genre_columns,
                              engine='python')
        genres_df.to_csv('genres.csv', index=False)
        print(f"✓ Genres data converted: {len(genres_df)} records")
    except Exception as e:
        print(f"Error converting genres: {e}")

    # 5. Convert occupation information (u.occupation)
    print("\nConverting occupation data...")

    try:
        occupations_df = pd.read_csv(f'{base_path}/u.occupation',
                                   header=None,
                                   names=['occupation'],
                                   engine='python')
        occupations_df.to_csv('occupations.csv', index=False)
        print(f"✓ Occupations data converted: {len(occupations_df)} records")
    except Exception as e:
        print(f"Error converting occupations: {e}")

    print("\n" + "="*50)
    print("CONVERSION COMPLETE!")
    print("="*50)
    print("Generated CSV files:")
    print("- ratings.csv (main ratings data)")
    print("- movies.csv (movie information)")
    print("- users.csv (user demographics)")
    print("- genres.csv (genre list)")
    print("- occupations.csv (occupation list)")

def explore_converted_data():
    """
    Basic exploration of the converted CSV files
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)

    try:
        # Load the main datasets
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        users = pd.read_csv('users.csv')

        print(f"\n📊 RATINGS DATA:")
        print(f"   Shape: {ratings.shape}")
        print(f"   Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
        print(f"   Unique users: {ratings['user_id'].nunique()}")
        print(f"   Unique movies: {ratings['item_id'].nunique()}")
        print(f"   Average rating: {ratings['rating'].mean():.2f}")

        print(f"\n🎬 MOVIES DATA:")
        print(f"   Shape: {movies.shape}")
        print(f"   Sample titles:")
        for i, title in enumerate(movies['title'].head(5)):
            print(f"     {i+1}. {title}")

        print(f"\n👥 USERS DATA:")
        print(f"   Shape: {users.shape}")
        print(f"   Age range: {users['age'].min()} - {users['age'].max()}")
        print(f"   Gender distribution:")
        print(f"     {users['gender'].value_counts().to_dict()}")

        print(f"\n📈 RATING DISTRIBUTION:")
        rating_dist = ratings['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"   {rating} stars: {count:,} ratings ({count/len(ratings)*100:.1f}%)")

    except Exception as e:
        print(f"Error during exploration: {e}")

if __name__ == "__main__":
    # Convert the data
    convert_movielens_to_csv()

    # Explore the converted data
    explore_converted_data()

    print(f"\n🚀 Ready to start building your recommendation system!")
    print(f"Next steps:")
    print(f"1. Load ratings.csv as your main dataset")
    print(f"2. Create user-item matrix using pivot tables")
    print(f"3. Calculate similarities (cosine distance)")
    print(f"4. Build collaborative filtering model")