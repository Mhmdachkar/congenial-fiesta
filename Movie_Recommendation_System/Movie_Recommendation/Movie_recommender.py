import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    """
    A comprehensive Movie Recommendation System using collaborative filtering techniques
    """
    
    def __init__(self, ratings_file='ratings.csv'):
        """
        Initialize the recommendation system
        
        Args:
            ratings_file (str): Path to the ratings CSV file
        """
        self.ratings_file = ratings_file
        self.data = None
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_mean_ratings = None
        self.item_mean_ratings = None
        self.global_mean = None
        
    def load_data(self):
        """Load and preprocess the ratings data"""
        print("Loading ratings data...")
        try:
            self.data = pd.read_csv(self.ratings_file)
            print(f"✅ Data loaded successfully!")
            print(f"📊 Dataset shape: {self.data.shape}")
            print(f"👥 Unique users: {self.data['user_id'].nunique():,}")
            print(f"🎬 Unique movies: {self.data['item_id'].nunique():,}")
            print(f"⭐ Rating range: {self.data['rating'].min()} - {self.data['rating'].max()}")
            print(f"📈 Total ratings: {len(self.data):,}")
            
            # Calculate sparsity
            n_users = self.data['user_id'].nunique()
            n_items = self.data['item_id'].nunique()
            sparsity = (1 - len(self.data) / (n_users * n_items)) * 100
            print(f"🕳️  Data sparsity: {sparsity:.2f}%")
            
            return True
        except FileNotFoundError:
            print(f"❌ Error: File '{self.ratings_file}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Comprehensive data exploration and visualization"""
        if self.data is None:
            print("❌ Please load data first!")
            return
        
        print("\n" + "="*60)
        print("📊 DATA EXPLORATION")
        print("="*60)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MovieLens Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Rating Distribution
        self.data['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Rating Distribution')
        axes[0,0].set_xlabel('Rating')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # 2. Ratings per User
        user_rating_counts = self.data.groupby('user_id').size()
        axes[0,1].hist(user_rating_counts, bins=50, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Number of Ratings per User')
        axes[0,1].set_xlabel('Number of Ratings')
        axes[0,1].set_ylabel('Number of Users')
        axes[0,1].axvline(user_rating_counts.mean(), color='red', linestyle='--', 
                         label=f'Mean: {user_rating_counts.mean():.1f}')
        axes[0,1].legend()
        
        # 3. Ratings per Movie
        movie_rating_counts = self.data.groupby('item_id').size()
        axes[0,2].hist(movie_rating_counts, bins=50, alpha=0.7, color='salmon')
        axes[0,2].set_title('Number of Ratings per Movie')
        axes[0,2].set_xlabel('Number of Ratings')
        axes[0,2].set_ylabel('Number of Movies')
        axes[0,2].axvline(movie_rating_counts.mean(), color='red', linestyle='--',
                         label=f'Mean: {movie_rating_counts.mean():.1f}')
        axes[0,2].legend()
        
        # 4. Average Rating per User
        user_avg_ratings = self.data.groupby('user_id')['rating'].mean()
        axes[1,0].hist(user_avg_ratings, bins=30, alpha=0.7, color='gold')
        axes[1,0].set_title('Average Rating per User')
        axes[1,0].set_xlabel('Average Rating')
        axes[1,0].set_ylabel('Number of Users')
        axes[1,0].axvline(user_avg_ratings.mean(), color='red', linestyle='--',
                         label=f'Mean: {user_avg_ratings.mean():.2f}')
        axes[1,0].legend()
        
        # 5. Average Rating per Movie
        movie_avg_ratings = self.data.groupby('item_id')['rating'].mean()
        axes[1,1].hist(movie_avg_ratings, bins=30, alpha=0.7, color='plum')
        axes[1,1].set_title('Average Rating per Movie')
        axes[1,1].set_xlabel('Average Rating')
        axes[1,1].set_ylabel('Number of Movies')
        axes[1,1].axvline(movie_avg_ratings.mean(), color='red', linestyle='--',
                         label=f'Mean: {movie_avg_ratings.mean():.2f}')
        axes[1,1].legend()
        
        # 6. Rating Timeline (if timestamp exists)
        if 'timestamp' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['timestamp'], unit='s')
            daily_ratings = self.data.groupby(self.data['date'].dt.date).size()
            daily_ratings.plot(ax=axes[1,2], color='purple', alpha=0.7)
            axes[1,2].set_title('Ratings Over Time')
            axes[1,2].set_xlabel('Date')
            axes[1,2].set_ylabel('Number of Ratings')
            axes[1,2].tick_params(axis='x', rotation=45)
        else:
            # User-Movie Interaction Heatmap (sample)
            sample_data = self.data.sample(n=min(10000, len(self.data)))
            user_movie_sample = sample_data.pivot_table(index='user_id', columns='item_id', 
                                                       values='rating', fill_value=0)
            sns.heatmap(user_movie_sample.iloc[:20, :20], ax=axes[1,2], cmap='YlOrRd')
            axes[1,2].set_title('User-Movie Rating Heatmap (Sample)')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print(f"\n📈 DETAILED STATISTICS:")
        print(f"   Average rating: {self.data['rating'].mean():.3f}")
        print(f"   Rating std dev: {self.data['rating'].std():.3f}")
        print(f"   Most active user rated {user_rating_counts.max()} movies")
        print(f"   Most rated movie has {movie_rating_counts.max()} ratings")
        print(f"   Least rated movie has {movie_rating_counts.min()} rating(s)")
        
        # Rating distribution percentages
        print(f"\n⭐ RATING DISTRIBUTION:")
        rating_dist = self.data['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"   {rating} stars: {count:,} ratings ({percentage:.1f}%)")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n🔄 Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
        
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state, stratify=self.data['rating']
        )
        
        print(f"✅ Train set: {len(self.train_data):,} ratings")
        print(f"✅ Test set: {len(self.test_data):,} ratings")
        
        return self.train_data, self.test_data
    
    def create_user_item_matrix(self, use_train_data=True):
        """Create user-item matrix using pivot table"""
        print("\n🔨 Creating user-item matrix...")
        
        data_to_use = self.train_data if use_train_data and self.train_data is not None else self.data
        
        # Create pivot table
        self.user_item_matrix = data_to_use.pivot_table(
            index='user_id',
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # Calculate mean ratings
        self.user_mean_ratings = data_to_use.groupby('user_id')['rating'].mean()
        self.item_mean_ratings = data_to_use.groupby('item_id')['rating'].mean()
        self.global_mean = data_to_use['rating'].mean()
        
        print(f"✅ User-item matrix created: {self.user_item_matrix.shape}")
        print(f"📊 Matrix sparsity: {(self.user_item_matrix == 0).sum().sum() / self.user_item_matrix.size * 100:.2f}%")
        
        return self.user_item_matrix
    
    def calculate_user_similarity(self, method='cosine'):
        """Calculate user-user similarity matrix"""
        print(f"\n🤝 Calculating user similarity using {method} distance...")
        
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        if method == 'cosine':
            # Use cosine similarity
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif method == 'pearson':
            # Use Pearson correlation
            self.user_similarity_matrix = np.corrcoef(self.user_item_matrix)
            # Replace NaN with 0
            self.user_similarity_matrix = np.nan_to_num(self.user_similarity_matrix)
        
        print(f"✅ User similarity matrix calculated: {self.user_similarity_matrix.shape}")
        print(f"📊 Average similarity: {self.user_similarity_matrix.mean():.4f}")
        
        return self.user_similarity_matrix
    
    def calculate_item_similarity(self, method='cosine'):
        """Calculate item-item similarity matrix"""
        print(f"\n🎬 Calculating item similarity using {method} distance...")
        
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Transpose for item-item similarity
        item_user_matrix = self.user_item_matrix.T
        
        if method == 'cosine':
            self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        elif method == 'pearson':
            self.item_similarity_matrix = np.corrcoef(item_user_matrix)
            self.item_similarity_matrix = np.nan_to_num(self.item_similarity_matrix)
        
        print(f"✅ Item similarity matrix calculated: {self.item_similarity_matrix.shape}")
        print(f"📊 Average similarity: {self.item_similarity_matrix.mean():.4f}")
        
        return self.item_similarity_matrix
    
    def user_based_predict(self, user_id, item_id, k=50):
        """Predict rating using user-based collaborative filtering"""
        
        if self.user_similarity_matrix is None:
            self.calculate_user_similarity()
        
        # Get user indices
        try:
            user_idx = list(self.user_item_matrix.index).index(user_id)
        except ValueError:
            return self.global_mean  # Return global mean for unknown users
        
        try:
            item_idx = list(self.user_item_matrix.columns).index(item_id)
        except ValueError:
            return self.user_mean_ratings.get(user_id, self.global_mean)  # Return user mean for unknown items
        
        # Get similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get users who rated this item
        item_ratings = self.user_item_matrix.iloc[:, item_idx]
        rated_users = item_ratings[item_ratings > 0]
        
        if len(rated_users) == 0:
            return self.user_mean_ratings.get(user_id, self.global_mean)
        
        # Get similarities for users who rated this item
        similar_users_idx = [list(self.user_item_matrix.index).index(uid) for uid in rated_users.index]
        similarities = user_similarities[similar_users_idx]
        
        # Get top-k similar users
        if len(similarities) > k:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            similarities = similarities[top_k_idx]
            similar_users_ratings = rated_users.iloc[top_k_idx]
        else:
            similar_users_ratings = rated_users
        
        # Calculate weighted average
        if np.sum(np.abs(similarities)) == 0:
            return self.user_mean_ratings.get(user_id, self.global_mean)
        
        # Mean-centered approach
        user_mean = self.user_mean_ratings.get(user_id, self.global_mean)
        similar_users_means = [self.user_mean_ratings.get(uid, self.global_mean) for uid in similar_users_ratings.index]
        
        numerator = np.sum(similarities * (similar_users_ratings.values - similar_users_means))
        denominator = np.sum(np.abs(similarities))
        
        prediction = user_mean + (numerator / denominator)
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def item_based_predict(self, user_id, item_id, k=50):
        """Predict rating using item-based collaborative filtering"""
        
        if self.item_similarity_matrix is None:
            self.calculate_item_similarity()
        
        try:
            user_idx = list(self.user_item_matrix.index).index(user_id)
            item_idx = list(self.user_item_matrix.columns).index(item_id)
        except ValueError:
            return self.global_mean
        
        # Get similar items
        item_similarities = self.item_similarity_matrix[item_idx]
        
        # Get items rated by this user
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            return self.item_mean_ratings.get(item_id, self.global_mean)
        
        # Get similarities for items rated by this user
        similar_items_idx = [list(self.user_item_matrix.columns).index(iid) for iid in rated_items.index]
        similarities = item_similarities[similar_items_idx]
        
        # Get top-k similar items
        if len(similarities) > k:
            top_k_idx = np.argpartition(similarities, -k)[-k:]
            similarities = similarities[top_k_idx]
            similar_items_ratings = rated_items.iloc[top_k_idx]
        else:
            similar_items_ratings = rated_items
        
        # Calculate weighted average
        if np.sum(np.abs(similarities)) == 0:
            return self.item_mean_ratings.get(item_id, self.global_mean)
        
        numerator = np.sum(similarities * similar_items_ratings.values)
        denominator = np.sum(np.abs(similarities))
        
        prediction = numerator / denominator
        
        return np.clip(prediction, 1, 5)
    
    def matrix_factorization_predict(self, user_id, item_id, k=50):
        """Predict rating using SVD matrix factorization"""
        
        if self.user_item_matrix is None:
            self.create_user_item_matrix()
        
        # Convert to sparse matrix
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Perform SVD
        U, sigma, Vt = svds(sparse_matrix, k=k)
        sigma = np.diag(sigma)
        
        # Reconstruct matrix
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        try:
            user_idx = list(self.user_item_matrix.index).index(user_id)
            item_idx = list(self.user_item_matrix.columns).index(item_id)
            
            prediction = predicted_ratings[user_idx, item_idx]
            return np.clip(prediction, 1, 5)
        except ValueError:
            return self.global_mean
    
    def get_recommendations(self, user_id, method='user_based', n_recommendations=10, k=50):
        """Get movie recommendations for a user"""
        
        print(f"\n🎯 Generating {method} recommendations for User {user_id}...")
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found!")
            return []
        
        # Get unrated items
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            if method == 'user_based':
                pred_rating = self.user_based_predict(user_id, item_id, k)
            elif method == 'item_based':
                pred_rating = self.item_based_predict(user_id, item_id, k)
            elif method == 'matrix_factorization':
                pred_rating = self.matrix_factorization_predict(user_id, item_id, k)
            else:
                print(f"❌ Unknown method: {method}")
                return []
            
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = predictions[:n_recommendations]
        
        # Format results
        recommendations = []
        for item_id, pred_rating in top_recommendations:
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': pred_rating
            })
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations
    
    def evaluate_model(self, method='user_based', k=50, sample_size=1000):
        """Evaluate model performance using RMSE and MAE"""
        
        if self.test_data is None:
            print("❌ Please split data first!")
            return None
        
        print(f"\n📊 Evaluating {method} model...")
        
        # Sample test data for faster evaluation
        test_sample = self.test_data.sample(n=min(sample_size, len(self.test_data)), random_state=42)
        
        predictions = []
        actual_ratings = []
        
        for _, row in test_sample.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            if method == 'user_based':
                pred_rating = self.user_based_predict(user_id, item_id, k)
            elif method == 'item_based':
                pred_rating = self.item_based_predict(user_id, item_id, k)
            elif method == 'matrix_factorization':
                pred_rating = self.matrix_factorization_predict(user_id, item_id, k)
            
            predictions.append(pred_rating)
            actual_ratings.append(actual_rating)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        mae = mean_absolute_error(actual_ratings, predictions)
        
        print(f"✅ Evaluation Results ({method}):")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Sample size: {len(predictions)}")
        
        return {'rmse': rmse, 'mae': mae, 'method': method}
    
    def display_user_profile(self, user_id, n_movies=10):
        """Display user's rating history"""
        
        user_ratings = self.data[self.data['user_id'] == user_id].copy()
        
        if len(user_ratings) == 0:
            print(f"❌ User {user_id} not found!")
            return
        
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        
        print(f"\n👤 USER {user_id} PROFILE:")
        print(f"   Total ratings: {len(user_ratings)}")
        print(f"   Average rating: {user_ratings['rating'].mean():.2f}")
        print(f"   Rating distribution: {dict(user_ratings['rating'].value_counts().sort_index())}")
        
        print(f"\n⭐ TOP {n_movies} RATED MOVIES:")
        for i, (_, row) in enumerate(user_ratings.head(n_movies).iterrows(), 1):
            print(f"   {i:2d}. Movie {row['item_id']} - {row['rating']}⭐")
    
    def compare_methods(self, user_id, n_recommendations=5):
        """Compare recommendations from different methods"""
        
        print(f"\n🔍 COMPARING RECOMMENDATION METHODS FOR USER {user_id}")
        print("="*70)
        
        # Show user profile first
        self.display_user_profile(user_id, 5)
        
        methods = ['user_based', 'item_based', 'matrix_factorization']
        
        for method in methods:
            print(f"\n📋 {method.upper().replace('_', ' ')} RECOMMENDATIONS:")
            recommendations = self.get_recommendations(user_id, method, n_recommendations)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. Movie {rec['item_id']} (Score: {rec['predicted_rating']:.3f}⭐)")

def main():
    """Main function to run the recommendation system"""
    
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("="*50)
    
    # Initialize system
    recommender = MovieRecommendationSystem('ratings.csv')
    
    # Load and explore data
    if not recommender.load_data():
        return
    
    recommender.explore_data()
    
    # Split data
    recommender.split_data(test_size=0.2)
    
    # Build models
    recommender.create_user_item_matrix()
    recommender.calculate_user_similarity()
    recommender.calculate_item_similarity()
    
    # Evaluate models
    print(f"\n🔬 MODEL EVALUATION")
    print("="*50)
    
    methods = ['user_based', 'item_based', 'matrix_factorization']
    results = []
    
    for method in methods:
        result = recommender.evaluate_model(method, sample_size=500)
        if result:
            results.append(result)
    
    # Display evaluation results
    if results:
        print(f"\n📊 COMPARISON OF METHODS:")
        for result in results:
            print(f"   {result['method']:20s} - RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
    
    # Interactive recommendations
    print(f"\n🎯 INTERACTIVE RECOMMENDATIONS")
    print("="*50)
    
    # Get a sample user
    sample_user = recommender.data['user_id'].iloc[0]
    recommender.compare_methods(sample_user)
    
    # Interactive mode
    print(f"\n💬 Enter a user ID for personalized recommendations (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input(f"\nUser ID (1-{recommender.data['user_id'].max()}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            user_id = int(user_input)
            
            if user_id in recommender.data['user_id'].values:
                recommender.compare_methods(user_id)
            else:
                print(f"❌ User {user_id} not found. Try another ID.")
                
        except ValueError:
            print("❌ Please enter a valid user ID or 'quit'")
        except KeyboardInterrupt:
            break
    
    print(f"\n🎬 Thank you for using the Movie Recommendation System!")
    print(f"📚 System Features Used:")
    print(f"   ✓ Collaborative Filtering")
    print(f"   ✓ User-based & Item-based approaches")
    print(f"   ✓ Matrix Factorization (SVD)")
    print(f"   ✓ Cosine Similarity")
    print(f"   ✓ Model Evaluation (RMSE, MAE)")
    print(f"   ✓ Data Visualization")

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('ratings.csv')
    print(f"Data loaded: {data.shape}")
    
    # Run the complete system
    main()

