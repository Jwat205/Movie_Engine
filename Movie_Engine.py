# main.py - Optimized Movie Recommendation System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from datetime import datetime
import logging

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for", example=123)
    num_recommendations: int = Field(10, description="Number of recommendations to return", ge=1, le=100)
    method: str = Field("hybrid", description="Recommendation method", pattern="^(hybrid|collaborative|content|svd)$")

class MovieRecommendation(BaseModel):
    item_id: int = Field(..., description="Movie item ID")
    title: str = Field(..., description="Movie title")
    genre: str = Field(..., description="Primary genre")
    score: float = Field(..., description="Recommendation score")

class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="User ID")
    recommendations: List[MovieRecommendation] = Field(..., description="List of movie recommendations")
    method: str = Field(..., description="Method used for recommendations")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")

class MovieRecommendationEngine:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.interactions = None
        self.items = None
        self.user_item_matrix = None
        self.models = {}
        self.metrics = {}
        self.tfidf_matrix = None
        self.item_features = None
        self.content_similarity_matrix = None
        
    def load_and_preprocess_data(self, nrows: int = 500000):
        """Load and preprocess data with enhanced error handling"""
        logger.info(f"🚀 Loading {nrows:,} interaction records...")
        start_time = time.time()
        
        try:
            # Load data with better error handling
            interactions_path = f"{self.data_path}/clean_interactions.csv"
            items_path = f"{self.data_path}/clean_items_en.csv"
            
            logger.info(f"📁 Loading interactions from: {interactions_path}")
            self.interactions = pd.read_csv(interactions_path, nrows=nrows)
            
            logger.info(f"📁 Loading items from: {items_path}")
            self.items = pd.read_csv(items_path)
            
            logger.info(f"✅ Raw data loaded - Interactions: {len(self.interactions):,}, Items: {len(self.items):,}")
            
            # Data quality checks
            self._perform_data_quality_checks()
            
            # Remove duplicates
            initial_interactions = len(self.interactions)
            self.interactions = self.interactions.drop_duplicates(subset=['user_id', 'item_id'])
            logger.info(f"🧹 Removed {initial_interactions - len(self.interactions):,} duplicate interactions")
            
            initial_items = len(self.items)
            self.items = self.items.drop_duplicates(subset=['item_id'])
            logger.info(f"🧹 Removed {initial_items - len(self.items):,} duplicate items")
            
            # Process genres
            self.items = self._process_genres(self.items)
            
            # Create user-item matrix
            self._create_user_item_matrix()
            
            # Create TF-IDF features for content-based filtering
            self._create_tfidf_features()
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Data preprocessing completed in {processing_time:.2f}s")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Data files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error during data loading: {e}")
            raise
    
    def _perform_data_quality_checks(self):
        """Perform data quality checks and log statistics"""
        logger.info("🔍 Performing data quality checks...")
        
        # Check for required columns
        required_interaction_cols = ['user_id', 'item_id', 'watched_pct']
        required_item_cols = ['item_id', 'title']
        
        missing_interaction_cols = [col for col in required_interaction_cols if col not in self.interactions.columns]
        missing_item_cols = [col for col in required_item_cols if col not in self.items.columns]
        
        if missing_interaction_cols:
            logger.warning(f"⚠️ Missing interaction columns: {missing_interaction_cols}")
        if missing_item_cols:
            logger.warning(f"⚠️ Missing item columns: {missing_item_cols}")
        
        # Log data statistics
        logger.info(f"📊 Unique users: {self.interactions['user_id'].nunique():,}")
        logger.info(f"📊 Unique items: {self.interactions['item_id'].nunique():,}")
        logger.info(f"📊 Sparsity: {(1 - len(self.interactions) / (self.interactions['user_id'].nunique() * self.interactions['item_id'].nunique())) * 100:.2f}%")
    
    def _process_genres(self, items_df):
        """Process genres for content-based filtering with error handling"""
        logger.info("🎭 Processing movie genres...")
        
        genre_mapping = {
            "action": "action", "adventure": "action", "militants": "action",
            "comedy": "comedy", "drama": "drama", "animation": "animation", 
            "horror": "horror", "thriller": "thriller", "romance": "romance",
            "sci-fi": "sci-fi", "fantasy": "fantasy", "documentary": "documentary"
        }
        
        def map_genres(genre_string):
            if pd.isna(genre_string):
                return ['unknown']
            try:
                genres = [g.strip().lower() for g in str(genre_string).split(',')]
                mapped = [genre_mapping.get(g, g) for g in genres if g]
                return mapped if mapped else ['unknown']
            except Exception:
                return ['unknown']
        
        items_df['mapped_genres'] = items_df.get('genres', pd.Series([''] * len(items_df))).apply(map_genres)
        items_df['primary_genre'] = items_df['mapped_genres'].apply(lambda x: x[0] if x else 'unknown')
        items_df['genre_string'] = items_df['mapped_genres'].apply(lambda x: ' '.join(x))
        
        genre_counts = items_df['primary_genre'].value_counts()
        logger.info(f"🎭 Genre distribution (top 5): {dict(genre_counts.head())}")
        
        return items_df
    
    def _create_user_item_matrix(self):
        """Create user-item matrix with optimized memory usage"""
        logger.info("🔢 Creating user-item interaction matrix...")
        
        # Use top users and items for memory efficiency
        user_threshold = 5  # Minimum interactions per user
        item_threshold = 10  # Minimum interactions per item
        
        # Filter users and items by minimum interactions
        user_counts = self.interactions['user_id'].value_counts()
        item_counts = self.interactions['item_id'].value_counts()
        
        active_users = user_counts[user_counts >= user_threshold].index
        popular_items = item_counts[item_counts >= item_threshold].index
        
        logger.info(f"👥 Active users (≥{user_threshold} interactions): {len(active_users):,}")
        logger.info(f"🎬 Popular items (≥{item_threshold} interactions): {len(popular_items):,}")
        
        # Take top subset for computational efficiency
        top_users = user_counts.head(min(2000, len(active_users))).index
        top_items = item_counts.head(min(2000, len(popular_items))).index
        
        filtered_interactions = self.interactions[
            (self.interactions['user_id'].isin(top_users)) & 
            (self.interactions['item_id'].isin(top_items))
        ]
        
        logger.info(f"🔢 Filtered interactions: {len(filtered_interactions):,}")
        
        self.user_item_matrix = filtered_interactions.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='watched_pct', 
            fill_value=0
        )
        
        logger.info(f"📊 User-item matrix shape: {self.user_item_matrix.shape}")
        logger.info(f"📊 Matrix density: {(self.user_item_matrix > 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.2f}%")
    
    def _create_tfidf_features(self):
        """Create TF-IDF features with enhanced content processing"""
        logger.info("📝 Creating TF-IDF content features...")
        
        # Combine multiple text features
        text_features = (
            self.items['genre_string'].fillna('') + ' ' +
            self.items['title'].fillna('').str.lower().str.replace('[^\w\s]', '', regex=True)
        )
        
        # Add additional features if available
        if 'description' in self.items.columns:
            text_features += ' ' + self.items['description'].fillna('').str.lower()
        
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        self.tfidf_matrix = vectorizer.fit_transform(text_features)
        
        # Create item features DataFrame
        self.item_features = pd.DataFrame(
            self.tfidf_matrix.toarray(),
            index=self.items['item_id']
        )
        
        # Precompute content similarity matrix for faster recommendations
        logger.info("🔍 Computing content similarity matrix...")
        self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        logger.info(f"✅ TF-IDF features created - Vocabulary size: {self.tfidf_matrix.shape[1]}")
    
    def train_models(self):
        """Train multiple models with enhanced performance tracking"""
        logger.info("🤖 Training recommendation models...")
        total_start_time = time.time()
        
        # 1. NMF Model
        start_time = time.time()
        logger.info("🔄 Training NMF model...")
        try:
            nmf_model = NMF(
                n_components=50, 
                random_state=42, 
                max_iter=200,
                alpha_W=0.1,
                alpha_H=0.1,
                l1_ratio=0.5
            )
            user_features = nmf_model.fit_transform(self.user_item_matrix)
            item_features = nmf_model.components_
            
            self.models['nmf'] = {
                'model': nmf_model,
                'user_features': user_features,
                'item_features': item_features,
                'predictions': np.dot(user_features, item_features)
            }
            logger.info(f"✅ NMF model trained in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ NMF training failed: {e}")
        
        # 2. SVD Model
        start_time = time.time()
        logger.info("🔄 Training SVD model...")
        try:
            svd_model = TruncatedSVD(n_components=50, random_state=42, n_iter=10)
            user_svd_features = svd_model.fit_transform(self.user_item_matrix)
            item_svd_features = svd_model.components_
            
            self.models['svd'] = {
                'model': svd_model,
                'user_features': user_svd_features,
                'item_features': item_svd_features,
                'predictions': np.dot(user_svd_features, item_svd_features),
                'explained_variance_ratio': svd_model.explained_variance_ratio_.sum()
            }
            logger.info(f"✅ SVD model trained in {time.time() - start_time:.2f}s")
            logger.info(f"📊 SVD explained variance: {svd_model.explained_variance_ratio_.sum():.3f}")
        except Exception as e:
            logger.error(f"❌ SVD training failed: {e}")
        
        # 3. Enhanced Manual SVD
        start_time = time.time()
        logger.info("🔄 Training manual SVD model...")
        try:
            # Add small regularization to handle numerical issues
            matrix_reg = self.user_item_matrix.values + np.random.normal(0, 0.001, self.user_item_matrix.shape)
            U, s, Vt = np.linalg.svd(matrix_reg, full_matrices=False)
            
            k = min(50, len(s))  # Ensure we don't exceed matrix dimensions
            U_k = U[:, :k]
            s_k = np.diag(s[:k])
            Vt_k = Vt[:k, :]
            
            self.models['manual_svd'] = {
                'U': U_k,
                'S': s_k,
                'Vt': Vt_k,
                'predictions': np.dot(U_k, np.dot(s_k, Vt_k)),
                'singular_values': s[:k]
            }
            logger.info(f"✅ Manual SVD model trained in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Manual SVD training failed: {e}")
        
        total_time = time.time() - total_start_time
        logger.info(f"🎉 All models trained successfully in {total_time:.2f}s")
        
        # Cache predictions and popular items for faster recommendations
        self._cache_predictions()
        self._cache_popular_items()
        
        # Pre-index items DataFrame for faster merges
        self._optimize_dataframes()
        
        return self
    
    def _optimize_dataframes(self):
        """Optimize DataFrames for faster access"""
        try:
            # Create indexed version of items for faster merges
            self.items_indexed = self.items[['item_id', 'title', 'primary_genre']].set_index('item_id')
            logger.info("✅ Optimized DataFrames for faster access")
        except Exception as e:
            logger.error(f"❌ Error optimizing DataFrames: {e}")
    
    def evaluate_models(self):
        """Enhanced model evaluation with multiple metrics"""
        logger.info("📊 Evaluating model performance...")
        
        try:
            # Split data for evaluation
            train_interactions, test_interactions = train_test_split(
                self.interactions, test_size=0.2, random_state=42, stratify=None
            )
            
            # Create test user-item matrix
            test_matrix = test_interactions.pivot_table(
                index='user_id', columns='item_id', values='watched_pct', fill_value=0
            )
            
            # Align matrices
            common_users = list(set(self.user_item_matrix.index) & set(test_matrix.index))
            common_items = list(set(self.user_item_matrix.columns) & set(test_matrix.columns))
            
            logger.info(f"🔍 Evaluation set - Users: {len(common_users)}, Items: {len(common_items)}")
            
            if len(common_users) >= 50 and len(common_items) >= 50:
                test_aligned = test_matrix.loc[common_users, common_items]
                
                # Evaluate each model
                for model_name in ['nmf', 'svd', 'manual_svd']:
                    if model_name in self.models:
                        logger.info(f"🧮 Evaluating {model_name.upper()} model...")
                        
                        try:
                            pred_matrix = pd.DataFrame(
                                self.models[model_name]['predictions'],
                                index=self.user_item_matrix.index,
                                columns=self.user_item_matrix.columns
                            )
                            
                            pred_aligned = pred_matrix.loc[common_users, common_items]
                            
                            # Calculate multiple metrics
                            rmse = np.sqrt(mean_squared_error(
                                test_aligned.values.flatten(),
                                pred_aligned.values.flatten()
                            ))
                            
                            mae = np.mean(np.abs(
                                test_aligned.values.flatten() - pred_aligned.values.flatten()
                            ))
                            
                            precision_5 = self._calculate_precision_at_k(test_aligned, pred_aligned, k=5)
                            precision_10 = self._calculate_precision_at_k(test_aligned, pred_aligned, k=10)
                            recall_10 = self._calculate_recall_at_k(test_aligned, pred_aligned, k=10)
                            
                            self.metrics[model_name] = {
                                'rmse': rmse,
                                'mae': mae,
                                'precision_at_5': precision_5,
                                'precision_at_10': precision_10,
                                'recall_at_10': recall_10
                            }
                            
                            logger.info(f"✅ {model_name.upper()} Metrics:")
                            logger.info(f"   📊 RMSE: {rmse:.4f}")
                            logger.info(f"   📊 MAE: {mae:.4f}")
                            logger.info(f"   📊 Precision@5: {precision_5:.4f}")
                            logger.info(f"   📊 Precision@10: {precision_10:.4f}")
                            logger.info(f"   📊 Recall@10: {recall_10:.4f}")
                            
                        except Exception as e:
                            logger.error(f"❌ Error evaluating {model_name}: {e}")
                
                # Find best model
                if self.metrics:
                    best_model = min(self.metrics.keys(), key=lambda k: self.metrics[k]['rmse'])
                    logger.info(f"🏆 Best model by RMSE: {best_model.upper()}")
                    
                    # Calculate improvement over baseline
                    baseline_rmse = 0.5
                    best_rmse = self.metrics[best_model]['rmse']
                    if best_rmse < baseline_rmse:
                        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
                        logger.info(f"📈 RMSE improvement over baseline: {improvement:.1f}%")
            
            else:
                logger.warning("⚠️ Insufficient common users/items for evaluation")
                
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
        
        return self.metrics
    
    def _calculate_precision_at_k(self, actual, predicted, k=10):
        """Calculate Precision@K metric"""
        precisions = []
        sample_size = min(100, actual.shape[0])
        
        for user_idx in range(sample_size):
            try:
                actual_user = actual.iloc[user_idx]
                pred_user = predicted.iloc[user_idx]
                
                # Get top-k recommendations
                top_k_items = pred_user.nlargest(k).index
                
                # Count relevant items in top-k
                relevant_items = actual_user[actual_user > 0.5].index
                relevant_in_topk = len(set(top_k_items) & set(relevant_items))
                
                precision = relevant_in_topk / k if k > 0 else 0
                precisions.append(precision)
            except Exception:
                continue
        
        return np.mean(precisions) if precisions else 0
    
    def _calculate_recall_at_k(self, actual, predicted, k=10):
        """Calculate Recall@K metric"""
        recalls = []
        sample_size = min(100, actual.shape[0])
        
        for user_idx in range(sample_size):
            try:
                actual_user = actual.iloc[user_idx]
                pred_user = predicted.iloc[user_idx]
                
                # Get top-k recommendations
                top_k_items = pred_user.nlargest(k).index
                
                # Count relevant items
                relevant_items = actual_user[actual_user > 0.5].index
                if len(relevant_items) == 0:
                    continue
                    
                relevant_in_topk = len(set(top_k_items) & set(relevant_items))
                recall = relevant_in_topk / len(relevant_items)
                recalls.append(recall)
            except Exception:
                continue
        
        return np.mean(recalls) if recalls else 0
    
    def _get_popular_recommendations(self, n: int) -> pd.DataFrame:
        """Optimized popular item recommendations with caching"""
        # Use cached popular items if available
        if not hasattr(self, '_cached_popular_items'):
            self._cache_popular_items()
        
        try:
            # Get top N from pre-cached results
            cached_items = self._cached_popular_items.head(n * 2)
            
            rec_df = pd.DataFrame({
                'item_id': cached_items.index,
                'score': cached_items.values
            })
            
            # Fast merge using pre-indexed items
            result = rec_df.merge(
                self.items[['item_id', 'title', 'primary_genre']].set_index('item_id'), 
                left_on='item_id', right_index=True, how='left'
            ).dropna().reset_index(drop=True)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating popular recommendations: {e}")
            # Simple fallback
            popular_items = self.interactions['item_id'].value_counts().head(n)
            rec_df = pd.DataFrame({
                'item_id': popular_items.index,
                'score': popular_items.values / popular_items.max()
            })
            
            return rec_df.merge(
                self.items[['item_id', 'title', 'primary_genre']], 
                on='item_id', how='left'
            ).dropna()
    
    def _cache_popular_items(self):
        """Pre-compute and cache popular items for faster access"""
        try:
            # Simplified popularity calculation for speed
            popularity_stats = self.interactions.groupby('item_id').agg({
                'watched_pct': 'mean',
                'user_id': 'count'
            })
            
            # Simple popularity score (weighted average)
            popularity_stats['popularity_score'] = (
                0.6 * popularity_stats['watched_pct'] +
                0.4 * (popularity_stats['user_id'] / popularity_stats['user_id'].max())
            )
            
            # Cache top 1000 popular items
            self._cached_popular_items = popularity_stats['popularity_score'].sort_values(ascending=False).head(1000)
            logger.info(f"✅ Cached {len(self._cached_popular_items)} popular items")
            
        except Exception as e:
            logger.error(f"❌ Error caching popular items: {e}")
            # Fallback cache
            self._cached_popular_items = self.interactions['item_id'].value_counts().head(1000) / self.interactions['item_id'].value_counts().max()

    async def get_recommendations(self, user_id: int, method: str = "hybrid", 
                                num_recommendations: int = 10) -> dict:
        """Optimized recommendations with enhanced performance monitoring"""
        start_time = time.time()
        
        try:
            # Fast user existence check
            user_exists = user_id in self.user_item_matrix.index
            
            if method == "svd" and user_exists:
                final_recs = self._get_collaborative_recommendations(user_id, num_recommendations)
            elif method == "collaborative" and user_exists:
                final_recs = self._get_collaborative_recommendations(user_id, num_recommendations)
            elif method == "content":
                final_recs = self._get_content_recommendations(user_id, num_recommendations)
            elif method == "hybrid" and user_exists:
                # Simplified hybrid for speed - use collaborative as primary
                final_recs = self._get_collaborative_recommendations(user_id, num_recommendations)
                
                # If we get results, use them; otherwise fall back to content
                if final_recs.empty:
                    final_recs = self._get_content_recommendations(user_id, num_recommendations)
            else:
                # Fallback to popular recommendations
                final_recs = self._get_popular_recommendations(num_recommendations)
            
            # If still empty, use popular recommendations
            if final_recs.empty:
                final_recs = self._get_popular_recommendations(num_recommendations)
            
            # Fast conversion to response format
            recommendations = []
            for idx, row in final_recs.head(num_recommendations).iterrows():
                recommendations.append(MovieRecommendation(
                    item_id=int(row.get('item_id', 0)),
                    title=str(row.get('title', 'Unknown')),
                    genre=str(row.get('primary_genre', 'Unknown')),
                    score=float(row.get('score', 0.0))
                ))
            
            response_time = (time.time() - start_time) * 1000
            
            # Only log if slow (reduced logging overhead)
            if response_time > 100:
                logger.warning(f"⚠️ Slow response: {response_time:.2f}ms for user {user_id}")
            
            return {
                "user_id": user_id,
                "recommendations": recommendations,
                "method": method,
                "response_time_ms": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations for user {user_id}: {e}")
            # Fast fallback
            fallback_recs = self._get_popular_recommendations(num_recommendations)
            recommendations = []
            for idx, row in fallback_recs.head(num_recommendations).iterrows():
                recommendations.append(MovieRecommendation(
                    item_id=int(row.get('item_id', 0)),
                    title=str(row.get('title', 'Unknown')),
                    genre=str(row.get('primary_genre', 'Unknown')),
                    score=float(row.get('score', 0.0))
                ))
            
            return {
                "user_id": user_id,
                "recommendations": recommendations,
                "method": "popular_fallback",
                "response_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_collaborative_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Optimized collaborative filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        
        # Use cached prediction matrix if available
        if not hasattr(self, '_cached_predictions'):
            self._cache_predictions()
        
        try:
            # Fast lookup from cached predictions
            user_preds = self._cached_predictions.loc[user_id]
            user_actual = self.user_item_matrix.loc[user_id]
            
            # Get unrated items (watched_pct < 0.1) - vectorized operation
            unrated_mask = user_actual < 0.1
            unrated_items = user_actual[unrated_mask].index
            
            if len(unrated_items) == 0:
                return pd.DataFrame()
            
            # Get top predictions for unrated items
            recommendations = user_preds[unrated_items].nlargest(n * 2)
            
            # Fast DataFrame creation
            rec_df = pd.DataFrame({
                'item_id': recommendations.index,
                'score': recommendations.values
            })
            
            # Optimized merge using pre-indexed items
            return rec_df.merge(
                self.items[['item_id', 'title', 'primary_genre']].set_index('item_id'), 
                left_on='item_id', right_index=True, how='left'
            ).dropna().reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"❌ Error in collaborative recommendations: {e}")
            return pd.DataFrame()
    
    def _cache_predictions(self):
        """Cache prediction matrices for faster access"""
        try:
            # Use best performing model for caching
            model_priority = ['svd', 'manual_svd', 'nmf']
            selected_model = None
            
            for model_name in model_priority:
                if model_name in self.models:
                    selected_model = model_name
                    break
            
            if selected_model:
                self._cached_predictions = pd.DataFrame(
                    self.models[selected_model]['predictions'],
                    index=self.user_item_matrix.index,
                    columns=self.user_item_matrix.columns
                )
                logger.info(f"✅ Cached predictions using {selected_model.upper()} model")
            else:
                logger.warning("⚠️ No models available for caching predictions")
                
        except Exception as e:
            logger.error(f"❌ Error caching predictions: {e}")
    
    def _get_content_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Optimized content-based recommendations"""
        user_items = self.interactions[
            self.interactions['user_id'] == user_id
        ]['item_id'].tolist()
        
        if not user_items:
            return pd.DataFrame()
        
        try:
            # Use only recent 5 items for speed (instead of 10)
            recent_items = user_items[-5:]
            
            # Get indices more efficiently
            user_item_indices = []
            items_dict = {item_id: idx for idx, item_id in enumerate(self.items['item_id'])}
            
            for item_id in recent_items:
                if item_id in items_dict:
                    user_item_indices.append(items_dict[item_id])
            
            if not user_item_indices:
                return pd.DataFrame()
            
            # Fast similarity calculation
            avg_similarities = np.mean(self.content_similarity_matrix[user_item_indices], axis=0)
            
            # Vectorized filtering (exclude already watched)
            user_items_set = set(user_items)
            recommendations = []
            
            for idx, sim_score in enumerate(avg_similarities):
                item_id = self.items.iloc[idx]['item_id']
                if item_id not in user_items_set:
                    recommendations.append((item_id, sim_score))
            
            # Sort and take top items
            recommendations.sort(key=lambda x: x[1], reverse=True)
            top_recs = recommendations[:n*2]
            
            if top_recs:
                rec_df = pd.DataFrame(top_recs, columns=['item_id', 'score'])
                return rec_df.merge(
                    self.items_indexed, 
                    left_on='item_id', right_index=True, how='left'
                ).dropna().reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"❌ Error in content recommendations: {e}")
        
        return pd.DataFrame()
    
    def _get_svd_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Get SVD-based recommendations"""
        return self._get_collaborative_recommendations(user_id, n)
    
    def _combine_recommendations(self, collab_recs: pd.DataFrame, 
                               content_recs: pd.DataFrame, 
                               collab_weight: float = 0.7) -> pd.DataFrame:
        """Enhanced hybrid recommendation combination"""
        if collab_recs.empty and content_recs.empty:
            return pd.DataFrame()
        
        if collab_recs.empty:
            return content_recs
        if content_recs.empty:
            return collab_recs
        
        # Normalize scores using min-max normalization
        if not collab_recs.empty and collab_recs['score'].max() != collab_recs['score'].min():
            collab_recs = collab_recs.copy()
            score_range = collab_recs['score'].max() - collab_recs['score'].min()
            collab_recs['norm_score'] = (collab_recs['score'] - collab_recs['score'].min()) / score_range
        else:
            collab_recs['norm_score'] = 0.5
            
        if not content_recs.empty and content_recs['score'].max() != content_recs['score'].min():
            content_recs = content_recs.copy()
            score_range = content_recs['score'].max() - content_recs['score'].min()
            content_recs['norm_score'] = (content_recs['score'] - content_recs['score'].min()) / score_range
        else:
            content_recs['norm_score'] = 0.5
        
        # Combine scores with diversity bonus
        all_items = set(collab_recs['item_id'].tolist() + content_recs['item_id'].tolist())
        combined_scores = {}
        
        for item_id in all_items:
            collab_score = 0
            content_score = 0
            
            if item_id in collab_recs['item_id'].values:
                collab_score = collab_recs[collab_recs['item_id'] == item_id]['norm_score'].iloc[0]
            
            if item_id in content_recs['item_id'].values:
                content_score = content_recs[content_recs['item_id'] == item_id]['norm_score'].iloc[0]
            
            # Boost items that appear in both recommendations
            diversity_bonus = 0.1 if (collab_score > 0 and content_score > 0) else 0
            
            final_score = (collab_weight * collab_score + 
                          (1 - collab_weight) * content_score + 
                          diversity_bonus)
            
            combined_scores[item_id] = final_score
        
        # Create final recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        rec_df = pd.DataFrame(sorted_items, columns=['item_id', 'score'])
        
        return rec_df.merge(
            self.items[['item_id', 'title', 'primary_genre']], 
            on='item_id', how='left'
        ).dropna()


# FastAPI Application - Simple version for Python 3.9 compatibility
app = FastAPI(
    title="Movie Recommendation API", 
    version="2.0.0",
    description="Advanced movie recommendation system with hybrid ML algorithms"
)

# Global recommendation engine
rec_engine = None

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with better error handling"""
    global rec_engine
    logger.info("🚀 Starting Movie Recommendation Engine...")
    
    try:
        # Initialize with your data path
        data_path = r"C:\Users\LookO\Movie-Rec\data"
        rec_engine = MovieRecommendationEngine(data_path)
        
        # Load and train models with progress tracking
        logger.info("📊 Loading and preprocessing data...")
        rec_engine.load_and_preprocess_data(nrows=500000)
        
        logger.info("🤖 Training ML models...")
        rec_engine.train_models()
        
        logger.info("📈 Evaluating model performance...")
        metrics = rec_engine.evaluate_models()
        
        # Log summary statistics
        if metrics:
            logger.info("🎉 Startup completed successfully!")
            logger.info("📊 Performance Summary:")
            for model, metric in metrics.items():
                logger.info(f"   {model.upper()}: RMSE={metric.get('rmse', 'N/A'):.4f}, "
                          f"P@10={metric.get('precision_at_10', 'N/A'):.4f}")
        else:
            logger.warning("⚠️ No evaluation metrics available")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations with enhanced validation and monitoring"""
    if rec_engine is None:
        logger.error("❌ Recommendation engine not ready")
        raise HTTPException(status_code=503, detail="Recommendation engine not ready")
    
    # Simple validation
    if request.user_id < 1:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    if request.num_recommendations < 1 or request.num_recommendations > 100:
        raise HTTPException(status_code=400, detail="num_recommendations must be between 1 and 100")
    
    if request.method not in ["hybrid", "collaborative", "content", "svd"]:
        raise HTTPException(status_code=400, detail="Invalid recommendation method")
    
    try:
        logger.info(f"🔍 Generating {request.method} recommendations for user {request.user_id}")
        
        result = await rec_engine.get_recommendations(
            user_id=request.user_id,
            method=request.method,
            num_recommendations=request.num_recommendations
        )
        
        logger.info(f"✅ Generated {len(result['recommendations'])} recommendations "
                   f"in {result['response_time_ms']:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎬 Movie Recommendation API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/metrics": "Model performance metrics",
            "/recommendations": "POST - Get movie recommendations",
            "/user/{user_id}/profile": "Get user profile",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    health_status = {
        "status": "healthy" if rec_engine is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }
    
    if rec_engine is not None:
        health_status.update({
            "models_loaded": len(rec_engine.models),
            "data_loaded": rec_engine.interactions is not None,
            "matrix_shape": list(rec_engine.user_item_matrix.shape) if rec_engine.user_item_matrix is not None else None
        })
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive model performance metrics"""
    if rec_engine is None or not rec_engine.metrics:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    # Add system statistics
    enhanced_metrics = {
        "model_performance": rec_engine.metrics,
        "system_stats": {
            "total_interactions": len(rec_engine.interactions) if rec_engine.interactions is not None else 0,
            "total_items": len(rec_engine.items) if rec_engine.items is not None else 0,
            "matrix_shape": list(rec_engine.user_item_matrix.shape) if rec_engine.user_item_matrix is not None else None,
            "models_available": list(rec_engine.models.keys())
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return enhanced_metrics

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get user interaction profile and preferences"""
    if rec_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not ready")
    
    try:
        user_interactions = rec_engine.interactions[
            rec_engine.interactions['user_id'] == user_id
        ]
        
        if user_interactions.empty:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate user statistics
        user_stats = {
            "user_id": user_id,
            "total_interactions": len(user_interactions),
            "avg_watch_percentage": user_interactions['watched_pct'].mean(),
            "favorite_genres": user_interactions.merge(
                rec_engine.items[['item_id', 'primary_genre']], 
                on='item_id'
            )['primary_genre'].value_counts().head(5).to_dict(),
            "recent_items": user_interactions.nlargest(10, 'watched_pct')[
                ['item_id', 'watched_pct']
            ].merge(
                rec_engine.items[['item_id', 'title', 'primary_genre']], 
                on='item_id'
            ).to_dict('records')
        }
        
        return user_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced load testing
async def load_test(num_requests: int = 1000, concurrent_users: int = 50):
    """Enhanced load testing with detailed analytics"""
    import aiohttp
    import asyncio
    from collections import defaultdict
    
    logger.info(f"🧪 Starting load test: {num_requests} requests, {concurrent_users} concurrent users")
    
    async def make_request(session, user_id, request_id):
        try:
            start_time = time.time()
            async with session.post('http://localhost:8000/recommendations', 
                                  json={
                                      "user_id": user_id, 
                                      "num_recommendations": 10,
                                      "method": "hybrid"
                                  }) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "success": True,
                    "response_time": (end_time - start_time) * 1000,
                    "status_code": response.status,
                    "request_id": request_id
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create batched requests to simulate realistic load
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def bounded_request(user_id, request_id):
            async with semaphore:
                return await make_request(session, user_id, request_id)
        
        # Generate requests
        tasks = [
            bounded_request(i % 100 + 1, i) 
            for i in range(num_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            analytics = {
                "total_requests": num_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / num_requests * 100,
                "duration_seconds": duration,
                "requests_per_second": len(successful_requests) / duration,
                "requests_per_minute": (len(successful_requests) / duration) * 60,
                "response_time_stats": {
                    "min_ms": min(response_times),
                    "max_ms": max(response_times),
                    "avg_ms": np.mean(response_times),
                    "median_ms": np.median(response_times),
                    "p95_ms": np.percentile(response_times, 95),
                    "p99_ms": np.percentile(response_times, 99)
                }
            }
            
            logger.info("🎯 Load Test Results:")
            logger.info(f"   📊 Success Rate: {analytics['success_rate']:.1f}%")
            logger.info(f"   ⚡ Requests/min: {analytics['requests_per_minute']:.0f}")
            logger.info(f"   📈 Avg Response Time: {analytics['response_time_stats']['avg_ms']:.2f}ms")
            logger.info(f"   📈 P95 Response Time: {analytics['response_time_stats']['p95_ms']:.2f}ms")
            logger.info(f"   📈 P99 Response Time: {analytics['response_time_stats']['p99_ms']:.2f}ms")
            
            return analytics
        else:
            logger.error("❌ All requests failed during load test")
            return {"error": "All requests failed"}

if __name__ == "__main__":
    import sys
    
    # Check if running in Jupyter/IPython
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Running in Jupyter - use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("🔧 Applied nest_asyncio for Jupyter compatibility")
    except ImportError:
        pass
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run load test
        async def run_test():
            await asyncio.sleep(2)  # Wait for server to start
            await load_test(num_requests=500, concurrent_users=25)
        
        asyncio.create_task(run_test())
    
    # Run the FastAPI server
    logger.info("🌟 Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Disable reload for better performance
    )