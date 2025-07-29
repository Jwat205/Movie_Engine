# Movie Recommendation Engine

A FastAPI-based movie recommendation system that I built to learn about machine learning and backend development. The system uses collaborative filtering and content-based approaches to suggest movies to users.

## What This Does

This is a REST API that recommends movies based on user preferences. I trained it on a dataset of 500,000 movie interactions and built several machine learning models to make predictions. The API can handle different recommendation methods and returns results pretty quickly.

## Features

- **Multiple recommendation algorithms**: Collaborative filtering (NMF, SVD), content-based filtering, and a hybrid approach
- **Fast response times**: Most requests complete in under 100ms after I optimized the caching
- **REST API**: Built with FastAPI, includes automatic documentation
- **Health monitoring**: Basic health checks and performance metrics
- **User profiles**: Can look up user viewing history and preferences

## Tech Stack

- **Backend**: FastAPI with Python 3.9
- **ML Libraries**: scikit-learn, pandas, numpy
- **Data**: MovieLens-style dataset with user ratings and movie metadata
- **Server**: Uvicorn ASGI server

## Getting Started

### Prerequisites

You'll need Python 3.9+ and about 8GB of RAM to process the full dataset.

### Installation

```bash
git clone https://github.com/jwat205/movie-recommendation-engine
cd movie-recommendation-engine

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn pydantic nest-asyncio
```

### Data Setup

You'll need two CSV files in a `data/` directory:
- `clean_interactions.csv` - user interactions with movies
- `clean_items_en.csv` - movie metadata (titles, genres, etc.)

Update the data path in `main.py`:
```python
data_path = r"C:\path\to\your\data"
```

### Running the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000` and you can see the docs at `http://localhost:8000/docs`.

## API Usage

### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": 1,
       "num_recommendations": 10,
       "method": "hybrid"
     }'
```

Available methods: `hybrid`, `collaborative`, `content`, `svd`

### Check System Health

```bash
curl http://localhost:8000/health
```

### Get Performance Metrics

```bash
curl http://localhost:8000/metrics
```

## How It Works

### Data Processing

The system loads user interaction data and movie metadata, then creates a user-item matrix. To keep memory usage reasonable, I filter to the most active users and popular movies (about 2000 each).

### Machine Learning Models

1. **NMF (Non-negative Matrix Factorization)**: Finds latent features in user preferences
2. **SVD (Singular Value Decomposition)**: Dimensionality reduction for collaborative filtering  
3. **Content-based**: Uses TF-IDF on movie genres and titles for similarity matching
4. **Hybrid**: Combines collaborative and content approaches

### Performance Optimizations

I spent time optimizing this after the initial version was pretty slow:

- Pre-compute prediction matrices during startup instead of every request
- Cache popular movie recommendations 
- Use pandas indexing for faster data merges
- Simplified the hybrid algorithm logic

This got response times down from ~130ms to about 8-12ms for most requests.

## Project Structure

```
movie-recommendation-engine/
├── main.py                 # Main FastAPI application
├── data/                   # CSV data files (not included)
│   ├── clean_interactions.csv
│   └── clean_items_en.csv
└── README.md
```

## Current Performance

- **Response times**: 8-12ms average for cached requests
- **Data processing**: ~6 seconds to load and process 500K interactions
- **Models**: NMF performs best with RMSE of 2.52
- **Concurrent requests**: Handles multiple requests well due to async design

## Known Issues

- The OpenAPI docs sometimes fail to load due to a Pydantic version compatibility issue
- Memory usage is high during initial data processing
- Some users might not have enough interaction data for good collaborative filtering

## Future Improvements

- Add user authentication and rate limiting
- Implement more sophisticated hybrid algorithms
- Add A/B testing framework for model comparison
- Better handling of cold start problem for new users
- Add caching with Redis for production deployment

## What I Learned

This project helped me understand:
- Building production APIs with FastAPI
- Machine learning model evaluation and optimization
- Performance profiling and optimization techniques
- Async programming patterns in Python
- System monitoring and observability

## Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

## License

MIT License - feel free to use this code for learning or your own projects.

---

This was a fun learning project that taught me a lot about both machine learning and backend development. The performance optimization part was particularly interesting - it's amazing how much difference good caching and data structure choices can make.
