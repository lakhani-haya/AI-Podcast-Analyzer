# 🎧 AI Podcast Topic Analyzer & Recommender

A tool for analyzing podcast episodes to extract topics, analyze sentiment, and find similar episodes.

## 📋 Features

- **📥 Data Fetching**: Get podcast episodes from ListenNotes API
- **🧠 Topic Extraction**: Find key topics in episode descriptions
- **😊 Sentiment Analysis**: Check if episodes are positive, negative, or neutral
- **💡 Episode Recommendations**: Find episodes similar to ones you like
- **📊 Visualizations**: See charts and graphs of your podcast data
- **🌐 Web Interface**: Browse results in your web browser

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Git

### 2. Installation

1. **Clone or download this repository**
2. **Navigate to the project directory**:
   ```bash
   cd podcast_topic_analyzer
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Setup API Key

1. **Get your free API key** from [ListenNotes](https://www.listennotes.com/api/)
2. **Update the `.env` file** with your API key:
   ```
   LISTEN_API_KEY=f972740ef94a4629af091853e32574b8
   ```

### 4. Run the Analysis

```bash
python run_analysis.py
```

### 5. Launch the Web Interface

```bash
cd app
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
podcast_topic_analyzer/
├── data/
│   ├── episodes.json
│   ├── topics.csv
│   ├── topics_with_sentiment.csv
│   └── recommendations.json
├── scripts/
│   ├── fetch_podcasts.py
│   ├── extract_topics.py
│   ├── sentiment_analysis.py
│   ├── recommender.py
│   └── visualize_data.py
├── visualizations/
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── .env
└── run_analysis.py
```

##  Output Files

### Data Files

1. **`episodes.json`**: Raw episode data from API
2. **`topics.csv`**: Episodes with extracted topics
3. **`topics_with_sentiment.csv`**: Complete analysis with sentiment
4. **`recommendations.json`**: Episode recommendations

### Visualizations

- **`sentiment_analysis.png`**: Sentiment distribution charts
- **`topic_analysis.png`**: Topic frequency and relationships
- **`temporal_analysis.png`**: Trends over time
- **`summary_dashboard.png`**: Comprehensive overview

## 🌐 Web Interface Features

### 📊 Dashboard
- Key metrics and statistics
- Sentiment distribution
- Topic overview

### 🔍 Episode Explorer
- Filter by sentiment and topic
- Search functionality
- Sortable episode list

### 💡 Recommendations
- Search-based recommendations
- Topic-based browsing
- Similarity scoring

### 📈 Analytics
- Temporal trends
- Topic-sentiment relationships
- Correlation analysis

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Errors**: Check your API key in `.env` file

3. **Memory Issues**: Reduce `max_episodes` in fetch script for large datasets

4. **Visualization Errors**: Install additional dependencies:
   ```bash
   pip install beautifulsoup4
   ```

### Performance Tips

- Start with fewer episodes for testing
- Use lighter transformer models for faster processing
- Cache embeddings to avoid recomputation

## 📚 Technical Details

### Libraries Used

- **Data Processing**: pandas, numpy
- **Topic Modeling**: BERTopic, sentence-transformers
- **Sentiment Analysis**: vaderSentiment
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Web Interface**: streamlit
- **API**: requests

### Models Used

- **Topic Extraction**: BERTopic with sentence-transformers
- **Embeddings**: all-MiniLM-L6-v2 (lightweight and fast)
- **Sentiment**: VADER (optimized for social media text)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [ListenNotes API](https://www.listennotes.com/api/) for podcast data
- [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are properly installed
4. Verify your API key configuration

---

**Happy analyzing! 🎧✨**