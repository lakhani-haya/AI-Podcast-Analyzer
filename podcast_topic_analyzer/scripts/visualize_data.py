import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for beplots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PodcastVisualizer:
    def __init__(self, data_path="data/topics_with_sentiment.csv", output_dir="visualizations"):
        """
        Initialize the visualizer
        
        Args:
            data_path (str): Path to the CSV file with podcast data
            output_dir (str): Directory to save visualizations
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load the podcast data"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            
            # Convert pub_date to datetime if it exists
            if 'pub_date' in self.df.columns:
                self.df['pub_date'] = pd.to_datetime(self.df['pub_date'])
            
            print(f"Loaded {len(self.df)} episodes for visualization")
            return True
        except FileNotFoundError:
            print(f"Data file not found: {self.data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def plot_sentiment_distribution(self):
        """Create sentiment distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall sentiment labels pie chart
        sentiment_counts = self.df['overall_sentiment_label'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment scores histogram
        axes[0, 1].hist(self.df['overall_sentiment_compound'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Sentiment Score (Compound)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Sentiment Scores')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        axes[0, 1].legend()
        
        # 3. Detailed sentiment components
        sentiment_components = ['overall_sentiment_positive', 'overall_sentiment_neutral', 'overall_sentiment_negative']
        sentiment_means = [self.df[col].mean() for col in sentiment_components]
        labels = ['Positive', 'Neutral', 'Negative']
        
        axes[1, 0].bar(labels, sentiment_means, color=['green', 'gray', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Average Score')
        axes[1, 0].set_title('Average Sentiment Component Scores')
        
        # 4. Box plot of sentiment by label
        sentiment_data = []
        sentiment_labels = []
        for label in self.df['overall_sentiment_label'].unique():
            sentiment_data.append(self.df[self.df['overall_sentiment_label'] == label]['overall_sentiment_compound'])
            sentiment_labels.append(label)
        
        axes[1, 1].boxplot(sentiment_data, labels=sentiment_labels)
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].set_title('Sentiment Score Distribution by Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sentiment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("Sentiment distribution plots saved!")
    
    def plot_topic_analysis(self):
        """Create topic analysis plots"""
        if 'topic_label' not in self.df.columns:
            print("Topic labels not found in data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Topic frequency
        topic_counts = self.df['topic_label'].value_counts().head(10)
        axes[0, 0].barh(range(len(topic_counts)), topic_counts.values)
        axes[0, 0].set_yticks(range(len(topic_counts)))
        axes[0, 0].set_yticklabels(topic_counts.index, fontsize=10)
        axes[0, 0].set_xlabel('Number of Episodes')
        axes[0, 0].set_title('Top 10 Most Common Topics')
        
        # 2. Topic sentiment relationship
        topic_sentiment = self.df.groupby('topic_label')['overall_sentiment_compound'].mean().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(topic_sentiment)), topic_sentiment.values, 
                       color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in topic_sentiment.values])
        axes[0, 1].set_yticks(range(len(topic_sentiment)))
        axes[0, 1].set_yticklabels(topic_sentiment.index, fontsize=10)
        axes[0, 1].set_xlabel('Average Sentiment Score')
        axes[0, 1].set_title('Average Sentiment by Topic')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Topic probability distribution
        if 'topic_probability' in self.df.columns:
            axes[1, 0].hist(self.df['topic_probability'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Topic Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Topic Assignment Probabilities')
        
        # 4. Sentiment vs Topic heatmap (top topics only)
        top_topics = self.df['topic_label'].value_counts().head(8).index
        sentiment_topic_crosstab = pd.crosstab(
            self.df[self.df['topic_label'].isin(top_topics)]['topic_label'],
            self.df[self.df['topic_label'].isin(top_topics)]['overall_sentiment_label']
        )
        
        sns.heatmap(sentiment_topic_crosstab, annot=True, fmt='d', ax=axes[1, 1], cmap='YlOrRd')
        axes[1, 1].set_title('Topic vs Sentiment Cross-tabulation')
        axes[1, 1].set_xlabel('Sentiment Label')
        axes[1, 1].set_ylabel('Topic Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'topic_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("Topic analysis plots saved!")
    
    def plot_temporal_analysis(self):
        """Create temporal analysis plots"""
        if 'pub_date' not in self.df.columns:
            print("Publication date not found in data")
            return
        
        # Filter out invalid dates
        valid_dates_df = self.df.dropna(subset=['pub_date'])
        if valid_dates_df.empty:
            print("No valid publication dates found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sentiment over time
        daily_sentiment = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date)['overall_sentiment_compound'].mean()
        
        axes[0, 0].plot(daily_sentiment.index, daily_sentiment.values, marker='o', alpha=0.7)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Sentiment Score')
        axes[0, 0].set_title('Sentiment Trends Over Time')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Episode count over time
        daily_counts = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date).size()
        
        axes[0, 1].bar(daily_counts.index, daily_counts.values, alpha=0.7)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Episodes')
        axes[0, 1].set_title('Episode Publication Frequency')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Monthly sentiment analysis
        monthly_sentiment = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.to_period('M'))['overall_sentiment_compound'].agg(['mean', 'std'])
        
        axes[1, 0].errorbar(range(len(monthly_sentiment)), monthly_sentiment['mean'], 
                           yerr=monthly_sentiment['std'], marker='o', capsize=5, alpha=0.7)
        axes[1, 0].set_xticks(range(len(monthly_sentiment)))
        axes[1, 0].set_xticklabels([str(period) for period in monthly_sentiment.index], rotation=45)
        axes[1, 0].set_ylabel('Average Sentiment Score')
        axes[1, 0].set_title('Monthly Sentiment with Standard Deviation')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Day of week analysis
        valid_dates_df['day_of_week'] = valid_dates_df['pub_date'].dt.day_name()
        day_sentiment = valid_dates_df.groupby('day_of_week')['overall_sentiment_compound'].mean()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sentiment = day_sentiment.reindex([day for day in day_order if day in day_sentiment.index])
        
        axes[1, 1].bar(day_sentiment.index, day_sentiment.values, alpha=0.7)
        axes[1, 1].set_ylabel('Average Sentiment Score')
        axes[1, 1].set_title('Average Sentiment by Day of Week')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("Temporal analysis plots saved!")
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        sentiment_counts = self.df['overall_sentiment_label'].value_counts()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.set_title('Sentiment Distribution')
        
        # 2. Sentiment histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.df['overall_sentiment_compound'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sentiment Score Distribution')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Top topics
        ax3 = fig.add_subplot(gs[0, 2:])
        if 'topic_label' in self.df.columns:
            topic_counts = self.df['topic_label'].value_counts().head(8)
            ax3.barh(range(len(topic_counts)), topic_counts.values)
            ax3.set_yticks(range(len(topic_counts)))
            ax3.set_yticklabels(topic_counts.index, fontsize=10)
            ax3.set_xlabel('Number of Episodes')
            ax3.set_title('Top Topics')
        
        # 4. Topic sentiment
        ax4 = fig.add_subplot(gs[1, :])
        if 'topic_label' in self.df.columns:
            topic_sentiment = self.df.groupby('topic_label')['overall_sentiment_compound'].mean().sort_values(ascending=False).head(10)
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in topic_sentiment.values]
            ax4.barh(range(len(topic_sentiment)), topic_sentiment.values, color=colors, alpha=0.7)
            ax4.set_yticks(range(len(topic_sentiment)))
            ax4.set_yticklabels(topic_sentiment.index, fontsize=10)
            ax4.set_xlabel('Average Sentiment Score')
            ax4.set_title('Topic Sentiment Analysis')
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 5. Time series (if available)
        if 'pub_date' in self.df.columns:
            valid_dates_df = self.df.dropna(subset=['pub_date'])
            if not valid_dates_df.empty:
                ax5 = fig.add_subplot(gs[2, :2])
                daily_sentiment = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date)['overall_sentiment_compound'].mean()
                ax5.plot(daily_sentiment.index, daily_sentiment.values, marker='o', alpha=0.7)
                ax5.set_xlabel('Date')
                ax5.set_ylabel('Avg Sentiment')
                ax5.set_title('Sentiment Over Time')
                ax5.tick_params(axis='x', rotation=45)
                ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                ax6 = fig.add_subplot(gs[2, 2:])
                daily_counts = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date).size()
                ax6.bar(daily_counts.index, daily_counts.values, alpha=0.7)
                ax6.set_xlabel('Date')
                ax6.set_ylabel('Episode Count')
                ax6.set_title('Episodes Over Time')
                ax6.tick_params(axis='x', rotation=45)
        
        # 6. Statistics summary
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary statistics
        stats_text = f"""
        PODCAST ANALYSIS SUMMARY
        
        Total Episodes: {len(self.df)}
        Average Sentiment Score: {self.df['overall_sentiment_compound'].mean():.3f}
        Most Positive Episode: {self.df.loc[self.df['overall_sentiment_compound'].idxmax(), 'title'][:60]}...
        Most Negative Episode: {self.df.loc[self.df['overall_sentiment_compound'].idxmin(), 'title'][:60]}...
        
        Sentiment Distribution:
        • Positive: {(self.df['overall_sentiment_label'] == 'positive').sum()} ({(self.df['overall_sentiment_label'] == 'positive').mean()*100:.1f}%)
        • Neutral: {(self.df['overall_sentiment_label'] == 'neutral').sum()} ({(self.df['overall_sentiment_label'] == 'neutral').mean()*100:.1f}%)
        • Negative: {(self.df['overall_sentiment_label'] == 'negative').sum()} ({(self.df['overall_sentiment_label'] == 'negative').mean()*100:.1f}%)
        """
        
        if 'topic_label' in self.df.columns:
            stats_text += f"\nTotal Topics: {self.df['topic_label'].nunique()}"
            stats_text += f"\nMost Common Topic: {self.df['topic_label'].value_counts().index[0]}"
        
        ax7.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("Summary dashboard saved!")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        if not self.load_data():
            return
        
        print("Generating visualizations...")
        
        try:
            self.plot_sentiment_distribution()
            self.plot_topic_analysis()
            self.plot_temporal_analysis()
            self.create_summary_dashboard()
            
            print(f"\nAll visualizations saved to {self.output_dir}/")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")

def main():
    """Main function to run visualizations"""
    print("Starting Podcast Data Visualization...")
    
    visualizer = PodcastVisualizer()
    visualizer.generate_all_visualizations()
    
    return visualizer

if __name__ == "__main__":
    visualizer = main()
