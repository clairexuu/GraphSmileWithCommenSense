import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd


def analyze_class_distribution(data_path):
    """Analyze class distribution in MELD dataset."""

    # Load the data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    (videoIDs, videoSpeakers, videoLabels, videoSentiments,
     videoText0, videoText1, videoText2, videoText3, videoAudio, videoVisual,
     videoSentence, trainVid, testVid, _) = data

    print("=" * 60)
    print("MELD Dataset Class Distribution Analysis")
    print("=" * 60)

    # Emotion labels mapping
    emotion_labels = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness',
                      4: 'joy', 5: 'disgust', 6: 'anger'}

    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def analyze_split(vid_list, split_name):
        print(f"\n{split_name.upper()} SET:")
        print("-" * 40)

        # Collect all labels for this split
        all_emotions = []
        all_sentiments = []
        dialogue_lengths = []

        for vid in vid_list:
            emotions = videoLabels[vid]
            sentiments = videoSentiments[vid]
            all_emotions.extend(emotions)
            all_sentiments.extend(sentiments)
            dialogue_lengths.append(len(emotions))

        print(f"Total dialogues: {len(vid_list)}")
        print(f"Total utterances: {len(all_emotions)}")
        print(f"Average dialogue length: {np.mean(dialogue_lengths):.2f}")
        print(f"Dialogue length range: {min(dialogue_lengths)}-{max(dialogue_lengths)}")

        # Emotion distribution
        emotion_counts = Counter(all_emotions)
        print(f"\nEMOTION DISTRIBUTION:")
        print(f"{'Class':<10} {'Count':<8} {'Percentage':<12} {'Label'}")
        print("-" * 45)

        total_emotions = len(all_emotions)
        emotion_stats = {}
        for emotion_id in sorted(emotion_counts.keys()):
            count = emotion_counts[emotion_id]
            percentage = (count / total_emotions) * 100
            label = emotion_labels[emotion_id]
            emotion_stats[emotion_id] = {'count': count, 'percentage': percentage, 'label': label}
            print(f"{emotion_id:<10} {count:<8} {percentage:<12.2f}% {label}")

        # Sentiment distribution
        sentiment_counts = Counter(all_sentiments)
        print(f"\nSENTIMENT DISTRIBUTION:")
        print(f"{'Class':<10} {'Count':<8} {'Percentage':<12} {'Label'}")
        print("-" * 45)

        total_sentiments = len(all_sentiments)
        sentiment_stats = {}
        for sentiment_id in sorted(sentiment_counts.keys()):
            count = sentiment_counts[sentiment_id]
            percentage = (count / total_sentiments) * 100
            label = sentiment_labels[sentiment_id]
            sentiment_stats[sentiment_id] = {'count': count, 'percentage': percentage, 'label': label}
            print(f"{sentiment_id:<10} {count:<8} {percentage:<12.2f}% {label}")

        return emotion_stats, sentiment_stats, dialogue_lengths

    # Analyze train and test splits
    train_emotion_stats, train_sentiment_stats, train_dialogue_lengths = analyze_split(trainVid, "train")
    test_emotion_stats, test_sentiment_stats, test_dialogue_lengths = analyze_split(testVid, "test")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Train emotion distribution
    train_emotion_labels = [train_emotion_stats[i]['label'] for i in sorted(train_emotion_stats.keys())]
    train_emotion_counts = [train_emotion_stats[i]['count'] for i in sorted(train_emotion_stats.keys())]

    axes[0, 0].bar(train_emotion_labels, train_emotion_counts, color='skyblue')
    axes[0, 0].set_title('Train Set - Emotion Distribution')
    axes[0, 0].set_xlabel('Emotion')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Test emotion distribution
    test_emotion_labels = [test_emotion_stats[i]['label'] for i in sorted(test_emotion_stats.keys())]
    test_emotion_counts = [test_emotion_stats[i]['count'] for i in sorted(test_emotion_stats.keys())]

    axes[0, 1].bar(test_emotion_labels, test_emotion_counts, color='lightcoral')
    axes[0, 1].set_title('Test Set - Emotion Distribution')
    axes[0, 1].set_xlabel('Emotion')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Train sentiment distribution
    train_sentiment_labels = [train_sentiment_stats[i]['label'] for i in sorted(train_sentiment_stats.keys())]
    train_sentiment_counts = [train_sentiment_stats[i]['count'] for i in sorted(train_sentiment_stats.keys())]

    axes[1, 0].bar(train_sentiment_labels, train_sentiment_counts, color='lightgreen')
    axes[1, 0].set_title('Train Set - Sentiment Distribution')
    axes[1, 0].set_xlabel('Sentiment')
    axes[1, 0].set_ylabel('Count')

    # Test sentiment distribution
    test_sentiment_labels = [test_sentiment_stats[i]['label'] for i in sorted(test_sentiment_stats.keys())]
    test_sentiment_counts = [test_sentiment_stats[i]['count'] for i in sorted(test_sentiment_stats.keys())]

    axes[1, 1].bar(test_sentiment_labels, test_sentiment_counts, color='gold')
    axes[1, 1].set_title('Test Set - Sentiment Distribution')
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate imbalance metrics
    print(f"\n" + "=" * 60)
    print("IMBALANCE ANALYSIS")
    print("=" * 60)

    # Emotion imbalance
    train_emotion_counts_list = [train_emotion_stats[i]['count'] for i in sorted(train_emotion_stats.keys())]
    max_emotion = max(train_emotion_counts_list)
    min_emotion = min(train_emotion_counts_list)
    emotion_imbalance_ratio = max_emotion / min_emotion

    print(f"Emotion Imbalance Ratio (Train): {emotion_imbalance_ratio:.2f}")
    print(f"Most frequent emotion: {emotion_labels[np.argmax(train_emotion_counts_list)]} ({max_emotion} samples)")
    print(f"Least frequent emotion: {emotion_labels[np.argmin(train_emotion_counts_list)]} ({min_emotion} samples)")

    # Sentiment imbalance
    train_sentiment_counts_list = [train_sentiment_stats[i]['count'] for i in sorted(train_sentiment_stats.keys())]
    max_sentiment = max(train_sentiment_counts_list)
    min_sentiment = min(train_sentiment_counts_list)
    sentiment_imbalance_ratio = max_sentiment / min_sentiment

    print(f"\nSentiment Imbalance Ratio (Train): {sentiment_imbalance_ratio:.2f}")
    print(
        f"Most frequent sentiment: {sentiment_labels[np.argmax(train_sentiment_counts_list)]} ({max_sentiment} samples)")
    print(
        f"Least frequent sentiment: {sentiment_labels[np.argmin(train_sentiment_counts_list)]} ({min_sentiment} samples)")

    # Save detailed statistics
    stats_df = pd.DataFrame({
        'Emotion_ID': list(train_emotion_stats.keys()),
        'Emotion_Label': [train_emotion_stats[i]['label'] for i in train_emotion_stats.keys()],
        'Train_Count': [train_emotion_stats[i]['count'] for i in train_emotion_stats.keys()],
        'Train_Percentage': [train_emotion_stats[i]['percentage'] for i in train_emotion_stats.keys()],
        'Test_Count': [test_emotion_stats[i]['count'] for i in train_emotion_stats.keys()],
        'Test_Percentage': [test_emotion_stats[i]['percentage'] for i in train_emotion_stats.keys()],
    })

    stats_df.to_csv('class_distribution_stats.csv', index=False)
    print(f"\nDetailed statistics saved to: class_distribution_stats.csv")
    print(f"Visualization saved to: class_distribution_analysis.png")

    return train_emotion_stats, test_emotion_stats, train_sentiment_stats, test_sentiment_stats


if __name__ == "__main__":
    data_path = "../features/meld_multi_features.pkl"
    analyze_class_distribution(data_path)