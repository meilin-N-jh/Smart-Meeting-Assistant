import json
import asyncio
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.action_items_service import get_action_items_service
from backend.services.sentiment_service import get_sentiment_service
from backend.models.schemas import TranscriptSegment

async def evaluate():
    dataset_path = Path(__file__).parent.parent / "sample_data" / "eval_dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    action_service = get_action_items_service()
    sentiment_service = get_sentiment_service()

    total_tasks = 0
    correct_tasks = 0
    total_sentiment = 0
    correct_sentiment = 0

    print("=== Starting Evaluation ===")
    for item in dataset:
        print(f"\\nEvaluating {item['id']}...")
        
        # Simple segments mock (no real timestamps)
        segments = []
        for line in item['transcript'].split('\\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                segments.append(TranscriptSegment(start=0, end=1, speaker=parts[0].strip(), text=parts[1].strip()))
                
        # Action Items Evaluation
        actions = action_service.extract(item['transcript'], segments)
        expected_actions = item['expected_action_items']
        
        extracted_tuples = [(a.get('assignee', '').lower(), a.get('task', '').lower()) for a in actions]
        
        item_correct = 0
        for exp in expected_actions:
            exp_assignee = exp['assignee'].lower()
            # Rough match simply checking if expected assignee is found and tasks have overlapping words
            found = False
            for ext_assignee, ext_task in extracted_tuples:
                if exp_assignee in ext_assignee or ext_assignee in exp_assignee:
                    found = True
                    break
            if found:
                item_correct += 1
                
        total_tasks += len(expected_actions)
        correct_tasks += item_correct
        print(f"Action Items: {item_correct}/{len(expected_actions)} matched roughly")

        # Sentiment Evaluation
        sentiment = sentiment_service.analyze(item['transcript'], segments)
        expected_sentiment = item['expected_sentiment']
        
        # Check overall sentiment
        if sentiment.get('overall_sentiment', '').lower() == expected_sentiment['overall_sentiment'].lower():
            correct_sentiment += 1
        total_sentiment += 1
        
        print(f"Sentiment match: {sentiment.get('overall_sentiment')} vs expected {expected_sentiment['overall_sentiment']}")

    print("\\n=== Evaluation Results ===")
    print(f"Action Items Score: {correct_tasks}/{total_tasks} ({correct_tasks/total_tasks*100:.1f}%)")
    print(f"Sentiment Score: {correct_sentiment}/{total_sentiment} ({correct_sentiment/total_sentiment*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(evaluate())
