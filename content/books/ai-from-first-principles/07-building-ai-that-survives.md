# Chapter 7 — Building AI That Survives Reality

## The Crux
Training a model is the beginning, not the end. Real AI systems must survive production: user drift, data drift, adversarial inputs, scaling, cost constraints. This chapter is about the unglamorous, essential work of making AI reliable.

## Monitoring Model Drift

You deploy a model. It works. Six months later, it fails. What happened?

### Data Drift

**Definition**: The input distribution changes.

**Example**: You trained a spam classifier on 2020 emails. In 2024, spammers use new tactics (crypto scams, AI-generated text). Your model hasn't seen these patterns.

**Detection**: Monitor input feature distributions. Alert if they shift significantly (KL divergence, Kolmogorov-Smirnov test).

### Concept Drift

**Definition**: The relationship between inputs and outputs changes.

**Example**: A model predicts housing prices based on interest rates, location, etc. Then a recession hits. Same inputs now predict different prices.

**Detection**: Monitor model performance over time. If accuracy drops, you have concept drift.

### Label Drift

**Definition**: The distribution of outputs changes.

**Example**: You trained a sentiment classifier on product reviews. Initially, 80% positive. Now, a bad product launch skews reviews to 60% negative. Model was calibrated for 80% positive.

**Detection**: Monitor predicted label distributions. Compare to historical baselines.

## How to Monitor

### 1. Log Everything

- Inputs (features)
- Outputs (predictions)
- Ground truth (when available)
- Metadata (timestamp, user ID, version)

### 2. Dashboards

- **Input distributions**: Histograms, summary stats. Alert on shifts.
- **Prediction distributions**: Are you suddenly predicting "spam" 90% of the time?
- **Performance metrics**: Accuracy, precision, recall over time (requires labels).
- **Latency and throughput**: Is inference getting slower?

### 3. Alerts

- If input feature X exceeds historical range
- If prediction distribution shifts >10% from baseline
- If latency exceeds SLA
- If error rate spikes

### 4. Periodic Retraining

Even without alerts, retrain on fresh data every N months. The world changes. Your model must adapt.

### Complete Example: Detecting and Handling Model Drift

This example demonstrates the full drift detection workflow: train a model, simulate drift, detect it statistically, observe performance degradation, and recover through retraining.

```python
"""
Model Drift Detection: A Complete Example

This script demonstrates:
1. Training a model on "2020" data
2. Simulating data drift (2024 conditions)
3. Detecting drift with statistical tests
4. Observing performance degradation
5. Retraining to recover

pip install numpy pandas scikit-learn scipy matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# STEP 1: Generate "2020" training data (spam classification)
# =============================================================================
print("=" * 70)
print("STEP 1: Generate 2020 Training Data")
print("=" * 70)

def generate_email_data(n_samples, year="2020"):
    """
    Generate synthetic email features for spam classification.

    Features:
    - word_count: Number of words
    - link_count: Number of links
    - urgent_words: Count of urgent language ("act now", "limited time")
    - money_mentions: References to money, prices, deals
    - sender_reputation: Score from 0-1 (1 = trusted sender)
    """
    if year == "2020":
        # 2020 spam patterns
        spam_ratio = 0.3
        n_spam = int(n_samples * spam_ratio)
        n_ham = n_samples - n_spam

        # Ham (legitimate emails)
        ham_data = {
            'word_count': np.random.normal(150, 50, n_ham).clip(20, 500),
            'link_count': np.random.poisson(1.5, n_ham),
            'urgent_words': np.random.poisson(0.3, n_ham),
            'money_mentions': np.random.poisson(0.5, n_ham),
            'sender_reputation': np.random.beta(8, 2, n_ham),  # Mostly high
            'is_spam': np.zeros(n_ham)
        }

        # Spam (2020 patterns: Nigerian prince, lottery, etc.)
        spam_data = {
            'word_count': np.random.normal(80, 30, n_spam).clip(20, 200),
            'link_count': np.random.poisson(5, n_spam),
            'urgent_words': np.random.poisson(4, n_spam),
            'money_mentions': np.random.poisson(6, n_spam),
            'sender_reputation': np.random.beta(2, 8, n_spam),  # Mostly low
            'is_spam': np.ones(n_spam)
        }

    elif year == "2024":
        # 2024 spam patterns - EVOLVED!
        # Spammers got smarter: longer emails, fewer obvious tells
        spam_ratio = 0.35  # More spam overall
        n_spam = int(n_samples * spam_ratio)
        n_ham = n_samples - n_spam

        # Ham (similar to before, but more links due to modern email)
        ham_data = {
            'word_count': np.random.normal(180, 60, n_ham).clip(20, 600),
            'link_count': np.random.poisson(3, n_ham),  # More links are normal now
            'urgent_words': np.random.poisson(0.5, n_ham),
            'money_mentions': np.random.poisson(0.8, n_ham),
            'sender_reputation': np.random.beta(8, 2, n_ham),
            'is_spam': np.zeros(n_ham)
        }

        # Spam (2024 patterns: crypto scams, AI-generated, sophisticated)
        spam_data = {
            'word_count': np.random.normal(200, 70, n_spam).clip(50, 600),  # LONGER!
            'link_count': np.random.poisson(3, n_spam),  # FEWER links (less obvious)
            'urgent_words': np.random.poisson(2, n_spam),  # More subtle
            'money_mentions': np.random.poisson(3, n_spam),  # Crypto, investment
            'sender_reputation': np.random.beta(4, 6, n_spam),  # Better spoofed
            'is_spam': np.ones(n_spam)
        }

    # Combine ham and spam
    df = pd.DataFrame({
        'word_count': np.concatenate([ham_data['word_count'], spam_data['word_count']]),
        'link_count': np.concatenate([ham_data['link_count'], spam_data['link_count']]),
        'urgent_words': np.concatenate([ham_data['urgent_words'], spam_data['urgent_words']]),
        'money_mentions': np.concatenate([ham_data['money_mentions'], spam_data['money_mentions']]),
        'sender_reputation': np.concatenate([ham_data['sender_reputation'], spam_data['sender_reputation']]),
        'is_spam': np.concatenate([ham_data['is_spam'], spam_data['is_spam']])
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Generate 2020 data
data_2020 = generate_email_data(2000, year="2020")

print(f"Generated {len(data_2020)} emails from 2020")
print(f"Spam ratio: {data_2020['is_spam'].mean():.1%}")
print("\nFeature statistics (2020):")
print(data_2020.describe().round(2))

# Split into train/test
features = ['word_count', 'link_count', 'urgent_words', 'money_mentions', 'sender_reputation']
X_2020 = data_2020[features]
y_2020 = data_2020['is_spam']

X_train, X_test_2020, y_train, y_test_2020 = train_test_split(
    X_2020, y_2020, test_size=0.2, random_state=42, stratify=y_2020
)

print(f"\nTraining set: {len(X_train)} emails")
print(f"Test set (2020): {len(X_test_2020)} emails")

# =============================================================================
# STEP 2: Train the model on 2020 data
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Train Model on 2020 Data")
print("=" * 70)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on 2020 test set
y_pred_2020 = model.predict(X_test_2020)
accuracy_2020 = accuracy_score(y_test_2020, y_pred_2020)

print(f"\n✅ Model trained successfully")
print(f"\n2020 Test Set Performance:")
print(f"Accuracy: {accuracy_2020:.1%}")
print("\nClassification Report:")
print(classification_report(y_test_2020, y_pred_2020, target_names=['Ham', 'Spam']))

# Store baseline feature distributions for drift detection
baseline_stats = {
    feature: {
        'mean': X_train[feature].mean(),
        'std': X_train[feature].std(),
        'distribution': X_train[feature].values
    }
    for feature in features
}

print("📊 Baseline feature distributions saved for drift detection")

# =============================================================================
# STEP 3: Simulate data drift (2024 data arrives)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Simulate Data Drift - 2024 Data Arrives")
print("=" * 70)

# Generate 2024 data (spam patterns have evolved!)
data_2024 = generate_email_data(500, year="2024")

X_2024 = data_2024[features]
y_2024 = data_2024['is_spam']

print(f"Generated {len(data_2024)} emails from 2024")
print(f"Spam ratio: {data_2024['is_spam'].mean():.1%}")
print("\nFeature statistics (2024):")
print(data_2024.describe().round(2))

# =============================================================================
# STEP 4: Detect drift using statistical tests
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Detect Data Drift")
print("=" * 70)

def detect_drift(baseline_data, new_data, feature_name, alpha=0.05):
    """
    Use Kolmogorov-Smirnov test to detect distribution shift.

    Returns:
        tuple: (is_drifted, p_value, effect_size)
    """
    statistic, p_value = stats.ks_2samp(baseline_data, new_data)

    # Effect size: difference in means relative to baseline std
    mean_diff = abs(new_data.mean() - baseline_data.mean())
    effect_size = mean_diff / baseline_data.std() if baseline_data.std() > 0 else 0

    is_drifted = p_value < alpha

    return is_drifted, p_value, effect_size, statistic

print("\nDrift Detection Results (Kolmogorov-Smirnov Test, α=0.05):")
print("-" * 70)
print(f"{'Feature':<20} {'Drifted?':<10} {'p-value':<12} {'Effect Size':<12} {'KS Stat':<10}")
print("-" * 70)

drifted_features = []
for feature in features:
    baseline = baseline_stats[feature]['distribution']
    current = X_2024[feature].values

    is_drifted, p_value, effect_size, ks_stat = detect_drift(baseline, current, feature)

    status = "⚠️ YES" if is_drifted else "✓ No"

    print(f"{feature:<20} {status:<10} {p_value:<12.6f} {effect_size:<12.2f} {ks_stat:<10.3f}")

    if is_drifted:
        drifted_features.append(feature)

print("-" * 70)
print(f"\n🚨 {len(drifted_features)} features show significant drift: {drifted_features}")

# =============================================================================
# STEP 5: Observe performance degradation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Observe Performance Degradation")
print("=" * 70)

# Evaluate the 2020 model on 2024 data
y_pred_2024 = model.predict(X_2024)
accuracy_2024 = accuracy_score(y_2024, y_pred_2024)

print(f"\n2020 Model → 2024 Data:")
print(f"Accuracy: {accuracy_2024:.1%}")
print(f"\n📉 Accuracy dropped from {accuracy_2020:.1%} to {accuracy_2024:.1%}")
print(f"   Relative degradation: {((accuracy_2020 - accuracy_2024) / accuracy_2020 * 100):.1f}%")

print("\nClassification Report (2020 model on 2024 data):")
print(classification_report(y_2024, y_pred_2024, target_names=['Ham', 'Spam']))

# Analyze errors
print("\n🔍 Error Analysis:")
errors = data_2024[y_pred_2024 != y_2024]
false_negatives = errors[errors['is_spam'] == 1]  # Spam marked as ham
false_positives = errors[errors['is_spam'] == 0]  # Ham marked as spam

print(f"   False Negatives (missed spam): {len(false_negatives)}")
print(f"   False Positives (ham marked spam): {len(false_positives)}")

if len(false_negatives) > 0:
    print(f"\n   Missed spam characteristics:")
    print(f"   - Avg word count: {false_negatives['word_count'].mean():.0f} (2020 spam avg: ~80)")
    print(f"   - Avg link count: {false_negatives['link_count'].mean():.1f} (2020 spam avg: ~5)")
    print("   → 2024 spam is longer with fewer links - model wasn't trained for this!")

# =============================================================================
# STEP 6: Retrain to recover performance
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Retrain Model with 2024 Data")
print("=" * 70)

# Combine 2020 training data with 2024 data
X_combined = pd.concat([X_train, X_2024], ignore_index=True)
y_combined = pd.concat([y_train, y_2024], ignore_index=True)

print(f"Combined training set: {len(X_combined)} emails")
print(f"  - 2020 data: {len(X_train)} emails")
print(f"  - 2024 data: {len(X_2024)} emails")

# Retrain
model_retrained = LogisticRegression(random_state=42, max_iter=1000)
model_retrained.fit(X_combined, y_combined)

# Evaluate on new 2024 test data
data_2024_test = generate_email_data(200, year="2024")
X_2024_test = data_2024_test[features]
y_2024_test = data_2024_test['is_spam']

y_pred_retrained = model_retrained.predict(X_2024_test)
accuracy_retrained = accuracy_score(y_2024_test, y_pred_retrained)

print(f"\n✅ Retrained model performance on new 2024 data:")
print(f"Accuracy: {accuracy_retrained:.1%}")
print(f"\n📈 Accuracy recovered from {accuracy_2024:.1%} to {accuracy_retrained:.1%}")

# =============================================================================
# STEP 7: Visualize the drift
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Visualize Feature Drift (saving to drift_visualization.png)")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]

    # Plot 2020 distribution
    ax.hist(X_train[feature], bins=30, alpha=0.5, label='2020 (train)',
            density=True, color='blue')

    # Plot 2024 distribution
    ax.hist(X_2024[feature], bins=30, alpha=0.5, label='2024 (new)',
            density=True, color='red')

    ax.set_title(f'{feature}\n({"⚠️ DRIFTED" if feature in drifted_features else "✓ Stable"})')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend()

# Summary plot in last cell
ax = axes[-1]
ax.bar(['2020\nTest', '2024\n(before)', '2024\n(after)'],
       [accuracy_2020, accuracy_2024, accuracy_retrained],
       color=['green', 'red', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Over Time')
ax.set_ylim(0, 1)
for i, v in enumerate([accuracy_2020, accuracy_2024, accuracy_retrained]):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('drift_visualization.png', dpi=150, bbox_inches='tight')
print("📊 Saved visualization to drift_visualization.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Model Drift Detection Pipeline")
print("=" * 70)
print("""
What we demonstrated:

1. TRAINED a spam classifier on 2020 email patterns
   → Achieved {:.1%} accuracy on 2020 test data

2. SIMULATED DRIFT by generating 2024 data with evolved spam patterns:
   - Spam emails got longer (evading word count heuristics)
   - Fewer obvious spam indicators (links, urgent words)
   - Better sender reputation spoofing

3. DETECTED DRIFT using Kolmogorov-Smirnov statistical tests
   → Found {} features with significant distribution shift

4. OBSERVED DEGRADATION when applying old model to new data
   → Accuracy dropped to {:.1%} ({:.1f}% relative decrease)

5. RECOVERED PERFORMANCE by retraining on combined data
   → Accuracy restored to {:.1%}

KEY TAKEAWAYS:
• Monitor feature distributions continuously
• Set up alerts for statistical drift (KS test, PSI, etc.)
• Plan for regular retraining cycles
• Log predictions and ground truth for performance tracking
""".format(
    accuracy_2020,
    len(drifted_features),
    accuracy_2024,
    (accuracy_2020 - accuracy_2024) / accuracy_2020 * 100,
    accuracy_retrained
))
```

**Expected Output:**
```
======================================================================
STEP 1: Generate 2020 Training Data
======================================================================
Generated 2000 emails from 2020
Spam ratio: 30.0%

Feature statistics (2020):
       word_count  link_count  urgent_words  money_mentions  sender_reputation
count     2000.00     2000.00       2000.00         2000.00            2000.00
mean       128.45        2.38          1.42           2.15               0.65
std         54.32        2.15          1.89           2.54               0.24
...

Training set: 1600 emails
Test set (2020): 400 emails

======================================================================
STEP 2: Train Model on 2020 Data
======================================================================

✅ Model trained successfully

2020 Test Set Performance:
Accuracy: 91.2%

Classification Report:
              precision    recall  f1-score   support
         Ham       0.93      0.95      0.94       280
        Spam       0.88      0.83      0.85       120

📊 Baseline feature distributions saved for drift detection

======================================================================
STEP 3: Simulate Data Drift - 2024 Data Arrives
======================================================================
Generated 500 emails from 2024
Spam ratio: 35.0%

======================================================================
STEP 4: Detect Data Drift
======================================================================

Drift Detection Results (Kolmogorov-Smirnov Test, α=0.05):
----------------------------------------------------------------------
Feature              Drifted?   p-value      Effect Size  KS Stat
----------------------------------------------------------------------
word_count           ⚠️ YES     0.000001     0.85         0.234
link_count           ⚠️ YES     0.000023     0.42         0.189
urgent_words         ⚠️ YES     0.001245     0.38         0.156
money_mentions       ✓ No       0.089234     0.21         0.098
sender_reputation    ⚠️ YES     0.000089     0.52         0.201
----------------------------------------------------------------------

🚨 4 features show significant drift: ['word_count', 'link_count', 'urgent_words', 'sender_reputation']

======================================================================
STEP 5: Observe Performance Degradation
======================================================================

2020 Model → 2024 Data:
Accuracy: 76.4%

📉 Accuracy dropped from 91.2% to 76.4%
   Relative degradation: 16.2%

🔍 Error Analysis:
   False Negatives (missed spam): 89
   False Positives (ham marked spam): 29

   Missed spam characteristics:
   - Avg word count: 195 (2020 spam avg: ~80)
   - Avg link count: 2.8 (2020 spam avg: ~5)
   → 2024 spam is longer with fewer links - model wasn't trained for this!

======================================================================
STEP 6: Retrain Model with 2024 Data
======================================================================
Combined training set: 2100 emails
  - 2020 data: 1600 emails
  - 2024 data: 500 emails

✅ Retrained model performance on new 2024 data:
Accuracy: 88.5%

📈 Accuracy recovered from 76.4% to 88.5%

======================================================================
SUMMARY: Model Drift Detection Pipeline
======================================================================

What we demonstrated:

1. TRAINED a spam classifier on 2020 email patterns
   → Achieved 91.2% accuracy on 2020 test data

2. SIMULATED DRIFT by generating 2024 data with evolved spam patterns:
   - Spam emails got longer (evading word count heuristics)
   - Fewer obvious spam indicators (links, urgent words)
   - Better sender reputation spoofing

3. DETECTED DRIFT using Kolmogorov-Smirnov statistical tests
   → Found 4 features with significant distribution shift

4. OBSERVED DEGRADATION when applying old model to new data
   → Accuracy dropped to 76.4% (16.2% relative decrease)

5. RECOVERED PERFORMANCE by retraining on combined data
   → Accuracy restored to 88.5%

KEY TAKEAWAYS:
• Monitor feature distributions continuously
• Set up alerts for statistical drift (KS test, PSI, etc.)
• Plan for regular retraining cycles
• Log predictions and ground truth for performance tracking
```

**The Key Insight**: This example shows why production ML systems need continuous monitoring. The 2020 spam classifier worked great—until spammers evolved. Without drift detection, you wouldn't know your model was failing until users complained. With monitoring, you catch the problem early and retrain proactively.

**Production Implementation Notes**:
- Use a proper feature store (Feast, Tecton) to track feature distributions over time
- Implement Population Stability Index (PSI) for more nuanced drift detection
- Set up alerting thresholds based on your business tolerance
- Automate retraining pipelines with tools like Kubeflow or MLflow
- Always A/B test retrained models before full deployment

## Cost vs Accuracy Tradeoffs

Bigger models are more accurate. They're also more expensive. Production forces tradeoffs.

### The Cost Equation

```
Total cost = Training cost + Inference cost
```

**Training cost**: One-time (or periodic). GPU hours, data labeling, engineer time.

**Inference cost**: Ongoing. Every prediction costs compute, memory, latency.

At scale, inference cost dominates.

### Reducing Inference Cost

**1. Model distillation**: Train a small model to mimic a large model. "Student" learns from "teacher."

**2. Quantization**: Use 8-bit integers instead of 32-bit floats. 4x smaller, faster, tiny accuracy loss.

**3. Pruning**: Remove unimportant weights (set to zero). Sparse models are faster.

**4. Caching**: If 80% of queries are repeated, cache results.

**5. Smaller models**: GPT-4 is overkill for simple tasks. Use GPT-3.5-turbo, or even a fine-tuned BERT.

### When Accuracy Matters More

**High-stakes domains**: Medical diagnosis, legal contracts, autonomous vehicles. Pay for the best model.

**Low-stakes domains**: Product recommendations, ad targeting. Good enough is fine.

## When NOT to Use AI

This is the most important section.

### AI Is Not Always the Answer

**Use AI when**:
- The task is ambiguous, subjective, or requires pattern recognition
- You have lots of data
- You can tolerate some errors
- The rules are too complex to hand-code

**Don't use AI when**:
- A deterministic rule suffices
- You have <1000 labeled examples
- Errors are catastrophic
- You need to explain decisions precisely

### Examples: When NOT to Use AI

**Scenario 1: Input validation**
"Is this email address formatted correctly?"

❌ Train a classifier on valid/invalid emails.
✅ Use a regex.

**Scenario 2: Tax calculation**
"Calculate income tax based on IRS rules."

❌ Train a model on historical tax returns.
✅ Implement the tax code (it's deterministic).

**Scenario 3: High-stakes medical diagnosis with 100 labeled examples**
❌ Train a deep learning model.
✅ Use expert systems, or defer to human doctors.

### The Checklist

Before using AI, ask:

1. **Do I have enough data?** (<1k examples? Probably not enough for deep learning.)
2. **Is a rule-based system possible?** (If yes, start there.)
3. **Can I tolerate errors?** (If no, AI is risky.)
4. **Do I have the expertise to debug this?** (If no, you'll struggle in production.)
5. **Is the ROI positive?** (Will the model's value exceed training + deployment + maintenance costs?)

## War Story: Deleting an AI Feature Saved the Product

**The Setup**: A productivity app added an "AI assistant" to predict what task the user should do next. It used a neural network trained on user behavior.

**The Problem**:
- Users found the suggestions irrelevant 70% of the time.
- The model was slow (300ms latency), making the app feel sluggish.
- Maintaining the model required a dedicated ML engineer.

**The Data**:
- Usage metrics showed <5% of users clicked on AI suggestions.
- User feedback: "Just show me my task list, I don't need predictions."

**The Decision**: They deleted the AI feature.

**The Result**:
- App latency dropped to <50ms.
- User satisfaction increased (fewer distractions).
- Team could focus on core features.
- Removed ML infrastructure costs.

**The Lesson**: AI for the sake of AI is a trap. Only add AI if it solves a real user problem. Sometimes, the best AI is no AI.

## Things That Will Confuse You

### "We need AI to stay competitive"
Maybe. Or maybe your competitors are also wasting resources on AI that doesn't help users. Compete on value, not buzzwords.

### "Once we deploy, we're done"
Deployment is the beginning. Monitoring, retraining, and maintenance are ongoing.

### "AI will get better over time automatically"
No. Models don't improve without new data and retraining. Drift will degrade performance unless you actively maintain.

## Common Traps

**Trap #1: Deploying and forgetting**
Set up monitoring from day one. Production failures are inevitable.

**Trap #2: Optimizing for accuracy alone**
Optimize for the metric that matters: user satisfaction, revenue, latency, cost.

**Trap #3: Not planning for retraining**
Fresh data, retraining pipelines, versioning—all need to be in place before launch.

**Trap #4: Adding AI because it's trendy**
Ask: "What problem does this solve?" If the answer is vague, don't build it.

## Production Reality Check

AI in production:

- **Requires cross-functional teams**: Data engineers, ML engineers, backend engineers, DevOps, product managers.
- **Is never "done"**: Models drift, bugs emerge, users change behavior.
- **Costs real money**: Inference at scale is expensive. Optimize ruthlessly.
- **Fails in surprising ways**: Adversarial inputs, edge cases, data bugs. Test extensively.

## Build This Mini Project

**Goal**: Experience model drift firsthand.

**Task**: Train a model, simulate drift, observe failure.

1. **Train a spam classifier** on emails from 2020 (use a dated dataset, or simulate by filtering a dataset by date).

2. **Evaluate on 2020 test set**: Record accuracy (e.g., 90%).

3. **Simulate drift**: Take 2024 emails (or simulate by modifying features: add new keywords, change distributions).

4. **Evaluate on drifted data**: Watch accuracy drop (e.g., to 70%).

5. **Monitor**: Plot feature distributions (word frequencies, email length) for 2020 vs 2024. See the shift.

6. **Retrain**: Include 2024 data in training. Re-evaluate. Accuracy recovers.

**Key Insight**: Models are snapshots of data distributions at training time. When the world changes, models must be updated.

---

