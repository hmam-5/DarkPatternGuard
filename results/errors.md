# Error analysis

## Errors by model, class pair, and text length

### bilstm
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Social Proof |       4 |
| Normal       | Urgency      |       5 |
| Scarcity     | Urgency      |       3 |
| Social Proof | Normal       |       3 |
| Social Proof | Urgency      |       1 |
| Urgency      | Normal       |       2 |

| length_bin   |   count |
|:-------------|--------:|
| short        |      10 |
| medium       |       3 |
| long         |       5 |

### cnn
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Social Proof |       1 |
| Scarcity     | Normal       |       1 |
| Scarcity     | Urgency      |       2 |
| Social Proof | Normal       |       2 |
| Urgency      | Normal       |       2 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       7 |
| medium       |       1 |
| long         |       0 |

### ensemble_classical_only
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Social Proof |       1 |
| Scarcity     | Normal       |       1 |
| Scarcity     | Urgency      |       1 |
| Social Proof | Normal       |       3 |
| Urgency      | Normal       |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       5 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_bilstm
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_cnn
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       2 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       3 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_finetune_bert
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_finetune_deberta
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_lightgbm
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_linear_svm
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_logistic_regression
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       2 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       3 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_naive_bayes
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       2 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       3 |
| medium       |       2 |
| long         |       0 |

### ensemble_leave_one_out_small_transformer
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       2 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       3 |
| medium       |       2 |
| long         |       0 |

### ensemble_neural_only
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Urgency     |       2 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_soft_voting
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_stacking
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Urgency     |       2 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### ensemble_weighted_average
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Urgency     |       2 |
| Social Proof | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       2 |
| medium       |       1 |
| long         |       0 |

### ensemble_without_llm
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       3 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       2 |
| long         |       0 |

### finetune_bert
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Urgency     |       2 |
| Social Proof | Normal      |       1 |
| Urgency      | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       2 |
| medium       |       2 |
| long         |       0 |

### finetune_deberta
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Scarcity     | Urgency     |       2 |
| Social Proof | Normal      |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       2 |
| medium       |       1 |
| long         |       0 |

### lightgbm
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Scarcity     |       2 |
| Scarcity     | Social Proof |       2 |
| Urgency      | Normal       |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       4 |
| medium       |       1 |
| long         |       0 |

### linear_svm
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Social Proof |       2 |
| Scarcity     | Normal       |       1 |
| Scarcity     | Urgency      |       1 |
| Social Proof | Normal       |       1 |
| Social Proof | Urgency      |       1 |
| Urgency      | Normal       |       1 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       5 |
| medium       |       2 |
| long         |       0 |

### logistic_regression
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Social Proof |       1 |
| Scarcity     | Normal       |       2 |
| Social Proof | Normal       |       5 |
| Urgency      | Normal       |       3 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       8 |
| medium       |       3 |
| long         |       0 |

### naive_bayes
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Normal       | Scarcity    |       1 |
| Scarcity     | Normal      |       1 |
| Scarcity     | Urgency     |       1 |
| Social Proof | Normal      |       4 |
| Urgency      | Normal      |       2 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       5 |
| medium       |       3 |
| long         |       1 |

### small_transformer
| true_label   | predicted    |   count |
|:-------------|:-------------|--------:|
| Normal       | Scarcity     |       1 |
| Normal       | Social Proof |       1 |
| Normal       | Urgency      |       1 |
| Scarcity     | Normal       |       1 |
| Scarcity     | Urgency      |       2 |
| Social Proof | Normal       |       4 |
| Urgency      | Normal       |       2 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       8 |
| medium       |       3 |
| long         |       1 |

## Baseline failures recovered by novel models

| Alternative | ID | Original text | True label | BERT prediction | Alternative prediction | Seed(s) |
| --- | --- | --- | --- | --- | --- | --- |
| DeBERTa | 23 | long dress style beach cover up with lace detail jessica in metairie, united states | Social Proof | Normal | Social Proof | 1729, 2718, 3141 |
| DeBERTa | 57 | spring sale extended | Urgency | Normal | Urgency | 1729, 2718, 3141 |
| DeBERTa | 0 | limited availability | Scarcity | Urgency | Scarcity | 3141 |
| DeBERTa | 73 | extremely high demand: expect 2-4 weeks for it to arrive (to be safe). limit 4 per person! | Scarcity | Urgency | Scarcity | 3141 |
| DeBERTa | 171 | an item you ordered is in high demand. no worries, we have prioritized your order. | Scarcity | Urgency | Scarcity | 3141 |
