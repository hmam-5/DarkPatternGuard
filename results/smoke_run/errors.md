# Error analysis

## Errors by model, class pair, and text length

### bilstm
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Normal       | Scarcity    |       5 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       5 |
| medium       |       0 |
| long         |       0 |

### cnn
No errors found.
### finetune_bert
| true_label   | predicted   |   count |
|:-------------|:------------|--------:|
| Urgency      | Normal      |       5 |

| length_bin   |   count |
|:-------------|--------:|
| short        |       5 |
| medium       |       0 |
| long         |       0 |

### finetune_deberta
No errors found.
### small_transformer
No errors found.
## Baseline failures recovered by novel models

| Alternative | ID | Original text | True label | BERT prediction | Alternative prediction |
| --- | --- | --- | --- | --- | --- |
| DeBERTa | 180 | Urgency | Normal | Urgency | (text unavailable in saved results) |
| DeBERTa | 184 | Urgency | Normal | Urgency | (text unavailable in saved results) |
| DeBERTa | 188 | Urgency | Normal | Urgency | (text unavailable in saved results) |
| DeBERTa | 192 | Urgency | Normal | Urgency | (text unavailable in saved results) |
| DeBERTa | 196 | Urgency | Normal | Urgency | (text unavailable in saved results) |
