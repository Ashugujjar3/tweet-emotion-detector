```markdown
# Document Similarity Finder — Run Report

## Date:
YYYY-MM-DD

## Corpus
- Folder: `data/docs/`
- # documents: 
- Filenames:

## Environment
- Python version:
- Packages (from `pip freeze`):

## Vectorizer & parameters
- TF-IDF params: stop_words='english', max_features=..., ngram_range=(..., ...)

## Results
### Top similar pairs (from `outputs/top_pairs.csv`)
| rank | docA | docB | cosine |
|------|------|------:|------:|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

### Similarity heatmap
- File: `outputs/similarity_matrix.png`

### Top TF-IDF terms for top pair
- File: `outputs/top_terms_top_pair.json`
- Short summary of why these two documents are similar:

## Observations
- Patterns found:
- Any duplicates or near-duplicates:
- Recommendations (e.g., use embeddings for semantic similarity):

## Next steps
- Try sentence/paragraph embeddings (SBERT) for semantic similarity.
- Cluster documents using hierarchical or KMeans on vectors.
