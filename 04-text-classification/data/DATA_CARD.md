# Data card: 20 Newsgroups

## Source
The 20 Newsgroups collection, distributed with scikit-learn (`sklearn.datasets.fetch_20newsgroups`). Usenet posts from the 1990s across 20 newsgroups, using the date-sorted "bydate" distribution prepared by Jason Rennie.
- Source page: http://qwone.com/~jason/20Newsgroups/
- Scikit-learn documentation: https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

Reference: Lang, K. "Newsweeder: Learning to filter netnews." Proceedings of the 12th International Conference on Machine Learning, 1995, pp. 331-339.

## Terms
Historical Usenet postings from 1995 collected for research. No explicit single license applies to all posts. Verify the exact terms on the source page before redistributing anything derived from it. The data is **not** committed. `python -m text_classification.download_data` fetches it through scikit-learn and writes one CSV with two versions of each post.

## Shape
Documented by scikit-learn: 18,846 posts, 11,314 train and 7,532 test, 20 classes. `text_classification/config.py` enforces these counts and `text_classification/data.py` enforces the schema and that every class appears in both splits.

| Column | Meaning |
|---|---|
| text | Post body with headers, footers, and quoted replies removed |
| text_raw | The same post with them kept. Used only for the metadata trap |
| label | Newsgroup name (target) |
| split | `train` or `test` (the official split) |

## Known quirks
- **Metadata identifies the poster.** Headers carry names, organisations, and thread paths. A model can score well by reading them. This is the central trap of the project.
- **Quoted replies repeat other posts.** A quote can put text from one post inside another, so near-duplicate content crosses documents and splits.
- **Crossposting and related groups.** Several newsgroups share topics (for example, two computer hardware groups and two religion groups), so confusion between them is expected and not always an error.
- **Empty posts after cleaning.** Removing metadata can leave an empty body. These are kept and counted, and `results/error_analysis.json` reports how the model does on them.
- **Old and informal text.** 1990s Usenet style with typos and signatures. Results do not transfer to current text.

## Not checked
- How the newsgroup labels were assigned beyond the group a post appeared in.
- Whether the date-based split fully prevents quoted text from crossing into test.
- Offensive or personal content in the posts. Review the data before showing examples publicly.
