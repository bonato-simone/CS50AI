import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    res = {}

    pages_number = len(corpus)
    base_probability = (1 - damping_factor) / pages_number

    links = corpus[page]
    links_number = len(links)

    if not links_number:
        probability = 1 / damping_factor
        for p in corpus:
            res[p] = probability
    else:
        for p in corpus:
            res[p] = base_probability

        added_probability = damping_factor / links_number

        for p in links:
            res[p] += added_probability

    return res


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    res = {}

    for p in corpus:
        res[p] = 0

    page = random.choice(list(corpus))

    for _ in range(n):
        res[page] += 1
        tm = transition_model(corpus, page, damping_factor)

        random_value = random.random()
        prob_limit = 0
        next_page = 0

        for p, prob in tm.items():
            prob_limit += prob
            if random_value <= prob_limit:
                next_page = p
                break

        page = next_page

    for p in res:
        res[p] /= n

    return res


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    res = {}
    starting_probability = 1 / len(corpus)

    for p in corpus:
        res[p] = [starting_probability, 0]

    base_probability = (1 - damping_factor) / len(corpus)

    while True:
        for p1 in res:
            added_probability = 0
            for p2 in corpus:
                if p1 in corpus[p2]:
                    added_probability += res[p2][0] / len(corpus[p2])
            res[p1][1] = base_probability + damping_factor * added_probability

        if all(abs(res[p][0] - res[p][1]) < 0.001 for p in res):
            break

        for p in res:
            res[p][0] = res[p][1]

    total = sum(v[1] for v in res.values())
    return {key: value[1] / total for key, value in res.items()}


if __name__ == "__main__":
    main()
