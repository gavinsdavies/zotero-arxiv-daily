"""
Keyword filter for HEP-EX neutrino oscillation papers
Pre-filters papers before expensive LLM processing to reduce computation time
"""

# Customize these keywords for your specific research interests
HEP_EX_KEYWORDS = [
    # Neutrino Oscillations
    "neutrino oscillation", "neutrino oscillations", "oscillation parameter",
    "mixing angle", "mass hierarchy", "mass ordering",
    "CP violation neutrino", "CP phase", "delta CP",
    
    # Neutrino Types and Interactions
    "electron neutrino", "muon neutrino", "tau neutrino",
    "sterile neutrino", "neutrino flavor",
    "neutrino beam", "neutrino interaction",
    "charged current", "neutral current",
    
    # Experiments
    "T2K", "NOvA", "DUNE", "Hyper-Kamiokande", "Hyper-K",
    "Super-Kamiokande", "Super-K", "IceCube",
    "KamLAND", "Daya Bay", "RENO", "Double Chooz",
    "MicroBooNE", "SBND", "ICARUS",
    "JUNO", "SNO", "Borexino",
    
    # Physics Phenomena
    "atmospheric neutrino", "solar neutrino", "reactor neutrino",
    "accelerator neutrino", "supernova neutrino",
    "theta13", "theta23", "theta12",
    "delta m", "mass splitting", "mass difference",
    "normal hierarchy", "inverted hierarchy",
    "octant", "maximal mixing",
    
    # Experimental Techniques
    "long baseline", "short baseline",
    "neutrino detector", "water Cherenkov",
    "liquid argon", "scintillator detector",
    "event selection", "neutrino energy reconstruction",
    
    # Physics Processes
    "disappearance", "appearance",
    "electron neutrino appearance", "muon neutrino disappearance",
    "PMNS", "Pontecorvo-Maki-Nakagawa-Sakata",
    "matter effect", "MSW effect",
]

# Optional: High-priority keywords that boost relevance
HIGH_PRIORITY_KEYWORDS = [
    "CP violation", "mass hierarchy", "mass ordering",
    "DUNE", "Hyper-Kamiokande", "T2K", "NOvA",
    "neutrino oscillation", "mixing angle",
    "theta23 octant", "delta CP"
]

def should_process_paper(title: str, abstract: str) -> bool:
    """
    Check if paper matches HEP-EX keywords
    
    Args:
        title: Paper title
        abstract: Paper abstract
        
    Returns:
        True if paper should be processed, False otherwise
    """
    if not title and not abstract:
        return False
        
    text = (title + " " + abstract).lower()
    
    # Check if any keyword matches
    return any(keyword.lower() in text for keyword in HEP_EX_KEYWORDS)

def calculate_keyword_bonus(title: str, abstract: str) -> float:
    """
    Calculate relevance bonus based on high-priority keyword matches
    
    Args:
        title: Paper title
        abstract: Paper abstract
        
    Returns:
        Bonus score between 0.0 and 0.5
    """
    text = (title + " " + abstract).lower()
    
    matches = sum(1 for keyword in HIGH_PRIORITY_KEYWORDS 
                  if keyword.lower() in text)
    
    # Each match adds 0.1, capped at 0.5
    return min(matches * 0.1, 0.5)

def filter_papers(papers: list) -> tuple[list, int, dict]:
    """
    Filter papers by keywords and calculate statistics
    
    Args:
        papers: List of arxiv.Result objects or similar
        
    Returns:
        Tuple of (filtered_papers, filtered_count, stats)
        stats contains keyword match information
    """
    filtered = []
    stats = {
        'total': len(papers),
        'matched': 0,
        'filtered': 0,
        'keyword_matches': {}
    }
    
    for paper in papers:
        title = paper.title if hasattr(paper, 'title') else str(paper)
        abstract = paper.summary if hasattr(paper, 'summary') else ""
        
        if should_process_paper(title, abstract):
            filtered.append(paper)
            stats['matched'] += 1
            
            # Track which keywords matched (for debugging/optimization)
            text = (title + " " + abstract).lower()
            for keyword in HEP_EX_KEYWORDS:
                if keyword.lower() in text:
                    stats['keyword_matches'][keyword] = stats['keyword_matches'].get(keyword, 0) + 1
    
    stats['filtered'] = stats['total'] - stats['matched']
    
    return filtered, stats['filtered'], stats

def print_filter_stats(stats: dict):
    """
    Print filtering statistics
    
    Args:
        stats: Statistics dictionary from filter_papers
    """
    print(f"\n{'='*60}")
    print(f"Keyword Filter Statistics")
    print(f"{'='*60}")
    print(f"Total papers:     {stats['total']}")
    print(f"Matched papers:   {stats['matched']}")
    print(f"Filtered papers:  {stats['filtered']}")
    print(f"Filter rate:      {stats['filtered']/stats['total']*100:.1f}%")
    
    if stats['keyword_matches']:
        print(f"\nTop matching keywords:")
        sorted_matches = sorted(stats['keyword_matches'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for keyword, count in sorted_matches:
            print(f"  - {keyword}: {count} papers")
    
    print(f"{'='*60}\n")

# Example usage (for testing)
if __name__ == "__main__":
    # Test with mock papers
    class MockPaper:
        def __init__(self, title, summary):
            self.title = title
            self.summary = summary
    
    test_papers = [
        MockPaper(
            "Measurement of neutrino oscillation parameters from muon neutrino disappearance with an off-axis beam",
            "We present measurements of neutrino oscillation parameters using T2K data. The analysis uses muon neutrino disappearance to constrain the mixing angle theta23 and mass splitting delta m^2_32. Results favor the normal mass ordering with CP phase delta_CP near maximal violation..."
        ),
        MockPaper(
            "Deep learning for particle physics",
            "We describe a neural network approach for general particle physics applications..."
        ),
        MockPaper(
            "Improved constraints on sterile neutrino mixing from disappearance searches at the NOvA experiment",
            "The NOvA experiment searches for sterile neutrino mixing through electron neutrino and muon neutrino disappearance channels. Using long baseline neutrino beam data, we set new limits on the 3+1 sterile neutrino model parameters..."
        ),
        MockPaper(
            "First results on solar neutrino oscillations from the JUNO experiment",
            "JUNO is a reactor neutrino experiment designed to determine the neutrino mass ordering. We report first measurements of solar neutrino oscillations observing 8B neutrinos, providing complementary constraints on the mixing angle theta12..."
        ),
    ]
    
    filtered, filtered_count, stats = filter_papers(test_papers)
    print_filter_stats(stats)
    print(f"Filtered papers: {len(filtered)}")
    for paper in filtered:
        print(f"  - {paper.title}")