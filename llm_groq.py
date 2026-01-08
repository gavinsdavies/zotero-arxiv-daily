"""
Optional Groq API integration for faster TLDR generation
Falls back to default method if not configured or if request fails
"""
import os
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-70b-versatile"  # Fast and high quality
GROQ_TIMEOUT = 30  # seconds

def is_groq_available() -> bool:
    """
    Check if Groq API is configured AND enabled
    
    Returns:
        True if GROQ_API_KEY is set AND USE_GROQ is enabled
    """
    use_groq = os.environ.get("USE_GROQ", "0").lower() in ["1", "true", "yes"]
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    
    if use_groq and not has_key:
        logger.warning("USE_GROQ is enabled but GROQ_API_KEY is not set")
        return False
    
    return use_groq and has_key

def generate_tldr_groq(
    title: str, 
    abstract: str, 
    intro: str = "", 
    conclusion: str = "", 
    language: str = "English"
) -> Optional[str]:
    """
    Generate TLDR using Groq API
    
    Args:
        title: Paper title
        abstract: Paper abstract
        intro: Introduction text (optional, will be truncated if too long)
        conclusion: Conclusion text (optional, will be truncated if too long)
        language: Language for TLDR generation
        
    Returns:
        TLDR string if successful, None if Groq not configured or request fails
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_key:
        logger.debug("Groq API key not found, skipping Groq TLDR generation")
        return None
    
    # Truncate long sections to stay within token limits
    intro_truncated = intro[:1500] if intro else ""
    conclusion_truncated = conclusion[:800] if conclusion else ""
    
    # Construct prompt
    prompt = f"""Given the title, abstract, introduction and conclusion of a scientific paper, generate a one-sentence TLDR summary in {language}.

The TLDR should:
- Be exactly ONE sentence
- Capture the main contribution or finding
- Be clear and concise
- Use technical terms appropriately but remain accessible

Paper Information:

Title: {title}

Abstract: {abstract}

{f"Introduction: {intro_truncated}" if intro_truncated else ""}

{f"Conclusion: {conclusion_truncated}" if conclusion_truncated else ""}

Provide ONLY the TLDR sentence, nothing else. No preamble, no explanation."""

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at summarizing scientific papers concisely."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 200,
                "top_p": 1,
                "stream": False
            },
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            tldr = result["choices"][0]["message"]["content"].strip()
            
            # Clean up any markdown or extra formatting
            tldr = tldr.replace("**", "").replace("*", "")
            
            # Remove common prefixes if present
            prefixes = ["TLDR:", "TL;DR:", "Summary:", "In short:"]
            for prefix in prefixes:
                if tldr.startswith(prefix):
                    tldr = tldr[len(prefix):].strip()
            
            logger.info(f"Generated TLDR via Groq API: {len(tldr)} chars")
            return tldr
            
        elif response.status_code == 429:
            logger.warning("Groq API rate limit reached, falling back to default method")
            return None
        else:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning("Groq API request timed out, falling back to default method")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Groq TLDR generation: {e}")
        return None

def generate_tldr_with_fallback(
    title: str,
    abstract: str,
    intro: str = "",
    conclusion: str = "",
    language: str = "English",
    fallback_function = None
) -> str:
    """
    Generate TLDR with automatic fallback to default method
    
    Args:
        title: Paper title
        abstract: Paper abstract
        intro: Introduction text
        conclusion: Conclusion text
        language: Language for TLDR
        fallback_function: Function to call if Groq fails (should have same signature)
        
    Returns:
        TLDR string (either from Groq or fallback method)
    """
    # Try Groq first if available
    if is_groq_available():
        logger.debug("Attempting TLDR generation with Groq API")
        tldr = generate_tldr_groq(title, abstract, intro, conclusion, language)
        
        if tldr:
            return tldr
        else:
            logger.info("Groq TLDR generation failed or unavailable, using fallback")
    
    # Fall back to provided function or simple default
    if fallback_function:
        return fallback_function(title, abstract, intro, conclusion, language)
    else:
        # Simple fallback if no function provided
        logger.warning("No fallback function provided, using simple abstract truncation")
        return abstract[:200] + "..." if len(abstract) > 200 else abstract

def get_groq_stats() -> dict:
    """
    Get statistics about Groq API availability and usage
    
    Returns:
        Dictionary with Groq configuration info
    """
    return {
        "enabled": os.environ.get("USE_GROQ", "0").lower() in ["1", "true", "yes"],
        "configured": bool(os.environ.get("GROQ_API_KEY")),
        "available": is_groq_available(),
        "model": GROQ_MODEL,
        "api_url": GROQ_API_URL,
        "timeout": GROQ_TIMEOUT
    }

# Example usage
if __name__ == "__main__":
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample paper
    test_title = "Measurement of neutrino oscillation parameters from muon neutrino disappearance with an off-axis beam"
    test_abstract = """We present measurements of neutrino oscillation parameters using 
    data from the T2K experiment. The analysis uses muon neutrino disappearance in the 
    off-axis neutrino beam to constrain the atmospheric mixing angle theta23 and the 
    atmospheric mass splitting delta m^2_32. Results favor the normal mass ordering at 
    2.7 sigma significance, with the CP-violating phase delta_CP near maximal violation. 
    The best-fit values are sin^2(theta23) = 0.53 and |delta m^2_32| = 2.45 x 10^-3 eV^2."""
    
    print("Testing Groq TLDR generation...")
    print(f"Groq configured: {is_groq_available()}")
    
    if is_groq_available():
        tldr = generate_tldr_groq(test_title, test_abstract)
        if tldr:
            print(f"\nGenerated TLDR:\n{tldr}")
        else:
            print("\nGroq generation failed")
    else:
        print("\nGroq API key not configured. Set GROQ_API_KEY environment variable to test.")
    
    print(f"\nGroq stats: {get_groq_stats()}")