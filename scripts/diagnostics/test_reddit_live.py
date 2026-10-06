import os
import sys
from dotenv import load_dotenv
import praw

def main():
    print("Testing live Reddit access...")
    
    # Load environment
    load_dotenv()
    
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "TrustLens/0.1")
    
    if not client_id or not client_secret:
        print("[ERROR] Credentials missing in .env.")
        sys.exit(1)
        
    print("[OK] Environment variables found (hidden).")
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
        reddit.read_only = True
        
        # We need a URL to test fetching
        url = input("Enter a valid Reddit submission URL to fetch: ").strip()
        if not url:
            print("[INFO] No URL provided. Exiting.")
            sys.exit(0)
            
        submission = reddit.submission(url=url)
        print(f"[OK] Fetched submission: {submission.title[:50]}...")
        
        # Expand comments
        submission.comments.replace_more(limit=0)
        comments = submission.comments.list()
        print(f"[OK] Fetched {len(comments)} comments.")
        
        print("\nReddit access test passed.")
        sys.exit(0)
        
    except Exception as e:
        print(f"[ERROR] Live testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
