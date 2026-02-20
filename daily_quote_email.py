"""
daily_quote_email.py

Fetches a random inspirational quote and emails it to mom every morning.

Setup:
  1. Copy .env.example to .env and fill in your values.
  2. Install dependencies: pip install -r requirements.txt
  3. Run once to test: python daily_quote_email.py
  4. Schedule with cron to run every morning at 8 AM:
       crontab -e
       0 8 * * * /usr/bin/python3 /path/to/daily_quote_email.py
"""

import json
import os
import smtplib
import urllib.request
import random
from datetime import date
from email.mime.text import MIMEText

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SENDER_EMAIL = os.environ["SENDER_EMAIL"]
SENDER_APP_PASSWORD = os.environ["SENDER_APP_PASSWORD"]
MOM_EMAIL = os.environ["MOM_EMAIL"]

# Used if the ZenQuotes API is unavailable
FALLBACK_QUOTES = [
    ("The only way to do great work is to love what you do.", "Steve Jobs"),
    ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
    ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
    ("Life is what happens when you're busy making other plans.", "John Lennon"),
    ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt"),
    ("Spread love everywhere you go. Let no one ever come to you without leaving happier.", "Mother Teresa"),
    ("When you reach the end of your rope, tie a knot in it and hang on.", "Franklin D. Roosevelt"),
    ("Always remember that you are absolutely unique. Just like everyone else.", "Margaret Mead"),
    ("Do not go where the path may lead, go instead where there is no path and leave a trail.", "Ralph Waldo Emerson"),
    ("You will face many defeats in life, but never let yourself be defeated.", "Maya Angelou"),
]


def get_quote():
    """Fetch a random quote from ZenQuotes API, falling back to a local list."""
    try:
        url = "https://zenquotes.io/api/random"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
        return data[0]["q"], data[0]["a"]
    except Exception:
        return random.choice(FALLBACK_QUOTES)


def send_email(quote, author):
    """Send the daily quote to mom via Gmail SMTP."""
    today = date.today().strftime("%B %d, %Y")
    subject = f"Good Morning! Your Daily Quote - {today}"

    body = (
        f"Good morning!\n\n"
        f'Here is your inspirational quote for today:\n\n'
        f'"{quote}"\n\n'
        f"  — {author}\n\n"
        f"Have a wonderful day!\n"
    )

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = MOM_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, MOM_EMAIL, msg.as_string())

    print(f"Quote emailed to {MOM_EMAIL}: \"{quote}\" — {author}")


if __name__ == "__main__":
    quote, author = get_quote()
    send_email(quote, author)
