import os
from twilio.rest import Client


def send_text(to_number: str, message: str) -> None:
    """Send an SMS to the given phone number using Twilio.

    Args:
        to_number: Recipient phone number in E.164 format (e.g. +15551234567).
        message: Text body to send.
    """
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_FROM_NUMBER"]

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=from_number,
        to=to_number,
    )
    print(f"Message sent to {to_number}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python send_text.py <to_number> <message>")
        print('Example: python send_text.py +15551234567 "Hello!"')
        sys.exit(1)

    send_text(to_number=sys.argv[1], message=sys.argv[2])
